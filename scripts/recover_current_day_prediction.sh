#!/usr/bin/env bash
set -euo pipefail

repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
workflow_file="on-demand-production-prediction.yml"
workflow_name="On-demand production prediction"
current_sha="$(gh api "repos/$repo/git/ref/heads/main" --jq '.object.sha')"
now_epoch="$(date -u +%s)"
day_jst="$(TZ=Asia/Tokyo date '+%Y-%m-%d')"
weekday="$(TZ=Asia/Tokyo date '+%u')"
hour_jst="$(TZ=Asia/Tokyo date '+%H')"
minute_jst="$(TZ=Asia/Tokyo date '+%M')"
local_minutes="$((10#${hour_jst} * 60 + 10#${minute_jst}))"

# Current-day prediction is a daytime recovery lane. It is intentionally
# inactive after 11:00 JST because the canonical evening market-cycle/watchlist
# lane owns the next-session prediction thereafter.
if [ "$weekday" -ge 6 ] || [ "$local_minutes" -lt 420 ] || [ "$local_minutes" -ge 660 ]; then
  echo "$workflow_name: outside current-day recovery window (JST=${hour_jst}:${minute_jst})"
  exit 0
fi

runs="$(gh api "repos/$repo/actions/workflows/$workflow_file/runs?per_page=50")"

# Accept only runs created today on the live main SHA. Older SHA executions are
# stale evidence and cannot satisfy current-day availability.
today_start_epoch="$(TZ=Asia/Tokyo date -d "$day_jst 00:00:00" +%s)"
run_rows="$(printf '%s' "$runs" | jq -c --arg sha "$current_sha" --argjson start "$today_start_epoch" '
  [ .workflow_runs[]
    | select(.head_sha == $sha)
    | select((.created_at | fromdateiso8601) >= $start)
  ]
  | sort_by(.run_number)
  | reverse
')"

active_count="$(printf '%s' "$run_rows" | jq '[.[] | select(.status == "queued" or .status == "pending" or .status == "waiting" or .status == "requested" or .status == "in_progress")] | length')"
if [ "$active_count" -gt 0 ]; then
  echo "$workflow_name: current-main run already active (count=$active_count)"
  exit 0
fi

success_count="$(printf '%s' "$run_rows" | jq '[.[] | select(.status == "completed" and .conclusion == "success")] | length')"
if [ "$success_count" -gt 0 ]; then
  echo "$workflow_name: current-day current-main run completed successfully"
  exit 0
fi

latest_id="$(printf '%s' "$run_rows" | jq -r '.[0].id // empty')"
latest_attempt="$(printf '%s' "$run_rows" | jq -r '.[0].run_attempt // 1')"
latest_conclusion="$(printf '%s' "$run_rows" | jq -r '.[0].conclusion // empty')"
latest_created="$(printf '%s' "$run_rows" | jq -r '.[0].created_at // empty')"
latest_created_epoch="$(date -d "$latest_created" +%s 2>/dev/null || echo 0)"

# First completed failure: bounded retry of the failed jobs/attempt.
if [ -n "$latest_id" ] && [ "$latest_conclusion" = "failure" ] && [ "$latest_attempt" -lt 2 ]; then
  echo "::warning title=$workflow_name first failure::rerunning failed jobs for run $latest_id"
  if gh run rerun "$latest_id" --repo "$repo" --failed; then
    echo "$workflow_name: bounded failure recovery requested"
    exit 0
  fi
  echo "::error title=$workflow_name retry failed::unable to rerun failed jobs for run $latest_id"
  exit 1
fi

# Once the daytime recovery threshold is reached, absence of any successful
# current-main run is treated as a scheduler miss. Dispatch exactly one fresh
# current-main execution; the workflow itself has a latest-main guard.
recovery_after="$((today_start_epoch + 8 * 3600 + 20 * 60))" # 08:20 JST
if [ "$now_epoch" -ge "$recovery_after" ]; then
  echo "::warning title=$workflow_name missing current-day run::no successful current-main run exists by 08:20 JST; dispatching recovery"
  if gh workflow run "$workflow_file" --repo "$repo" --ref main; then
    echo "$workflow_name: current-day scheduler recovery dispatched"
    exit 0
  fi
  echo "::error title=$workflow_name scheduler recovery failed::unable to dispatch current-main prediction"
  exit 1
fi

if [ -n "$latest_created_epoch" ] && [ "$latest_created_epoch" -gt 0 ]; then
  echo "$workflow_name: latest current-main run is not yet successful; waiting inside pre-recovery window"
else
  echo "$workflow_name: no current-day current-main run yet; waiting until 08:20 JST recovery threshold"
fi
