#!/usr/bin/env bash
set -euo pipefail

workflow_file="daily-watchlist.yml"
workflow_name="Daily priority watchlist"
repo="${GITHUB_REPOSITORY:?GITHUB_REPOSITORY is required}"
now_epoch="$(date -u +%s)"
day_jst="$(TZ=Asia/Tokyo date +%Y-%m-%d)"
weekday="$(TZ=Asia/Tokyo date +%u)"

# Run only on JP/US weekday scheduling days and only after the canonical 18:50 JST
# schedule has had a 45-minute grace period.
if [ "${weekday}" -ge 6 ]; then
  exit 0
fi

recovery_after_epoch="$(TZ=Asia/Tokyo date -d "${day_jst} 19:35:00" +%s)"
if [ "${now_epoch}" -lt "${recovery_after_epoch}" ]; then
  exit 0
fi

window_start_epoch="$(TZ=Asia/Tokyo date -d "${day_jst} 18:40:00" +%s)"

runs="$(gh api "repos/${repo}/actions/workflows/${workflow_file}/runs?per_page=20" 2>/dev/null)" || {
  echo "::error title=${workflow_name} schedule-miss API unavailable::unable to inspect today's Daily Priority Watchlist runs"
  exit 1
}

scheduled_window_runs="$(
  printf '%s' "${runs}" | jq --argjson start "${window_start_epoch}" --argjson now "${now_epoch}" '
    [ .workflow_runs[]
      | select(.created_at != null)
      | ((.created_at | fromdateiso8601)) as $created
      | select($created >= $start and $created <= $now)
    ] | length
'
)"

if [ "${scheduled_window_runs}" -eq 0 ]; then
  echo "::warning title=${workflow_name} schedule miss::no Daily Priority Watchlist run was created in today's 18:40 JST onward schedule/recovery window; dispatching one current-main recovery"
  if gh workflow run "${workflow_file}" --repo "${repo}" --ref main; then
    echo "${workflow_name}: missed-schedule recovery dispatched on current main"
    if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
      {
        echo "## Daily priority watchlist schedule recovery"
        echo "- Action: MISSED_SCHEDULE_RECOVERY"
        echo "- Date (JST): ${day_jst}"
        echo "- Expected schedule: 18:50 JST"
        echo "- Recovery threshold: 19:35 JST"
        echo "- Existing runs in schedule window: 0"
      } >> "${GITHUB_STEP_SUMMARY}"
    fi
  else
    echo "::error title=${workflow_name} schedule-miss recovery failed::unable to dispatch current-main Daily Priority Watchlist"
    exit 1
  fi
else
  echo "${workflow_name}: today's scheduled/recovery run exists; no missed-schedule dispatch needed"
fi
