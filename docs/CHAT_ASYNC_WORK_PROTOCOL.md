# Chat asynchronous work protocol

このリポジトリでChatGPTから長い作業を依頼するときは、チャット側で重いActionsの完走を待たない。

1. `ops/chat_queue/<task_id>.request.json` を1回だけmainへ追加する。
2. `Chat request handoff` が3分以内にJSONを検証してartifactへ固定する。
3. `Chat request worker` がhandoff完了後に別runとして非同期実行する。
4. ChatGPT側はworkerの完走を待たず、task_idとGitHub Actions runを返す。
5. 結果確認時のみworker run/artifactを再取得する。

実行可能タスクはallowlist制で、任意shellコマンドは受け付けない。最大実行時間は50分、worker job自体は55分で強制終了する。production変更を伴うタスクはこの経路へ追加せず、既存のrelease gateを使用する。
