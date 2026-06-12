# LEARNINGS

Distilled, verified facts and rules for this project. Read this before working; do not re-derive.

## Environment

- **[verified 2026-06-12]** podman は使用不可：`podman machine start` は成功するが、イメージ `localhost/2d_pinn` がマシン内に存在せず、リポジトリに Dockerfile も無いため再ビルド不可。
  **Rule:** 実行は venv を使う — `C:\Projects\venv\Scripts\python.exe`（Python 3.13.13, numpy 2.4.4, scipy 1.17.1, torch 2.9.1+rocm, `torch.cuda.is_available()==True`, pytest 9.0.3）。RUBRIC の podman コマンドは venv 相当に読み替える。
- **[verified 2026-06-12]** 周波数ビン数：50〜8000 Hz を 10 Hz 刻みで生成すると (8000−50)/10+1 = **796** ビン。RUBRIC 初版の「797」は計算誤り。

## Rules

- /goal-loop 中は許可プロンプトを要する操作（対話コマンド、破壊的操作）を避け、ユーザーに確認せずループを完結させる（ユーザー指示 2026-06-12）。
