# LEARNINGS

Distilled, verified facts and rules for this project. Read this before working; do not re-derive.

## Environment

- **[verified 2026-06-12]** podman は使用不可：`podman machine start` は成功するが、イメージ `localhost/2d_pinn` がマシン内に存在せず、リポジトリに Dockerfile も無いため再ビルド不可。
  **Rule:** 実行は venv を使う — `C:\Projects\venv\Scripts\python.exe`（Python 3.13.13, numpy 2.4.4, scipy 1.17.1, torch 2.9.1+rocm, `torch.cuda.is_available()==True`, pytest 9.0.3）。RUBRIC の podman コマンドは venv 相当に読み替える。
- **[verified 2026-06-12]** 周波数ビン数：50〜8000 Hz を 10 Hz 刻みで生成すると (8000−50)/10+1 = **796** ビン。RUBRIC 初版の「797」は計算誤り。

## Training

- **[verified 2026-06-12]** SIREN（5x256, ω0=30）+ Helmholtz PDE 損失（k²正規化済み）は lr=2e-3 でも lr=5e-4 でも発散する（PDE loss が step 1000 までに ~1e17–1e19 へ爆発、data loss は 0.5 で停滞）。初期ステップは正常（PDE loss ~2.5）なので初期化の問題ではない。
  **Rule:** SIREN+PDE の発散は lr 調整だけでは直らない。PDE 項の有無・勾配クリッピングで切り分けてから対処する。

- **[verified 2026-06-12]** 広帯域 (50–8000 Hz) Helmholtz PINN では PDE 残差損失そのものが学習を破綻させる（tanh: 停滞 +0.07 dB / SIREN: 発散 / mmlp: 部分学習 −5.14 dB、grad_clip でも SIREN の発散は止まらない）。**物理をアーキテクチャに埋め込む**（平面波基底 HerglotzNet、等価音源 PointSourceNet — どちらも Helmholtz を厳密充足、data 損失のみで学習）と、64 mics で −16 dB、16 mics で −10 dB を達成。
  **Rule:** 高波数 (kL ≳ 50) の Helmholtz では PDE を損失でなく基底/構造で課す。
- **[verified 2026-06-12]** torch.special.bessel_j0/y0 は autograd 微分が未登録（backward が RuntimeError）。J0'=−J1, Y0'=−Y1 のカスタム autograd Function でラップする（src/models.py の _BesselJ0/_BesselY0）。
- **[verified 2026-06-12]** 学習可能位置の等価音源は全帯域一括学習だと局所解に陥る（16 mics で主音源を取り逃し −3.76 dB）。低周波 5% から帯域を線形拡大する周波数カリキュラム（60% 時点で全帯域）で 3 音源すべて誤差 ~0.02 で同定、−10.21 dB。
  **Rule:** 音源位置最適化は必ず周波数マーチングで行う。

## Rules

- /goal-loop 中は許可プロンプトを要する操作（対話コマンド、破壊的操作）を避け、ユーザーに確認せずループを完結させる（ユーザー指示 2026-06-12）。
