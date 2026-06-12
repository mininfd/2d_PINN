# Rubric: 2D PINN による測定範囲外音源の RIR 内挿

**Goal:** 2次元波動方程式に従う理想データに対し、測定領域内の疎なマイク点から領域全体の音場（RIR）を PINN で内挿し、密な評価グリッド上で NMSE ≤ -15 dB（64 mics）/ ≤ -10 dB（16 mics）を達成する。

**Baseline:** tanh-MLP PINN（データ損失 + Helmholtz PDE 残差損失）を 8×8=64 mics で学習し、33×33 密グリッド上の NMSE [dB] を `results/baseline.log` に記録する。未測定。

**Budget:** 最大 15 イテレーション / 4 時間（承認済み）。

## Problem setup（固定条件）

- 物理: 2次元同次 Helmholtz 方程式 ΔP + k²P = 0（k = 2πf/c）、c = 343 m/s
- 測定領域: [0,1]×[0,1] m
- 音源: 領域外 3 点音源（直接音 + 2 反射像）、データは 2D Green 関数（Hankel H₀⁽¹⁾）から解析生成
  - Source 1: (-0.5, 0.5), amp=1.0
  - Source 2: (-0.5, -0.5), amp=0.7（y=0 鏡像）
  - Source 3: (-0.5, 1.5), amp=0.7（y=1 鏡像）
- 信号帯域: 50 Hz ～ 8000 Hz（10 Hz ステップ、796 周波数ビン）※初版の 797 は計算誤り（LEARNINGS.md 参照）
- 学習マイク: 領域内格子点 {8×8=64, 6×6=36, 4×4=16}
- 評価: 33×33 密グリッド（held-out）での周波数域 NMSE [dB] = 10·log10(Σ|P̂−P|²/Σ|P|²)
- 実行環境: venv `C:\Projects\venv`（Python 3.13, torch 2.9.1+rocm, GPU 利用可）
  - podman はイメージ消失により使用不可（LEARNINGS.md 参照）。実行コマンド: `C:\Projects\venv\Scripts\python.exe ...`（チェックコマンド中の `podman run ... python3` / `pytest` は venv 相当に読み替え）

## Criteria

| # | Criterion | Check (command or inspection) | Type | Status |
|---|-----------|-------------------------------|------|--------|
| 1 | コンテナ環境が動作（torch + CUDA 利用可能） | `podman run --rm --device nvidia.com/gpu=all localhost/2d_pinn python3 -c "import torch; assert torch.cuda.is_available()"` | mechanical | ☑ |
| 2 | データ生成器が解析解と一致（単体テスト通過） | `C:\Projects\venv\Scripts\python.exe -m pytest tests -x -q` | mechanical | ☑ |
| 3 | Baseline（tanh-MLP PINN, 64 mics）の NMSE が記録済み | `cat results/baseline.log` | mechanical | ☑ |
| 4 | SIREN PINN が baseline を上回る（NMSE < baseline） | `cat results/summary.csv`（siren 行の NMSE < baseline NMSE） | mechanical | ☐ |
| 5 | mMLP PINN が実装され結果表に記載 | `cat results/summary.csv` に mmlp 行が存在 | mechanical | ☐ |
| 6 | 最良モデルが 64 mics で NMSE ≤ −15 dB | `cat results/summary.csv` | mechanical | ☐ |
| 7 | 疎マイク実験 {64,36,16} が完了し 16 mics で NMSE ≤ −10 dB | `cat results/sparse_study.csv` | mechanical | ☐ |
| 8 | 単一コマンドで全パイプライン再現可能 | `bash run_all.sh` が完走 | mechanical | ☐ |
| 9 | README に手法・結果・可視化（音場スナップショット図）が記載 | verifier subagent が README と図を検査 | judgment | ☐ |

## Loop state

- **Phase:** 2 loop
- **Iterations used:** 3 of 15
- **In-flight change:** Iter 3 — HerglotzNet（平面波基底 J=256、係数 SIREN(k)、Helmholtz 厳密充足、data 損失のみ）を実装して 64 mics で実行
- **Last known-good state:** baseline 完了（NMSE +0.07 dB）、コミット 1fc5f53
- **Next action:** HerglotzNet 完了 → NMSE 確認・ログ・コミット → 良ければ criterion 5 用 mMLP 実行 → 疎マイク実験

## Experiment log

| # | Change | Structural / Scalar | Result vs. baseline | Keep? | Notes |
|---|--------|--------------------|--------------------|-------|-------|
| 0 | baseline | — | NMSE **+0.07 dB**（7.7 min, 20k steps） | keep | tanh-MLP 5x256, 64 mics。data loss 0.50→0.24 で停滞 — スペクトラルバイアスにより広帯域 (50–8000 Hz) をほぼ表現できず |
| 1 | SIREN ω0=30, lr=2e-3 | structural | **発散**: PDE loss 2.1→1e19、data loss 0.54 停滞、NMSE +0.05 dB | no | 初期は正常 → 学習中に発散 = lr 過大の症状。構造自体は棄却しない（Iter 2 で lr を下げて検証） |
| 2 | SIREN lr=5e-4 | scalar | **発散**: PDE loss ~1e17、NMSE −0.06 dB | no | lr 仮説棄却。診断: data-only は loss 1.6e-6 まで収束（NMSE +0.27 dB = 過学習）、grad_clip=1.0 でも PDE は 1e8 に爆発 → PDE 損失項が根本原因 |
