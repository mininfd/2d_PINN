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
- **Iterations used:** 8 of 15
- **In-flight change:** Iter 8 — 最終構成（psource + curriculum + 40k steps, lr 5e-4, reg 1.0）で sparse_study.csv 再生成（{64,36,16}）+ 64 mics を summary.csv へ（best_model.pt 保存）。続けて SIREN 正当改善の単発試行（pde_weight 0.01 + clip 1.0 + lr 1e-4 + curriculum）
- **Last known-good state:** psource 16 mics −18.12 dB（40k, diag.csv）、コミット a9f53bf
- **Next action:** 結果記録 → run_all.sh 更新 → 可視化生成 → README 執筆 → Phase 3 検証（機械チェック + verifier subagent）

## Experiment log

| # | Change | Structural / Scalar | Result vs. baseline | Keep? | Notes |
|---|--------|--------------------|--------------------|-------|-------|
| 0 | baseline | — | NMSE **+0.07 dB**（7.7 min, 20k steps） | keep | tanh-MLP 5x256, 64 mics。data loss 0.50→0.24 で停滞 — スペクトラルバイアスにより広帯域 (50–8000 Hz) をほぼ表現できず |
| 1 | SIREN ω0=30, lr=2e-3 | structural | **発散**: PDE loss 2.1→1e19、data loss 0.54 停滞、NMSE +0.05 dB | no | 初期は正常 → 学習中に発散 = lr 過大の症状。構造自体は棄却しない（Iter 2 で lr を下げて検証） |
| 2 | SIREN lr=5e-4 | scalar | **発散**: PDE loss ~1e17、NMSE −0.06 dB | no | lr 仮説棄却。診断: data-only は loss 1.6e-6 まで収束（NMSE +0.27 dB = 過学習）、grad_clip=1.0 でも PDE は 1e8 に爆発 → PDE 損失項が根本原因 |
| 3 | HerglotzNet J=256（平面波基底 + SIREN(k) 係数、data 損失のみ） | structural | **NMSE −16.11 dB**（1.1 min）— baseline 比 −16.2 dB 改善、criterion 6 達成 | **keep** | PDE はアーキテクチャで厳密充足（テストで autograd 検証）。途中 loss スパイクあり（cosine lr で回復） |
| 4 | 疎マイク {64,36,16} + mMLP 64 | — | herglotz: 64→−16.05 / 36→−10.71 / **16→−5.76 dB（未達）**。mmlp: −5.14 dB | keep | 16 mics は 256 方向基底に対し劣決定で過学習。mmlp は baseline を 5 dB 上回る（criterion 5 達成）。siren(+PDE) 爆発・tanh 停滞に対し mmlp は部分的に学習する点も興味深い |
| 5 | Herglotz 係数 ℓ2（Tikhonov）reg ∈ {1e-3,1e-2,1e-1} @16 mics | scalar | −6.49 / **−6.88** / −6.31 dB（−5.76 から改善も目標 −10 dB に遠い） | no | reg 依存性が平坦 = 係数ノルムは本質でない。16 mics の鍵は広帯域（k 方向）コヒーレンス → 周波数フラット音源という構造事前知識が必要 |
| 6 | PointSourceNet（ESM, M=16 学習可能位置）@16 mics | structural | −3.76 dB（data loss 0.20 で停滞） | 調査→継続 | 局所解: 鏡像 2 音源 (−0.42,−0.43)/(−0.42,+1.43) はほぼ同定、主音源 (−0.5,+0.5) が未捕捉（最寄り候補が (−0.86,+0.47) で停止）。位置地形が高周波で振動的 → 周波数カリキュラムで解消を図る |
| 7 | psource + 周波数カリキュラム @16 mics | structural | **NMSE −10.21 dB**（data loss 0.008、目標 −10 dB 達成） | **keep** | 3 真音源すべてを誤差 ~0.02 で同定（各 2 候補が振幅分担）。カリキュラム仮説検証済み。マージン薄 → 40k steps で強化を試行 |
| 7b | 同上 + 40k steps | scalar | **NMSE −18.12 dB** @16 mics（+8 dB 改善） | **keep** | 全帯域到達後の精密化時間が支配的。これを最終構成とする |
