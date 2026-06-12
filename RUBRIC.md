# Rubric: PDE 損失を使う PINN で mMLP 超え

**Goal:** PDE 残差損失（pde_weight > 0）で学習する NN ベースの PINN を探索し、64 mics で mMLP の NMSE −7.78 dB を確実に（再現性をもって）上回る。物理厳密アーキテクチャ（herglotz / psource、PDE 損失なし）は対象外 — 「PDE を損失として使う」系譜での最良を更新する。ストレッチ目標は −12 dB（成功条件には含めない）。

**Baseline:** mmlp 64 mics 20k steps → **−7.78 dB**（results/summary.csv 記録済み、run_all 再現値）。再測定しない。

**Budget:** 最大 12 イテレーション / 計算時間 ~3.5 時間。

## 前提（LEARNINGS.md より、再導出しない）

- 標準 SIREN + PDE は発散（曲率が k 非追従）。k·x 入力（ksiren）で安定化済み: 20k/40k/80k → −4.13/−5.87/−7.10 dB（単調改善、PDE ~0.02 安定）
- 周波数マーチング（k_curriculum）は有効
- GPU 非決定性により ~0 dB 付近の NN-PINN は走行間で大きくブレる（tanh: +0.07 ↔ −6.18）→ 勝者は再現実行で確認する
- 実行: `C:\Projects\venv\Scripts\python.exe`（podman 不可）

## Criteria

| # | Criterion | Check (command or inspection) | Type | Status |
|---|-----------|-------------------------------|------|--------|
| 1 | PDE 損失 PINN（pde_weight > 0、NN ベース）が NMSE < −7.78 dB @64 mics | `cat results/summary.csv`（該当行の notes に PDE 使用を明記） | mechanical | ☑（ksiren 160k −7.83） |
| 2 | 勝者構成の再現実行（同一コマンド 2 回目）でも < −7.78 dB | `cat results/summary.csv` に同構成 2 行 | mechanical | ☑（再現も −7.83） |
| 3 | 単体テストが通過 | `C:\Projects\venv\Scripts\python.exe -m pytest tests -q` | mechanical | ☐ |
| 4 | 勝者の再現コマンドが README に記載され、手法・結果が文書化 | verifier subagent | judgment | ☐ |
| 5 | 失敗から得た検証済みルールが LEARNINGS.md に追記 | inspection | mechanical | ☐ |

## 探索候補（優先順）

1. **KPlaneMMLP**: sine 平面波特徴層（k·x 入力、曲率 ∝ k²）+ mMLP ボディ — ksiren の安定化と mmlp の最適化特性の合成
2. **ksiren 160k**: 単調改善トレンドの外挿（~−8.1 dB 見込み、70 分）
3. ksiren の幅/ω0 調整、L-BFGS 仕上げ、残差ベース損失重み付け（RBA）

## Loop state

- **Phase:** 2 loop
- **Iterations used:** 3 of 12
- **In-flight change:** Phase 3 — pytest（criterion 3）→ README/LEARNINGS 更新（criteria 4,5）→ verifier subagent
- **Last known-good state:** ksiren 160k −7.83 dB ×2（criteria 1,2 ✓）
- **Next action:** verifier 判定 → 最終報告

## Experiment log

| # | Change | Structural / Scalar | Result vs. baseline (−7.78) | Keep? | Notes |
|---|--------|--------------------|-----------------------------|-------|-------|
| 1 | KPlaneMMLP（sine 特徴 + mMLP body, 40k, curriculum） | structural | **−5.31 dB**（未達） | no | 発散せず安定（曲率整合は機能）だが data loss 0.154 で ksiren 40k（0.099）より悪い。mMLP body は sine 特徴と相乗しない |
| 2 | ksiren 160k + curriculum | scalar | **−7.83 dB**（baseline 比 −0.05 dB、criterion 1 達成） | **keep** | data 0.002 / pde 0.0066。ダブリングあたり改善は −0.73 dB に鈍化（飽和傾向）。68.3 分 |
| 3 | 同一コマンド再現実行 | — | **−7.83 dB**（criterion 2 達成） | **keep** | 終端 loss は異なる（data 0.0039 vs 0.0020）のに NMSE は 2 桁一致 → ksiren の走行間分散は小さい（tanh の双安定と対照的） |
