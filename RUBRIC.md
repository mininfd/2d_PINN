# Rubric (loop3): PDE 損失 PINN を −9 dB 圏へ

**Goal:** loop2 の勝者 ksiren（KScaledSiren, PDE 損失あり）160k = −7.83 dB を起点に、PDE 残差損失（pde_weight > 0）で学習する NN ベース PINN の記録を **−9.0 dB 以下**へ更新する。物理厳密アーキテクチャ（herglotz / psource）は対象外。ストレッチは −12 dB（成功条件に含めない）。

**Baseline:** ksiren 64 mics 160k + curriculum → **−7.83 dB**（summary.csv に 2 行記録済み）。再測定しない。スクリーニング参照値: ksiren 40k = **−5.87 dB**。

**Budget:** 最大 12 イテレーション / 計算時間 ~5 時間。スクリーニングは 40k（~17 分）で行い、40k で −5.87 を ≳0.5 dB 上回った候補のみ 160k（~70 分）へスケール。

## 前提（LEARNINGS.md より、再導出しない）

- k·x 入力スケーリング（ksiren）が PDE 損失安定化の鍵。sine 系に tanh ボディを混ぜても相乗しない（KPlaneMMLP −5.31）
- 周波数マーチング（k_curriculum）は常用
- ksiren の走行間分散は小さい（−7.83/−7.83）が、勝者確定は再現実行で行う
- ステップ倍増あたりの改善は鈍化中（…−0.73 dB）→ スカラー延長だけでは −9 に届かない見込み。構造的変更（L-BFGS 仕上げ、幅、PDE 重み/サンプリング、RBA）を優先
- 実行: `C:\Projects\venv\Scripts\python.exe`（podman 不可）

## Criteria

| # | Criterion | Check (command or inspection) | Type | Status |
|---|-----------|-------------------------------|------|--------|
| 1 | PDE 損失 PINN（pde_weight > 0、NN ベース）が NMSE ≤ −9.0 dB @64 mics | `cat results/summary.csv`（該当行 notes に PDE 使用明記） | mechanical | ☐ |
| 2 | 勝者構成の再現実行（同一コマンド 2 回目）でも NMSE ≤ −9.0 dB | `cat results/summary.csv` に同構成 2 行 | mechanical | ☐ |
| 3 | 単体テストが通過 | `C:\Projects\venv\Scripts\python.exe -m pytest tests -q` | mechanical | ☐ |
| 4 | 勝者の再現コマンドが README に記載され、手法・結果が文書化 | verifier subagent | judgment | ☐ |
| 5 | 失敗から得た検証済みルールが LEARNINGS.md に追記 | inspection | mechanical | ☐ |

## 探索候補（優先順）

1. **L-BFGS 仕上げ**: Adam 後に固定コロケーション集合で L-BFGS 数百ステップ（PINN の定番、終端精度に効く）
2. **幅拡大**: ksiren width 256→384/512（--width で即試行可）
3. **PDE サンプリング/重み**: 高 k 側コロケーション強調、pde_weight 調整
4. **RBA**: 残差ベースの自適応重み付け
5. **320k 延長**（スカラー、~137 分、~−8.3 見込み）— 構造変更が出尽くした場合の合わせ技

## Loop state

- **Phase:** 中止（2026-06-13、イテレーション消化 0/12）
- **Status:** ユーザー指示により loop3 は「3 次元での測定範囲外音源の RIR 内挿」（`C:\Projects\3d_PINN`）へ再定義。この −9 dB 押し込みタスクは未着手のまま終了（in-flight だった L-BFGS 実装は revert 済み）
- **Next action:** なし（後継: `C:\Projects\3d_PINN\RUBRIC.md`）

## Experiment log

| # | Change | Structural / Scalar | Result vs. screening ref (40k −5.87) or baseline (−7.83) | Keep? | Notes |
|---|--------|--------------------|-----------------------------------------------------------|-------|-------|
