# 2D PINN: 測定領域外音源の広帯域音場内挿

2 次元同次 Helmholtz 方程式 ΔP + k²P = 0 に従う理想音場を、測定領域
[0,1]×[0,1] m 内の疎なマイク格子点だけから、領域全体（33×33 held-out
グリッド）で内挿するタスク。音源は領域**外**の 3 点音源（直接音 1 +
鏡像反射 2、2D Green 関数 (i/4)H₀⁽¹⁾(kr) で解析生成）、帯域は
50–8000 Hz（10 Hz 刻み、796 ビン）。

## 最終結果サマリ

評価指標は広帯域 NMSE [dB] = 10·log₁₀(Σ|P̂−P|²/Σ|P|²)（33×33 グリッド、全 796 周波数）。
64 mics（8×8 格子）での比較:

| モデル | 物理の課し方 | NMSE [dB] | 学習時間 |
|---|---|---|---|
| tanh-MLP（baseline） | PDE 残差損失 | −6.18 ※ | 7.6 min |
| SIREN | PDE 残差損失 | 発散（≈0） | 8.2 min |
| SIREN（縮小 128×3、data のみ） | なし | +1.03（過学習） | 0.5 min |
| KScaledSiren（k·x 入力、80k+カリキュラム） | PDE 残差損失 | −7.10 | 34.3 min |
| **KScaledSiren（160k+カリキュラム）** | PDE 残差損失 | **−7.83**（PDE 損失系の最良、再現 2 回とも） | 68.3 min |
| modified MLP (Wang+ 2021) | PDE 残差損失 | −7.78 | 13.1 min |
| HerglotzNet（平面波基底） | **アーキテクチャで厳密充足** | −15.84 | 1.1 min |
| **PointSourceNet（ESM）+ 周波数カリキュラム** | **アーキテクチャで厳密充足** | **−34.91** | 1.3 min |

※ baseline は同一シードでも GPU 非決定性で +0.07 dB（停滞）と −6.18 dB（部分学習）に分かれる
双安定な学習を示す。表は `run_all.sh` 再現実行の値。

疎マイク実験（PointSourceNet + カリキュラム、40k steps）:

| マイク数 | NMSE [dB] |
|---|---|
| 64 (8×8) | −25.15 |
| 36 (6×6) | −31.38 |
| 16 (4×4) | **−22.06** |

16 mics（マイク間隔 0.33 m、空間ナイキスト ≈500 Hz）でも 8 kHz まで −22 dB で内挿できる。

## 試行過程

### Loop 1 — 音場内挿の基礎実験（2026-06-12）

**目的:** 2D Helmholtz 音場内挿のベースラインを確立し、PDE 残差損失と物理埋込みアーキを比較。

**試行 1: tanh-MLP + PDE 残差損失（baseline）**
- 結果: −6.18 dB（GPU 非決定性で +0.07 dB と −6.18 dB に二分される双安定な学習）
- 発見: PDE 損失付き tanh-MLP は停滞または部分学習。kL ≈ 146 の高波数で PDE 項が学習を阻害する。

**試行 2: SIREN + PDE 残差損失**
- 結果: 発散（PDE loss が step 1000 までに 1e17–1e19 へ爆発、data loss は 0.5 で停滞）
- lr=2e-3 → lr=5e-4 変更、grad_clip 追加でも改善せず。
- 原因: SIREN の固有曲率 (2ω₀W)² は k 非依存に大きく、k² 正規化 PDE 残差が低周波で増幅される。

**試行 3: SIREN（data 損失のみ、縮小版 128×3）**
- 結果: +1.03 dB（完全過学習）
- 確認: 発散は表現力の問題ではなく PDE 損失の最適化の問題。

**試行 4: HerglotzNet（平面波基底、J=256 方向）**
- アーキテクチャ: P(x,k) = Σⱼ cⱼ(k)·exp(ik dⱼ·x)。単位円上の J=256 平面波は Helmholtz を厳密充足。係数 cⱼ(k) は周波数上の SIREN（高速振動する k 応答に tanh は不適）。
- 結果: −15.84 dB（data 損失のみ、1.1 min）
- 発見: 「物理をアーキテクチャに埋め込む」アプローチが PDE 損失 PINN を圧倒（9.66 dB 改善）。

**試行 5: PointSourceNet（等価音源法 ESM、M=16）+ 周波数カリキュラム**
- アーキテクチャ: 域外位置を持つ M=16 モノポール + 複素振幅 + 共有 k 変調 MLP。Green 関数 (i/4)H₀⁽¹⁾(kr) で Helmholtz を厳密充足。
- カリキュラム: 低周波 5% から帯域を線形拡大（60% 時点で全帯域）。
- 結果: **−34.91 dB**（40k steps、1.3 min）
- 学習後の音源位置は真値 (−0.5,±0.5), (−0.5,1.5) を誤差 ~0.02 で同定。
- 発見: 全帯域一括学習（カリキュラムなし）では −3.76 dB（局所解）。周波数マーチングが位置最適化に必須。

**疎マイク実験（Loop 1 完了後）**
- PointSourceNet + カリキュラム 40k steps を 16/36/64 mics で比較。
- 16 mics (4×4、マイク間隔 0.33 m) でも −22.06 dB。鍵は広帯域コヒーレンス: マイクの周波数応答の振動レートが音源距離を符号化しており、周波数フラットな点音源という構造事前知識がこれを引き出す。

---

### Loop 2 — PDE 損失 PINN の改善（2026-06-13）

**目的:** SIREN の発散原因を解明し、安定学習できる PDE 損失 PINN を開発。

**発散原因の解析**
- SIREN の第 1 層 sin(ω₀Wx) は入力 x に対して固有曲率 (2ω₀W)² を持つ。
- Helmholtz 残差 ΔP + k²P を k² で正規化しても、低周波（k が小さい）では曲率項が増幅される（step 1 で PDE loss ≈ 35）。
- k·x にスケールした入力（KScaledSiren）は第 1 層が sin(W·kx) となり、曲率 ∝ k² に追従する。

**試行 6: KScaledSiren（k·x スケーリング）**
- 20k steps: −4.13 dB / 40k: −5.87 dB / 80k: −7.10 dB / 160k: −7.83 dB（単調改善）
- 160k cosine lr で tanh-MLP baseline（−6.18）を超え、mmlp（−7.78）も上回る。
- 走行間分散が小さい（同一コマンド 2 回で −7.83/−7.83 dB）。
- **PDE 損失系の最良**となるが、ESM（−34.91 dB）との差は依然 27 dB 以上。

**試行 7: Modified MLP (Wang+ 2021, mmlp)**
- 結果: −7.78 dB（13.1 min）
- KScaledSiren（−7.83 dB）とほぼ同等。

**試行 8: KPlaneMMLP（sine 特徴 + mmlp ボディ）**
- 曲率整合した sine 平面波特徴層の後段に tanh 系 mmlp ボディを合成。
- 結果: −5.31 dB（発散しないが、pure sine や mmlp より劣る）
- 発見: sine 特徴と tanh ボディは相乗せず。sine 系は一貫して sine で構成する必要がある。

**結論（Loop 2）**
- PDE 損失 PINN でも kL≈146 の広帯域では ESM の 5 分の 1 の精度（−7.83 vs −34.91 dB）。
- 実用的には PDE をアーキテクチャに埋め込む ESM/HerglotzNet が圧倒的優位。

---

## 主な知見

1. **高波数では PDE 残差損失が学習を破綻させる。** kL ≈ 146 の広帯域 Helmholtz で tanh-MLP は停滞〜部分学習、SIREN は発散（lr・grad_clip・PDE 重み・カリキュラムでも回復せず）。発散の根本原因は SIREN の固有曲率 (2ω₀W)² が k 非追従で低周波 PDE 残差を増幅することにある。k·x スケーリング（KScaledSiren）は安定化するが精度は ESM に遠く及ばない。

2. **物理は損失でなくアーキテクチャで課す。**
   - **HerglotzNet**: 単位円上の J=256 方向の平面波は各々 Helmholtz を厳密充足。係数 cⱼ(k) を k 上の SIREN で学習 → −15.84 dB。
   - **PointSourceNet（ESM）**: 学習可能な域外位置の M=16 モノポール。音源位置を直接最適化することで真値 3 音源を自動同定 → −34.91 dB。

3. **音源位置の最適化には周波数マーチングが必須。** 全帯域一括では損失地形が振動的で局所解（−3.76 dB）。低周波 5% から拡大するカリキュラムで −22.06 dB（16 mics）。

## 図

- `results/field_snapshots.png` — 真の音場と PINN 再構成の Re(P) スナップショット（500 / 2000 / 6000 Hz）
- `results/nmse_vs_freq.png` — 周波数別 NMSE と広帯域値

## 再現方法

```bash
bash run_all.sh   # テスト → 全モデル学習 → 疎マイク実験 → 図の生成
```

実行環境: `C:\Projects\venv`（Python 3.13, PyTorch 2.9 ROCm）。
`PY=<python へのパス> bash run_all.sh` で他環境の Python を指定可能。

個別実行例:

```bash
python experiments/run.py --model psource --mics 8 --lr 5e-4 \
    --pde-weight 0 --reg 1.0 --curriculum --steps 40000
python experiments/sparse_study.py --model psource --pde-weight 0 \
    --lr 5e-4 --reg 1.0 --curriculum --steps 40000
python experiments/visualize.py --ckpt results/best_model.pt

# KScaledSiren（run_all.sh には含まれない追加実験。160k が PDE 損失系の最良）
python experiments/run.py --model ksiren --lr 5e-4 --steps 160000 --curriculum
```

## 構成

```
src/data.py        解析データ生成（2D Green 関数、Helmholtz 充足を単体テストで検証）
src/models.py      MLP / SIREN / ModifiedMLP / HerglotzNet / PointSourceNet / KScaledSiren
src/train.py       学習（PDE 残差・正則化・周波数カリキュラム）と NMSE 評価
experiments/       baseline.py, run.py, sparse_study.py, visualize.py
tests/             データ生成と物理厳密性の単体テスト（12 件）
```
