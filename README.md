# 2D PINN: 測定領域外音源の広帯域音場内挿

2 次元同次 Helmholtz 方程式 ΔP + k²P = 0 に従う理想音場を、測定領域
[0,1]×[0,1] m 内の疎なマイク格子点だけから、領域全体（33×33 held-out
グリッド）で内挿するタスク。音源は領域**外**の 3 点音源（直接音 1 +
鏡像反射 2、2D Green 関数 (i/4)H₀⁽¹⁾(kr) で解析生成）、帯域は
50–8000 Hz（10 Hz 刻み、796 ビン）。

## 結果サマリ

評価指標は広帯域 NMSE [dB] = 10·log₁₀(Σ|P̂−P|²/Σ|P|²)（33×33 グリッド、
全 796 周波数）。64 mics（8×8 格子）での比較:

| モデル | 物理の課し方 | NMSE [dB] | 学習時間 |
|---|---|---|---|
| tanh-MLP（baseline） | PDE 残差損失 | −6.18 ※ | 7.6 min |
| SIREN | PDE 残差損失 | 発散（≈0） | 8.2 min |
| SIREN（縮小 128×3, data のみ） | なし | +1.03（過学習） | 0.5 min |
| KScaledSiren（k·x 入力、80k+カリキュラム） | PDE 残差損失 | **−7.10** | 34.3 min |
| modified MLP (Wang+ 2021) | PDE 残差損失 | −7.78 | 13.1 min |
| HerglotzNet（平面波基底） | **アーキテクチャで厳密充足** | −15.84 | 1.1 min |
| **PointSourceNet（ESM）+ 周波数カリキュラム** | **アーキテクチャで厳密充足** | **−34.91** | 1.3 min |

※ baseline は同一シードでも GPU 非決定性で +0.07 dB（停滞）と −6.18 dB
（部分学習）に分かれる双安定な学習を示す。表は `run_all.sh` 再現実行の値
（縮小 SIREN と KScaledSiren は追加実験。出典は RUBRIC.md の実験ログで、
KScaledSiren は summary.csv にも追記済み。`run_all.sh` を再実行すると
summary.csv は基本 5 行に再生成される）。同様に GPU 非決定性のため、psource の 64 mics は実行間で
−25〜−35 dB 程度の揺らぎがある（音源位置精密化の到達点が走行ごとに異なる。
疎マイク表で 36 mics が 64 mics を上回るのも同じ理由で、マイク数の効果では
ない）。

疎マイク実験（PointSourceNet + カリキュラム、40k steps）:

| マイク数 | NMSE [dB] |
|---|---|
| 64 (8×8) | −25.15 |
| 36 (6×6) | −31.38 |
| 16 (4×4) | **−22.06** |

16 mics（マイク間隔 0.33 m、空間ナイキスト ≈500 Hz）でも 8 kHz まで
−22 dB で内挿できる。鍵は広帯域コヒーレンス: 各マイクの周波数応答の
振動レートが音源距離を符号化しており、周波数フラットな点音源という
構造事前知識がこれを引き出す。

## 主な知見

1. **高波数では PDE 残差損失が学習を破綻させる。** kL ≈ 146 の広帯域
   Helmholtz で、PDE 損失を持つ tanh-MLP は停滞〜部分学習、SIREN は発散
   （lr 1e-4・勾配クリッピング・PDE 重み 0.01・周波数カリキュラムでも
   回復せず）。SIREN は data 損失のみなら 1.6e-6 まで収束する（ただし
   完全に過学習）— 問題は表現力ではなく PDE 項の最適化。
   発散の根本原因は**低 k 側**にある: SIREN の固有曲率 (2ω₀W)² は k に
   依存せず大きいため、k² 正規化残差が低周波で増幅される（初期 PDE
   loss 35）。空間入力を k·x にスケールした KScaledSiren（第 1 層 =
   ランダム平面波 sin(W·kx)、曲率 ∝ k²）は同一 PDE 損失で発散せず安定
   学習し（PDE ~0.02）、学習ステップに対し単調改善（20k/40k/80k で
   −4.13/−5.87/−7.10 dB）して 80k steps で baseline（−6.18 dB）を上回る。
   PDE 損失で学習できた唯一の sine 系ネットワークである。
2. **物理は損失でなくアーキテクチャで課す。**
   - **HerglotzNet**: P(x,k) = Σⱼ cⱼ(k)·exp(i k dⱼ·(x−x₀))。単位円上の
     J=256 方向の平面波は各々 Helmholtz を厳密に満たす（autograd 検証の
     単体テストあり）。係数 cⱼ(k) は k 上の SIREN（音源の係数は k に
     対して高速振動するため tanh では表現不能）。
   - **PointSourceNet（等価音源法）**: 学習可能な域外位置を持つ M=16 個の
     モノポール + 周波数フラット複素振幅 + 共有のなめらかな k 変調 MLP。
     学習後の音源位置は真値 (−0.5,±0.5), (−0.5,1.5) を誤差 ~0.02 で同定。
3. **音源位置の最適化には周波数マーチングが必須。** 全帯域一括では
   位置の損失地形が振動的で局所解に陥る（16 mics で −3.76 dB）。
   低周波 5% から帯域を線形拡大するカリキュラム（学習の 60% で全帯域、
   残りで精密化）により −22.06 dB。精密化時間も支配的（20k→40k steps
   で −10.2 → −18.1 dB @16 mics）。

## 図

- `results/field_snapshots.png` — 真の音場と PINN 再構成の Re(P)
  スナップショット（500 / 2000 / 6000 Hz、33×33 グリッド）
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

# KScaledSiren（run_all.sh には含まれない追加実験）
python experiments/run.py --model ksiren --lr 5e-4 --steps 80000 --curriculum
```

## 構成

```
src/data.py        解析データ生成（2D Green 関数、Helmholtz 充足を単体テストで検証）
src/models.py      MLP / SIREN / ModifiedMLP / HerglotzNet / PointSourceNet
src/train.py       学習（PDE 残差・正則化・周波数カリキュラム）と NMSE 評価
experiments/       baseline.py, run.py, sparse_study.py, visualize.py
tests/             データ生成と物理厳密性の単体テスト（12 件）
```
