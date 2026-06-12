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
| tanh-MLP（baseline） | PDE 残差損失 | +0.07 | 7.7 min |
| SIREN | PDE 残差損失 | 発散（≈0） | 8.1 min |
| SIREN（縮小 128×3, data のみ） | なし | +1.03（過学習） | 0.5 min |
| modified MLP (Wang+ 2021) | PDE 残差損失 | −5.14 | 13.4 min |
| HerglotzNet（平面波基底） | **アーキテクチャで厳密充足** | −16.11 | 1.1 min |
| **PointSourceNet（ESM）+ 周波数カリキュラム** | **アーキテクチャで厳密充足** | **−33.27** | 1.3 min |

疎マイク実験（PointSourceNet + カリキュラム、40k steps）:

| マイク数 | NMSE [dB] |
|---|---|
| 64 (8×8) | −25.65 |
| 36 (6×6) | −31.39 |
| 16 (4×4) | **−22.06** |

16 mics（マイク間隔 0.33 m、空間ナイキスト ≈500 Hz）でも 8 kHz まで
−22 dB で内挿できる。鍵は広帯域コヒーレンス: 各マイクの周波数応答の
振動レートが音源距離を符号化しており、周波数フラットな点音源という
構造事前知識がこれを引き出す。

## 主な知見

1. **高波数では PDE 残差損失が学習を破綻させる。** kL ≈ 146 の広帯域
   Helmholtz で、PDE 損失を持つ tanh-MLP は停滞、SIREN は発散
   （lr 1e-4・勾配クリッピング・PDE 重み 0.01 でも回復せず）。
   診断の決め手: SIREN は data 損失のみなら 1.6e-6 まで収束する
   （ただし完全に過学習）— 問題は表現力ではなく PDE 項の最適化。
   なお、試したどの SIREN 構成も baseline を意味のある差では上回れて
   いない（表の「発散（≈0）」は数値上 baseline の +0.07 dB をわずかに
   下回るが、いずれもノイズレベル）。
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
```

## 構成

```
src/data.py        解析データ生成（2D Green 関数、Helmholtz 充足を単体テストで検証）
src/models.py      MLP / SIREN / ModifiedMLP / HerglotzNet / PointSourceNet
src/train.py       学習（PDE 残差・正則化・周波数カリキュラム）と NMSE 評価
experiments/       baseline.py, run.py, sparse_study.py, visualize.py
tests/             データ生成と物理厳密性の単体テスト（12 件）
```
