# MRNet Dataset v1.0 データ構造説明

## 概要

MRNetは、スタンフォード大学機械学習グループが公開した膝MRI画像データセットです。膝の異常（Abnormal）、前十字靭帯（ACL）損傷、半月板（Meniscus）損傷の検出を目的としています。

## ディレクトリ構造

```
MRNet-v1.0/
├── train/                        # 訓練データ（1130症例: 0000-1129）
│   ├── axial/                    # 軸位断面画像
│   │   ├── 0000.npy
│   │   ├── 0001.npy
│   │   └── ... (1130ファイル)
│   ├── coronal/                  # 冠状断面画像
│   │   ├── 0000.npy
│   │   ├── 0001.npy
│   │   └── ... (1130ファイル)
│   └── sagittal/                 # 矢状断面画像
│       ├── 0000.npy
│       ├── 0001.npy
│       └── ... (1130ファイル)
├── valid/                        # 検証データ（120症例: 1130-1249）
│   ├── axial/                    # 軸位断面画像
│   │   ├── 1130.npy
│   │   └── ... (120ファイル)
│   ├── coronal/                  # 冠状断面画像
│   │   ├── 1130.npy
│   │   └── ... (120ファイル)
│   └── sagittal/                 # 矢状断面画像
│       ├── 1130.npy
│       └── ... (120ファイル)
├── train-abnormal.csv            # 訓練データ: 異常ラベル
├── train-acl.csv                 # 訓練データ: ACL損傷ラベル
├── train-meniscus.csv            # 訓練データ: 半月板損傷ラベル
├── valid-abnormal.csv            # 検証データ: 異常ラベル
├── valid-acl.csv                 # 検証データ: ACL損傷ラベル
└── valid-meniscus.csv            # 検証データ: 半月板損傷ラベル
```

## データ詳細

### 画像データ（.npy形式）

- **ファイル形式**: NumPy配列 (.npy)
- **命名規則**: `{症例ID}.npy` (4桁ゼロパディング)
- **3つの撮像方向**:
  - **axial（軸位断）**: 膝を上から見た断面
  - **coronal（冠状断）**: 膝を前から見た断面
  - **sagittal（矢状断）**: 膝を横から見た断面

### ラベルファイル（CSV形式）

各CSVファイルは以下の形式です：

| カラム | 内容 |
|--------|------|
| 1列目 | 症例ID（4桁数字） |
| 2列目 | ラベル（0 または 1） |

#### ラベルの意味

| ラベル | 意味 |
|--------|------|
| 0 | 陰性（損傷・異常なし） |
| 1 | 陽性（損傷・異常あり） |

#### 3つの診断タスク

| ファイル | 診断内容 |
|----------|----------|
| `*-abnormal.csv` | 膝の異常全般の有無 |
| `*-acl.csv` | 前十字靭帯（ACL）損傷の有無 |
| `*-meniscus.csv` | 半月板損傷の有無 |

## データ統計

| セット | 症例数 | 症例ID範囲 |
|--------|--------|------------|
| 訓練データ (train) | 1,130 | 0000 - 1129 |
| 検証データ (valid) | 120 | 1130 - 1249 |
| **合計** | **1,250** | 0000 - 1249 |

## 使用例

### Pythonでの読み込み例

```python
import numpy as np
import pandas as pd

# 画像データの読み込み
image = np.load('train/sagittal/0000.npy')
print(f"Image shape: {image.shape}")

# ラベルの読み込み
labels = pd.read_csv('train-acl.csv', header=None, names=['case_id', 'label'])
print(labels.head())
```

## 参考文献

- Stanford ML Group MRNet: https://stanfordmlgroup.github.io/competitions/mrnet/
- 論文: Bien et al., "Deep-learning-assisted diagnosis for knee magnetic resonance imaging: Development and retrospective validation of MRNet" (PLOS Medicine, 2018)

## ライセンス

データセットの使用にはスタンフォード大学の利用規約に従う必要があります。研究目的での使用が許可されています。
