# データセット構造ガイド

このドキュメントでは、本リポジトリで学習に使用されるデータセットの最終的なフォルダ構造とファイル形式について説明します。

---

## 概要

本リポジトリでは、3D医用画像の分類タスクを行います。データセットは前処理後、特定のフォルダ構造とファイル形式で保存される必要があります。

### 主な特徴
- **画像形式**: Blosc2圧縮形式（`.b2nd`拡張子）
- **メタデータ**: JSON形式（ラベル、分割情報）
- **交差検証**: K-Fold（3〜5分割）

---

## データセット構造パターン

本リポジトリには2つの主要なデータセット構造パターンがあります。

---

### パターン1: 標準構造（ABIDE, MRNet, RSNA_Spine）

```
<data_root_dir>/
├── nnUNetResEncUNetLPlans_3d_fullres/
│   ├── case001.b2nd
│   ├── case002.b2nd
│   ├── case003.b2nd
│   └── ...
├── labelsTr.json
└── splits_final.json
```

#### ファイル説明

| ファイル/フォルダ | 説明 |
|---|---|
| `nnUNetResEncUNetLPlans_3d_fullres/` | 前処理済み画像ファイルを格納するフォルダ |
| `*.b2nd` | Blosc2圧縮された3D画像データ |
| `labelsTr.json` | 各サンプルのラベル情報 |
| `splits_final.json` | 交差検証用のtrain/val分割情報 |

#### labelsTr.json の形式

```json
{
  "case001": [0, 1],
  "case002": [0, 0],
  "case003": [0, 1]
}
```

> 注: ABIDEとRSNA_Spineでは `labels[i][1]` でラベルを取得（配列の2番目の要素）

または単純な形式:
```json
{
  "case001": 0,
  "case002": 1,
  "case003": 0
}
```

#### splits_final.json の形式

```json
[
  {
    "train": ["case001", "case002", "case003", ...],
    "val": ["case010", "case011", ...]
  },
  {
    "train": ["case001", "case010", "case011", ...],
    "val": ["case002", "case003", ...]
  },
  ...
]
```

> 配列のインデックスがfold番号に対応（fold=0, fold=1, ...）

---

### パターン2: モダリティ別構造（abide_1mm_cropped_160シリーズ）

```
<data_root_dir>/
├── abide_1mm_cropped_160/          # または abide_1mm_cropped_160_new/
│   ├── case001_crop.b2nd           # 古い形式
│   ├── case001_0000.b2nd           # 新しい形式（モダリティ0）
│   ├── case001_0001.b2nd           # 新しい形式（モダリティ1）
│   ├── case002_0000.b2nd
│   └── ...
├── labels.json
└── splits.json
```

#### ファイル命名規則

| 形式 | ファイル名パターン | 説明 |
|---|---|---|
| 旧形式 | `{case_id}_crop.b2nd` | 単一モダリティ |
| 新形式 | `{case_id}_000{mod_id}.b2nd` | マルチモダリティ対応 |

#### labels.json の形式

```json
{
  "case001.nii.gz": 0,
  "case002.nii.gz": 1,
  "case003.nii.gz": 0
}
```

> 注: キーは元のNIfTIファイル名

#### splits.json の形式

```json
[
  {
    "train": ["case001.nii.gz", "case002.nii.gz", ...],
    "val": ["case010.nii.gz", "case011.nii.gz", ...]
  },
  ...
]
```

---

## 前処理パイプライン

前処理前の**生データ**は以下の構造で準備します：

```
<raw_data_dir>/
├── imagesTr/
│   ├── case001_0000.nii.gz    # モダリティ0
│   ├── case001_0001.nii.gz    # モダリティ1（複数モダリティの場合）
│   ├── case002_0000.nii.gz
│   └── ...
└── masks/                      # （オプション）脳抽出用マスク
    ├── case001_0000_bet.nii.gz
    └── ...
```

### 前処理ステップ

1. **HD-BET脳抽出**（オプション）: 脳マスクを生成
2. **リサンプリング**: 目標スペーシング（例: 1mm等方性）にリサンプル
3. **クロッピング**: マスク中心から固定サイズでクロップ（例: 160×160×160）
4. **正規化**: Z-score正規化
5. **保存**: Blosc2形式で圧縮保存
6. **分割ファイル生成**: K-Fold交差検証用のsplits.jsonを生成

---

## データセット別設定

### ABIDE 1mm Cropped 160

| 項目 | 値 |
|---|---|
| パッチサイズ | [160, 160, 160] |
| クラス数 | 2 |
| 入力チャンネル | 1 |
| 交差検証分割数 | 3 |

### MRNet

| 項目 | 値 |
|---|---|
| パッチサイズ | [32, 256, 256] |
| クラス数 | 3 |
| 入力チャンネル | 1 |
| 交差検証分割数 | 5 |
| タスク | multilabel |

### RSNA Spine

| 項目 | 値 |
|---|---|
| パッチサイズ | [160, 192, 192] |
| クラス数 | 2 |
| 入力チャンネル | 1 |
| 交差検証分割数 | 5 |

---

## Blosc2ファイル形式

### 保存

```python
from datasets.blosc2io import Blosc2IO

# 保存
Blosc2IO.save(
    data=numpy_array,          # np.ndarray
    filepath="output.b2nd",    # 出力パス
    chunks=chunk_size,         # チャンクサイズ
    blocks=block_size,         # ブロックサイズ
    metadata={"key": "value"}, # メタデータ（オプション）
    clevel=8,                  # 圧縮レベル（0-9）
    codec=blosc2.Codec.ZSTD    # 圧縮コーデック
)
```

### 読み込み

```python
from datasets.blosc2io import Blosc2IO

# 読み込み
data, metadata = Blosc2IO.load("input.b2nd", mode="r")
numpy_array = data[...]  # 全データをnumpy配列として取得
```

---

## カスタムデータセットの追加方法

1. 上記の構造に従ってデータを準備
2. `datasets/`に新しいPythonファイルを作成
3. `BaseDataModule`を継承したDataModuleクラスを実装
4. `cli_configs/data/`に対応するYAML設定ファイルを作成

詳細は[README.md](README.md)を参照してください。

---

## まとめ

```
最終的なデータセット構造:
=====================================

パターン1（標準）:
<data_root>/
├── nnUNetResEncUNetLPlans_3d_fullres/
│   └── *.b2nd
├── labelsTr.json
└── splits_final.json

パターン2（モダリティ別）:
<data_root>/
├── <dataset_name>/
│   └── *_000X.b2nd
├── labels.json
└── splits.json
```
