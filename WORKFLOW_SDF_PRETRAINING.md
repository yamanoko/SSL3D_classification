# SDF事前学習 → ABIDEファインチューニング ワークフロー

FDSLxSDF4Segで生成したSDF合成データでモデルを事前学習し、ABIDEデータセットでファインチューニングすることで、SDF事前学習の効果を検証するワークフローです。

論文 "[An OpenMind for 3D medical vision self-supervised learning](https://arxiv.org/abs/2412.17041)" で提示されたSSL事前学習の結果と比較するための実験設計を含みます。

---

## 目次

1. [実験の全体像](#1-実験の全体像)
2. [なぜチェックポイント変換が必要か](#2-なぜチェックポイント変換が必要か)
3. [Step 1: SDFデータセット生成](#step-1-sdfデータセット生成)
4. [Step 2: SDFデータで事前学習](#step-2-sdfデータで事前学習)
5. [Step 3: チェックポイント変換](#step-3-チェックポイント変換)
6. [Step 4: ABIDEでファインチューニング](#step-4-abideでファインチューニング)
7. [比較実験の設計](#比較実験の設計)
8. [注意事項](#注意事項)

---

## 1. 実験の全体像

```
┌──────────────────────────────────────────────────────────────────────┐
│  FDSLxSDF4Seg                                                       │
│  generate_sdf_dataset_classification.py                              │
│  → SDF分類データセット生成 (Blosc2形式, 160³)                          │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ データ
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  SSL3D_classification                                                │
│  main.py  data=sdf_classification  model=resenc                      │
│  → SDF分類タスクで学習 (エンコーダの特徴表現を獲得)                     │
│  → Lightning チェックポイント (.ckpt) を保存                           │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ .ckpt
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  convert_sdf_checkpoint.py                                           │
│  → Lightning形式 → nnUNet/SSL形式 に変換                              │
│  → エンコーダ重みのみ抽出、キー名を変換                                │
│  → .pth ファイルとして保存                                             │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ .pth (network_weights形式)
                         ▼
┌──────────────────────────────────────────────────────────────────────┐
│  SSL3D_classification                                                │
│  main.py  data=abide_1mm_cropped_160_new  model=resenc               │
│           model.pretrained=True  model.chpt_path=xxx.pth             │
│  → ABIDEで分類ファインチューニング                                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 2. なぜチェックポイント変換が必要か

同じSSL3D_classificationフレームワークを使っているのに変換が必要な理由は、**フレームワーク内に2種類の異なるチェックポイント形式が存在する**ためです。

### 形式の違い

| 項目 | Lightning チェックポイント (.ckpt) | SSL事前学習チェックポイント (.pth) |
|------|---|----|
| **保存元** | SSL3D_classification の学習時に `ModelCheckpoint` が自動保存 | nnUNet SSLフレームワーク (MAE, Spark等) が保存 |
| **トップレベルキー** | `state_dict`, `epoch`, `optimizer_states`, ... | `network_weights` |
| **含まれる重み** | モデル全体（エンコーダ＋分類ヘッド＋メトリクス等） | エンコーダのみ |
| **キーの命名規則** | `encoder.res_unet.encoder.stem.convs.0...` | `encoder.stem.convs.0...` |
| **読み込み関数** | `Trainer.fit(ckpt_path=...)` | `load_pretrained_weights()` |

### 具体的なキー名の違い

```python
# Lightning (.ckpt) の state_dict キー例:
"encoder.res_unet.encoder.stem.convs.0.all_modules.0.weight"
"encoder.res_unet.encoder.stages.0.0.convs.0.all_modules.0.weight"
"cls_head.fc.fc.weight"        # ← 分類ヘッド（不要）
"train_metrics.Accuracy.xxx"   # ← メトリクス（不要）

# load_pretrained_weights() が期待するキー例:
"encoder.stem.convs.0.all_modules.0.weight"
"encoder.stages.0.0.convs.0.all_modules.0.weight"
```

### なぜこうなっているか

`load_pretrained_weights()` は元々 **nnUNetベースのSSL事前学習フレームワーク**（MAE, Spark等）が出力するチェックポイントを読むために設計されています。そのフレームワークはSSL3D_classificationとは別のコードベースであり、独自の `{"network_weights": {...}}` 形式で重みを保存します。

一方、SSL3D_classification自体はPyTorch Lightningベースであり、`ModelCheckpoint` はLightning標準の `.ckpt` 形式で保存します。

**つまり、「保存する仕組み」と「読み込む仕組み」が別々に設計されているため、変換が必要**になります。

### convert_sdf_checkpoint.py が行うこと

1. Lightning `.ckpt` から `state_dict` を抽出
2. エンコーダ部分のキーのみをフィルタリング（分類ヘッド・メトリクス等を除外）
3. キー名から余分なプレフィックスを除去（`encoder.res_unet.` → `` ）
4. `{"network_weights": {...}}` 形式で `.pth` として保存

---

## Step 1: SDFデータセット生成

FDSLxSDF4Segリポジトリで実行します。

**重要**: ABIDEのpatch_sizeが `[160, 160, 160]` なので、SDFデータも **160³** で生成してエンコーダの構造を一致させます。

```bash
# FDSLxSDF4Seg リポジトリで実行
uv run python src/fdslxsdf4seg/generate_sdf_dataset_classification.py \
    --out_dir /path/to/data/sdf_classification \
    --D 160 --H 160 --W 160 \
    --samples_per_class 200 \
    --primitives sphere cylinder torus cone octahedron \
    --sdf_mappers inverse_cube linear \
    --grid_scale 0.45 \
    --dataset_name sdf_classification
```

この例では: 5プリミティブ × 2マッパー = **10クラス** × 200サンプル = **2000サンプル**

### 生成されるファイル

データセット生成時に、以下のファイルが自動的に作成されます：

- `nnUNetResEncUNetLPlans_3d_fullres/` - 学習データ（Blosc2形式）
- `labelsTr.json` - ラベル情報
- `splits_final.json` - データ分割情報
- **`sdf_classification.yaml`** - SSL3D_classification用の設定ファイル（patch_size、num_classes、data_root_dir が設定済み）

### クラス数の確認

```bash
python -c "import json; labels=json.load(open('/path/to/data/sdf_classification/labelsTr.json')); print('num_classes:', max(labels.values())+1)"
```

---

## Step 2: SDFデータで事前学習

SSL3D_classificationリポジトリで実行します。

### 環境設定

#### 1. YAMLファイルのコピー

FDSLxSDF4Segが生成した設定ファイルをSSL3D_classificationにコピーします：

```bash
# Linuxの場合
cp /path/to/data/sdf_classification/sdf_classification.yaml \
   /path/to/SSL3D_classification/cli_configs/data/sdf_classification.yaml

# Windowsの場合（PowerShell）
Copy-Item C:\path\to\data\sdf_classification\sdf_classification.yaml `
          C:\path\to\SSL3D_classification\cli_configs\data\sdf_classification.yaml
```

このファイルには以下が既に設定されています：
- `num_classes`: 生成したクラス数
- `patch_size`: データ生成時に指定したサイズ
- `data_root_dir`: データセットのパス（`${data_dir}` 変数を使用）

#### 2. 環境変数の設定

`cli_configs/env/local.yaml` の `data_dir` を設定:

```yaml
# cli_configs/env/local.yaml
data_dir: '/path/to/data/sdf_classification'
exp_dir: '/path/to/experiments'
```

> `data_dir` は生成したデータセットのルートディレクトリ（`nnUNetResEncUNetLPlans_3d_fullres/`, `labelsTr.json`, `splits_final.json` が直下にある場所）を指定。

#### 3. コピーしたYAMLの確認

`cli_configs/data/sdf_classification.yaml` の内容を確認:

```yaml
  num_classes: 10        # ← 生成時のクラス数が自動設定済み
  patch_size: [160, 160, 160]  # ← 生成時のサイズが自動設定済み
  data_root_dir: ${data_dir}   # ← 環境変数を参照
```

必要に応じて `batch_size` や `accumulate_grad_batches` などの学習パラメータを調整できます（またはコマンドラインでオーバーライド可能）。

### 学習の実行

```bash
# ResEncoder で事前学習
python main.py \
    data=sdf_classification \
    model=resenc \
    env=local \
    data.num_classes=10 \
    data.patch_size="[160,160,160]" \
    data.module.batch_size=2 \
    trainer.devices=1 \
    trainer.max_epochs=200 \
    trainer.enable_checkpointing=True
```

チェックポイントは `<exp_dir>/<teamname>/classification/sdf_classification/checkpoints/<uid>/<fold>/` 以下に保存されます。

---

## Step 3: チェックポイント変換

Lightning `.ckpt` を `load_pretrained_weights()` が読める `.pth` 形式に変換します。

```bash
python convert_sdf_checkpoint.py \
    --input_path /path/to/checkpoints/<epoch>-<val_loss>.ckpt \
    --output_path ./pretrained_sdf/sdf_resenc_pretrained.pth \
    --model_type resenc
```

Primusモデルの場合:

```bash
python convert_sdf_checkpoint.py \
    --input_path /path/to/checkpoints/<epoch>-<val_loss>.ckpt \
    --output_path ./pretrained_sdf/sdf_primus_pretrained.pth \
    --model_type primus
```

変換スクリプトは抽出されたキー名と形状を表示するので、内容を確認できます。

---

## Step 4: ABIDEでファインチューニング

```bash
# SDF事前学習済み重みでABIDEファインチューニング
python main.py \
    data=abide_1mm_cropped_160_new \
    model=resenc \
    env=local \
    model.pretrained=True \
    model.chpt_path=./pretrained_sdf/sdf_resenc_pretrained.pth \
    model.finetune_method=full \
    data.module.batch_size=2 \
    trainer.devices=1 \
    trainer.max_epochs=200
```

### finetune_method の選択

| メソッド | 説明 | 用途 |
|---------|------|------|
| `full` | エンコーダ含む全パラメータを更新 | 一般的なファインチューニング |
| `linear_probing` | エンコーダを凍結し分類ヘッドのみ更新 | 事前学習の特徴表現の品質を評価 |

```bash
# Linear probing で特徴表現の品質を評価
python main.py \
    data=abide_1mm_cropped_160_new \
    model=resenc \
    env=local \
    model.pretrained=True \
    model.chpt_path=./pretrained_sdf/sdf_resenc_pretrained.pth \
    model.finetune_method=linear_probing \
    data.module.batch_size=2 \
    trainer.devices=1 \
    trainer.max_epochs=200
```

---

## 比較実験の設計

### 実験条件

| 実験名 | 事前学習データ | 事前学習方法 | ファインチューニング | コマンド要点 |
|--------|-------------|------------|------|------|
| **Baseline** | なし | - | ABIDE | `model.pretrained=False` |
| **SSL pretrain** | OpenMind 114k | MAE / Spark 等 | ABIDE | `model.chpt_path=<ssl_ckpt>.pth` |
| **SDF pretrain** | SDF合成データ | 分類タスク | ABIDE | `model.chpt_path=<sdf_ckpt>.pth` |

### 実行コマンド一覧

```bash
# (A) Baseline: 事前学習なし
python main.py \
    data=abide_1mm_cropped_160_new \
    model=resenc \
    env=local \
    model.pretrained=False \
    data.module.batch_size=2 \
    trainer.devices=1 \
    trainer.max_epochs=200

# (B) SSL事前学習（論文の手法）
python main.py \
    data=abide_1mm_cropped_160_new \
    model=resenc \
    env=local \
    model.pretrained=True \
    model.chpt_path=/path/to/ssl_pretrained_checkpoint.pth \
    data.module.batch_size=2 \
    trainer.devices=1 \
    trainer.max_epochs=200

# (C) SDF事前学習（full finetuning）
python main.py \
    data=abide_1mm_cropped_160_new \
    model=resenc \
    env=local \
    model.pretrained=True \
    model.chpt_path=./pretrained_sdf/sdf_resenc_pretrained.pth \
    model.finetune_method=full \
    data.module.batch_size=2 \
    trainer.devices=1 \
    trainer.max_epochs=200

# (D) SDF事前学習（linear probing）
python main.py \
    data=abide_1mm_cropped_160_new \
    model=resenc \
    env=local \
    model.pretrained=True \
    model.chpt_path=./pretrained_sdf/sdf_resenc_pretrained.pth \
    model.finetune_method=linear_probing \
    data.module.batch_size=2 \
    trainer.devices=1 \
    trainer.max_epochs=200
```

### 評価指標

ABIDEの設定（`abide_1mm_cropped_160_new.yaml`）で定義されているメトリクス:
- **F1 Score** (macro)
- **Balanced Accuracy**
- **Average Precision (AP)**
- **AUROC**

---

## 注意事項

### patch_size の統一

SDFデータとABIDEデータで **同じ `patch_size`** を使用してください。エンコーダの入出力次元が一致しないと重みの読み込みに失敗します。

- ABIDE: `[160, 160, 160]`
- SDF: `[160, 160, 160]` に設定 ← 合わせる

### クラス数の違い

SDF（例: 10クラス）→ ABIDE（2クラス）でクラス数が異なりますが、`load_pretrained_weights()` は分類ヘッド（`.seg_layers.` を含むキー）を自動的にスキップするため問題ありません。ABIDEファインチューニング時に新しい分類ヘッドがランダム初期化されます。

### GPUメモリ

160³のボリュームはメモリ消費が大きいため:

- `batch_size`: 1〜2
- `accumulate_grad_batches`: 48〜96（実効バッチサイズを確保）

```bash
# メモリが足りない場合（batch_sizeを1に減らし、accumulate_grad_batchesで実効バッチサイズを確保）
python main.py \
    data=abide_1mm_cropped_160_new \
    model=resenc \
    env=local \
    data.module.batch_size=1 \
    trainer.devices=1 \
    trainer.accumulate_grad_batches=96 \
    trainer.max_epochs=200
```

### Cross Validation

ABIDEの設定はデフォルトで `cv.k=3`（3-fold CV）です。公平な比較のため、全foldの結果を報告してください。
