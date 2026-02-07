# MedMNIST 3D データセットでの学習ワークフロー

このドキュメントでは、MedMNISTの3Dデータセットを使用した学習の完全なワークフローを説明します。

## 目次

1. [概要](#概要)
2. [環境セットアップ](#環境セットアップ)
3. [利用可能なデータセット](#利用可能なデータセット)
4. [基本的な実行方法](#基本的な実行方法)
5. [設定のカスタマイズ](#設定のカスタマイズ)
6. [出力とログの確認](#出力とログの確認)
7. [トラブルシューティング](#トラブルシューティング)

---

## 概要

MedMNIST 3Dは、医用画像分類のための標準化された3Dベンチマークデータセットです。このプロジェクトでは、以下の6つの3D分類データセットをサポートしています：

- **OrganMNIST3D**: 11クラスの臓器分類（多クラス分類）
- **NoduleMNIST3D**: 肺結節の良性/悪性分類（2値分類）
- **AdrenalMNIST3D**: 副腎の正常/過形成分類（2値分類）
- **FractureMNIST3D**: 3種類の肋骨骨折分類（多クラス分類）
- **VesselMNIST3D**: 血管/動脈瘤分類（2値分類）
- **SynapseMNIST3D**: 興奮性/抑制性シナプス分類（2値分類）

**データ仕様:**
- 解像度: 64×64×64（または28×28×28）
- チャンネル数: 1（グレースケール）
- データ形式: `.npz`（NumPy圧縮形式）
- スプリット: 事前定義されたtrain/val/test

---

## 環境セットアップ

### 1. 依存関係のインストール

```bash
# プロジェクトのルートディレクトリで実行
cd c:\Users\yaman\SSL3D_classification

# requirements.txtから全依存関係をインストール
pip install -r requirements.txt
```

`requirements.txt`には`medmnist>=3.0.2`が含まれており、自動的にインストールされます。

### 2. データ保存ディレクトリの準備

MedMNISTデータを保存するディレクトリを作成します：

```bash
# 例: プロジェクトルート直下にdataディレクトリを作成
mkdir data
mkdir data\medmnist
```

または、任意の場所を指定できます（実行時に`data.module.data_root_dir`パラメータで指定）。

---

## 利用可能なデータセット

### データセット一覧と特徴

| データセット | Flag | タスク | クラス数 | サンプル数（train/val/test） | 設定ファイル |
|------------|------|--------|---------|---------------------------|------------|
| **OrganMNIST3D** | `organmnist3d` | Multi-class | 11 | 971/161/610 | `organmnist3d.yaml` |
| **NoduleMNIST3D** | `nodulemnist3d` | Binary | 2 | 1158/165/310 | `nodulemnist3d.yaml` |
| **AdrenalMNIST3D** | `adrenalmnist3d` | Binary | 2 | 1188/98/298 | `adrenalmnist3d.yaml` |
| **FractureMNIST3D** | `fracturemnist3d` | Multi-class | 3 | 1027/103/240 | `fracturemnist3d.yaml` |
| **VesselMNIST3D** | `vesselmnist3d` | Binary | 2 | 1335/191/382 | `vesselmnist3d.yaml` |
| **SynapseMNIST3D** | `synapsemnist3d` | Binary | 2 | 1230/177/352 | `synapsemnist3d.yaml` |

### クラスラベル詳細

<details>
<summary>OrganMNIST3D (11クラス)</summary>

- 0: liver（肝臓）
- 1: kidney-right（右腎臓）
- 2: kidney-left（左腎臓）
- 3: femur-right（右大腿骨）
- 4: femur-left（左大腿骨）
- 5: bladder（膀胱）
- 6: heart（心臓）
- 7: lung-right（右肺）
- 8: lung-left（左肺）
- 9: spleen（脾臓）
- 10: pancreas（膵臓）

</details>

<details>
<summary>FractureMNIST3D (3クラス)</summary>

- 0: buckle rib fracture（座屈型肋骨骨折）
- 1: nondisplaced rib fracture（非転位性肋骨骨折）
- 2: displaced rib fracture（転位性肋骨骨折）

</details>

---

## 基本的な実行方法

### 1. 最小限の設定で実行

```bash
# OrganMNIST3Dで学習を開始
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist

# NoduleMNIST3Dで学習を開始
python main.py \
    data=nodulemnist3d \
    data.module.data_root_dir=./data/medmnist
```

**初回実行時の動作:**
- データが自動的にダウンロードされます（約数秒～数分）
- `data_root_dir`で指定したディレクトリに`.npz`ファイルが保存されます
- 例: `./data/medmnist/organmnist3d_64.npz`

### 2. 複数のデータセットで実験

```bash
# 全6データセットで順次実行する例
for dataset in organmnist3d nodulemnist3d adrenalmnist3d fracturemnist3d vesselmnist3d synapsemnist3d
do
    python main.py \
        data=$dataset \
        data.module.data_root_dir=./data/medmnist \
        trainer.max_epochs=100
done
```

### 3. GPUの指定

```bash
# 特定のGPUを使用
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    trainer.devices=1 \
    trainer.accelerator=gpu

# 複数GPUで並列学習（DDP）
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    trainer.devices=2 \
    trainer.accelerator=gpu \
    trainer.strategy=ddp
```

---

## 設定のカスタマイズ

### 1. データ解像度の変更

デフォルトは64×64×64ですが、28×28×28に変更可能：

```bash
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    data.module.size=28 \
    data.patch_size=[28,28,28]
```

### 2. バッチサイズと学習率の調整

```bash
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    data.module.batch_size=8 \
    model.lr=0.001 \
    trainer.accumulate_grad_batches=12
```

**メモリ不足の場合:**
- `batch_size`を2または1に減らす
- `accumulate_grad_batches`を増やして実効バッチサイズを保つ

### 3. モデルの選択

```bash
# ResNetベースのモデルを使用（デフォルト）
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    model=resenc

# PRIMUSモデルを使用
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    model=primus
```

### 4. Data Augmentationの調整

YAMLファイルを編集するか、コマンドラインでオーバーライド：

```bash
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    data.module.train_transforms.rotation_for_DA=0.785398 \
    data.module.train_transforms.mirror_axes=[0,1,2]
```

**主要なaugmentationパラメータ:**
- `rotation_for_DA`: 回転角度（ラジアン）、デフォルト0.523599（約30度）
- `mirror_axes`: ミラーリングする軸、デフォルト[0,1,2]（全軸）
- `do_dummy_2d_data_aug`: 2D風augmentation、デフォルトFalse

### 5. チェックポイント保存の有効化

```bash
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    trainer.enable_checkpointing=True
```

チェックポイントは以下に保存されます：
```
{exp_dir}/{teamname}/classification/{dataset_name}/checkpoints/{uid}/{fold}/
```

### 6. Weights & Biasesロギングの設定

```bash
# オフラインモードで実行
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    trainer.logger.offline=True

# プロジェクト名を変更
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    trainer.logger.project=MyMedMNISTExperiment
```

---

## 出力とログの確認

### 1. コンソール出力

学習開始時に以下の情報が表示されます：

```
[MedMNIST3D] Loaded organmnist3d split=train size=64: 971 samples, task=multi-class, n_channels=1
[MedMNIST3D] Loaded organmnist3d split=val size=64: 161 samples, task=multi-class, n_channels=1
[MedMNIST3D] Loaded organmnist3d split=test size=64: 610 samples, task=multi-class, n_channels=1
```

### 2. Weights & Biasesダッシュボード

デフォルトでWandBにログが送信されます：

- **メトリクス**: accuracy, f1, balanced_accuracy, loss など
- **学習曲線**: train/val loss, learning rate など
- **ハイパーパラメータ**: 全設定が自動記録

WandBダッシュボードのURL: https://wandb.ai/

### 3. ローカルログファイル

```
{exp_dir}/{teamname}/classification/{dataset_name}/
```

このディレクトリに以下が保存されます：
- WandBログ（`wandb/`）
- チェックポイント（有効化した場合）

---

## トラブルシューティング

### 問題1: データのダウンロードが失敗する

**症状:**
```
RuntimeError: Dataset not found. You can use download=True to download it
```

**解決策:**
1. インターネット接続を確認
2. `download=True`が設定されているか確認（デフォルトで有効）
3. 手動でダウンロード:
   ```bash
   python -c "from medmnist import OrganMNIST3D; OrganMNIST3D(split='train', download=True, root='./data/medmnist', size=64)"
   ```

### 問題2: CUDA Out of Memory

**症状:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**解決策:**
```bash
# バッチサイズを減らす
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    data.module.batch_size=2 \
    trainer.accumulate_grad_batches=48

# 精度を16-bit mixedに変更（デフォルト）
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    trainer.precision=16-mixed
```

### 問題3: `data_root_dir`が指定されていない

**症状:**
```
omegaconf.errors.MissingMandatoryValue: Missing mandatory value: data.module.data_root_dir
```

**解決策:**
常に`data.module.data_root_dir`を指定してください：
```bash
python main.py data=organmnist3d data.module.data_root_dir=./data/medmnist
```

### 問題4: 不正なデータセットflagを指定

**症状:**
```
ValueError: Unsupported dataset_flag 'xxx'. Supported flags: [...]
```

**解決策:**
以下のいずれかを使用してください：
- `organmnist3d`
- `nodulemnist3d`
- `adrenalmnist3d`
- `fracturemnist3d`
- `vesselmnist3d`
- `synapsemnist3d`

### 問題5: transformが機能しない

**症状:**
Data augmentationが適用されていないように見える

**解決策:**
1. `train_transforms`がNoneでないことを確認
2. YAMLファイルの設定を確認:
   ```yaml
   train_transforms:
     _target_: augmentation.policies.batchgenerators.get_training_transforms
     patch_size: ${data.patch_size}
     rotation_for_DA: 0.523599
     mirror_axes: [0, 1, 2]
     do_dummy_2d_data_aug: False
   ```

---

## 高度な使用例

### 1. ハイパーパラメータスイープ（Weights & Biases Sweeps）

`sweep_config.yaml`を作成:
```yaml
program: main.py
method: bayes
metric:
  name: val_balanced_acc
  goal: maximize
parameters:
  data:
    value: organmnist3d
  data.module.data_root_dir:
    value: ./data/medmnist
  model.lr:
    min: 0.00001
    max: 0.001
  data.module.batch_size:
    values: [2, 4, 8]
  model.weight_decay:
    min: 0.0001
    max: 0.01
```

実行:
```bash
wandb sweep sweep_config.yaml
wandb agent <sweep-id>
```

### 2. 複数データセットでのアンサンブル

各データセットで学習した後、予測を組み合わせる：

```bash
# 各データセットで学習
for dataset in organmnist3d nodulemnist3d adrenalmnist3d
do
    python main.py \
        data=$dataset \
        data.module.data_root_dir=./data/medmnist \
        trainer.enable_checkpointing=True \
        model.save_preds=True
done

# 予測結果を使ってアンサンブル（別途実装が必要）
```

### 3. 事前学習済みモデルのファインチューニング

```bash
python main.py \
    data=organmnist3d \
    data.module.data_root_dir=./data/medmnist \
    model.pretrained=True \
    model.finetune_method=full
```

---

## 参考資料

- **MedMNIST公式リポジトリ**: https://github.com/MedMNIST/MedMNIST
- **MedMNIST論文**: Yang et al. (2021, 2023)
- **データダウンロード**: https://zenodo.org/records/10519652
- **既存のワークフロー**: [WORKFLOW_SDF_PRETRAINING.md](WORKFLOW_SDF_PRETRAINING.md)

---

## 実装ファイル

- **データセット実装**: [`datasets/medmnist_3d.py`](datasets/medmnist_3d.py)
- **設定ファイル**: [`cli_configs/data/`](cli_configs/data/)
  - `organmnist3d.yaml`
  - `nodulemnist3d.yaml`
  - `adrenalmnist3d.yaml`
  - `fracturemnist3d.yaml`
  - `vesselmnist3d.yaml`
  - `synapsemnist3d.yaml`
- **メイン実行スクリプト**: [`main.py`](main.py)

---

## まとめ

MedMNIST 3Dデータセットを使用した学習の基本的なフロー：

1. ✅ 依存関係のインストール（`pip install -r requirements.txt`）
2. ✅ データ保存先ディレクトリの準備
3. ✅ データセットの選択（6種類から選択）
4. ✅ コマンド実行（`python main.py data=<dataset> data.module.data_root_dir=<path>`）
5. ✅ 結果の確認（WandBダッシュボードまたはローカルログ）

**最小限のコマンド例:**
```bash
python main.py data=organmnist3d data.module.data_root_dir=./data/medmnist
```

これでMedMNIST 3Dデータセットでの学習が開始されます！
