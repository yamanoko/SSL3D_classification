"""
SDF事前学習済みチェックポイントを、SSL3D_classificationのload_pretrained_weights()が
読み込める形式に変換するスクリプト。

使い方:
    python convert_sdf_checkpoint.py \
        --input_path <Lightning checkpoint path (.ckpt)> \
        --output_path <output path (.pth)> \
        --model_type resenc  # or primus

Lightning チェックポイントの state_dict キーは以下の形式:
    ResEncoder:  encoder.res_unet.encoder.xxx → encoder.xxx (nnUNet形式)
    Eva_MAE:     eva_encoder.xxx              → encoder.xxx (nnUNet形式)

load_pretrained_weights() が期待する形式:
    {"network_weights": {"encoder.xxx": tensor, ...}}
"""

import argparse
from collections import OrderedDict
from pathlib import Path

import torch


def convert_resenc_checkpoint(state_dict: dict) -> dict:
    """
    ResEncoder_Classifier の Lightning チェックポイントから
    エンコーダ重みを抽出し、nnUNet形式に変換する。

    Lightning key: encoder.res_unet.encoder.stem.convs.0.all_modules.0.weight
    Target key:    encoder.stem.convs.0.all_modules.0.weight
    """
    network_weights = OrderedDict()

    for key, value in state_dict.items():
        # エンコーダ部分のみ抽出 (classification head は除外)
        if key.startswith("encoder.res_unet."):
            # "encoder.res_unet." → "" (nnUNetの元の名前空間に戻す)
            new_key = key.replace("encoder.res_unet.", "")
            network_weights[new_key] = value

    return {"network_weights": network_weights}


def convert_primus_checkpoint(state_dict: dict) -> dict:
    """
    Eva_MAE の Lightning チェックポイントから
    エンコーダ重みを抽出し、nnUNet形式に変換する。

    Lightning key: eva_encoder.eva.blocks.0.xxx
    Target key:    encoder.eva.blocks.0.xxx
    """
    network_weights = OrderedDict()

    for key, value in state_dict.items():
        # eva_encoder 部分のみ抽出 (classification head は除外)
        if key.startswith("eva_encoder."):
            # "eva_encoder." → "encoder." (load_pretrained_weights が "encoder." を除去する)
            new_key = key.replace("eva_encoder.", "encoder.")
            network_weights[new_key] = value

    return {"network_weights": network_weights}


def main():
    parser = argparse.ArgumentParser(
        description="SDF事前学習チェックポイントをSSL3D形式に変換"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Lightning チェックポイントのパス (.ckpt)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="変換後の出力パス (.pth)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["resenc", "primus"],
        help="モデルタイプ: resenc (ResEncoder) または primus (Eva_MAE)",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint from: {args.input_path}")
    checkpoint = torch.load(args.input_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["state_dict"]

    print(f"Model type: {args.model_type}")
    print(f"Total keys in checkpoint: {len(state_dict)}")

    if args.model_type == "resenc":
        converted = convert_resenc_checkpoint(state_dict)
    elif args.model_type == "primus":
        converted = convert_primus_checkpoint(state_dict)

    print(f"Extracted encoder keys: {len(converted['network_weights'])}")

    # 抽出されたキーを表示
    for key in sorted(converted["network_weights"].keys()):
        shape = converted["network_weights"][key].shape
        print(f"  {key}: {shape}")

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, args.output_path)
    print(f"\nSaved converted checkpoint to: {args.output_path}")


if __name__ == "__main__":
    main()
