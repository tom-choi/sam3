"""Small helpers for Live2D SAM3 smoke fine-tuning."""

from __future__ import annotations

import logging
from collections.abc import Iterable

from sam3 import build_sam3_image_model


def build_live2d_smoke_model(
    *,
    bpe_path: str,
    checkpoint_path: str,
    load_from_HF: bool = False,
    enable_segmentation: bool = True,
    device: str = "cpu",
    eval_mode: bool = False,
    trainable_prefixes: Iterable[str] = ("segmentation_head.",),
):
    """Build SAM3 and freeze everything except selected modules."""

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=checkpoint_path,
        load_from_HF=load_from_HF,
        enable_segmentation=enable_segmentation,
        device=device,
        eval_mode=eval_mode,
    )

    trainable_prefixes = tuple(trainable_prefixes)
    trainable_params = 0
    frozen_params = 0
    for name, parameter in model.named_parameters():
        should_train = name.startswith(trainable_prefixes)
        parameter.requires_grad_(should_train)
        if should_train:
            trainable_params += parameter.numel()
        else:
            frozen_params += parameter.numel()

    logging.info(
        "Live2D smoke model trainable prefixes=%s, trainable=%d, frozen=%d",
        trainable_prefixes,
        trainable_params,
        frozen_params,
    )
    return model
