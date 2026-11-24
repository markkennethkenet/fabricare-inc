"""Reject photos that show no fabric (a wall, a cupboard, a door) before they are scanned.

An ImageNet-pretrained ConvNeXt-Tiny scores each photo; the "textile score" is its total probability on the 51
ImageNet classes that are clothing or fabric (velvet, wool, jean, jersey, cardigan, towel, curtain and so on).
On the team's calibration set a minimum of 0.04 kept 96% of fabric photos (n = 240) and 97% of damage close-ups
(n = 200), and rejected all three wall and cupboard photos (scores 0.016-0.031). About 3 in 100 real fabric
photos are therefore refused and must be retaken.

FABRIC_PRESENCE=off disables the check; FABRIC_PRESENCE_MIN changes the threshold. If the ImageNet weights cannot
be loaded the check is skipped rather than blocking every scan.
"""
from __future__ import annotations

import os
from threading import Lock

import numpy as np

from model_runtime import prepare_fabric_rgb

# ImageNet class indexes for clothing and fabric (velvet, wool, handkerchief, jean, gown, shower curtain, jersey,
# Windsor tie, bow tie, mailbag, backpack, cardigan, maillot x2, bath towel, overskirt, bulletproof vest, poncho,
# pajama, hoopskirt, sarong, trench coat, sweatshirt, diaper, cloak, swimming trunks, apron, miniskirt, purse,
# quilt, pillow, bib, suit, stole, fur coat, bonnet, mosquito net, sleeping bag, kimono, abaya, academic gown,
# lab coat, military uniform, vestment, brassiere, mitten, sock, Christmas stocking, theater curtain, dishrag,
# knee pad).
TEXTILE_CLASS_IDS = (
    399, 400, 411, 414, 434, 443, 452, 457, 459, 465, 474, 496, 501, 529, 533, 568, 578, 591, 601, 608, 610, 614,
    615, 617, 636, 638, 639, 652, 655, 658, 669, 689, 697, 721, 735, 748, 750, 775, 794, 797, 806, 824, 834, 841,
    842, 854, 869, 885, 887, 906, 911)
DEFAULT_MIN_TEXTILE_SCORE = 0.04
NO_FABRIC_REASON = "No fabric detected. Point the camera at the garment's fabric"

_lock = Lock()
_model = {}


def enabled():
    return os.environ.get("FABRIC_PRESENCE", "on").strip().lower() not in {"0", "off", "false", "none"}


def min_score():
    try:
        return float(os.environ.get("FABRIC_PRESENCE_MIN", DEFAULT_MIN_TEXTILE_SCORE))
    except ValueError:
        return DEFAULT_MIN_TEXTILE_SCORE


def _load():
    with _lock:
        if "net" not in _model:
            try:
                import tensorflow as tf

                _model["net"] = tf.keras.applications.ConvNeXtTiny(weights="imagenet", include_preprocessing=True)
            except Exception:
                _model["net"] = None
        return _model["net"]


def textile_score(probabilities):
    return float(np.asarray(probabilities)[list(TEXTILE_CLASS_IDS)].sum())


def check(image_path):
    """{"checked", "fabric_detected", "textile_score"}; checked is False when the check is off or unavailable."""
    if not enabled():
        return {"checked": False, "fabric_detected": True, "textile_score": None}
    net = _load()
    if net is None:
        return {"checked": False, "fabric_detected": True, "textile_score": None}
    pixels = np.asarray(prepare_fabric_rgb(image_path), dtype=np.float32)
    probabilities = np.asarray(net(np.stack([pixels, pixels[:, ::-1, :]]), training=False)).mean(axis=0)
    score = textile_score(probabilities)
    return {"checked": True, "fabric_detected": score >= min_score(), "textile_score": round(score, 4)}
