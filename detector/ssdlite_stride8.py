"""SSDLite MobileNetV3 with an extra stride-8 detection level (docs/DECISIONS.md #23 option B, #26).

Stock SSDLite's finest feature map is stride 16 (the expansion layer of
MobileNetV3's stride-16 -> 32 block), so its smallest anchor is 0.2 of
the input: no FPV drone in VisioDECT fits it (#25). This variant also
taps the expansion layer of the stride-8 -> 16 block, the same way
SSDLite taps the stride-16 one (MobileNetV3 paper, section 6.3), giving
7 levels instead of 6: at 640 input, 80x80, 40x40, 20x20, 10x10, 5x5,
3x3, 2x2.

Anchor scales are geometric (`GEOMETRIC_SCALES`), not SSDLite's linear
min_ratio..max_ratio spread: with 7 levels, linear spacing from 0.025
leaves apparent sizes ~4-14% of the frame without any anchor at IoU >= 0.5
(#25, 42.5% of FPV drones matchable); geometric spacing closes that gap
(74.7%). `eval/anchor_coverage.py --architecture ssdlite_stride8` checks
this on the model's real anchors.

Weights: every backbone block and the 4 extra blocks keep their
COCO-pretrained SSDLite weights (the modules are reused, not copied).
The stock regression head's 6 levels move to levels 1-6; level 0
(stride 8) starts fresh. The classification head is new, as in the stock
single-class model (detector/train.py's build_single_class_model).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead, SSDLiteRegressionHead

# geomspace(0.025, 0.95, 8) to 6 decimals: 7 level scales plus the extra one
# DefaultBoxGenerator uses for each level's sqrt(s_k * s_k+1) box.
GEOMETRIC_SCALES = (0.025, 0.042036, 0.070682, 0.118848, 0.199836, 0.336014, 0.564989, 0.95)
NUM_LEVELS = 7


class Stride8FeatureExtractor(nn.Module):
    """Stock SSDLite extractor, with its first stage split at the
    stride-8 -> 16 block's expansion layer into two outputs."""

    def __init__(self, stock_extractor: nn.Module) -> None:
        super().__init__()
        first, second = stock_extractor.features
        *blocks, c4_expansion = list(first)  # first = backbone[:c4_pos] + backbone[c4_pos].block[0]
        strided = [i for i, b in enumerate(blocks) if getattr(b, "_is_cn", False)]
        c3_pos = strided[-1]  # the last strided block before C4: stride 8 -> 16
        if blocks[c3_pos].use_res_connect:
            raise ValueError("stride-8 tap block must not have a residual connection")
        self.features = nn.Sequential(
            nn.Sequential(*blocks[:c3_pos], blocks[c3_pos].block[0]),  # stride 8: C3 expansion layer
            nn.Sequential(blocks[c3_pos].block[1:], *blocks[c3_pos + 1 :], c4_expansion),  # stride 16, as stock
            second,  # stride 32, as stock
        )
        self.extra = stock_extractor.extra

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output = []
        for block in self.features:
            x = block(x)
            output.append(x)
        for block in self.extra:
            x = block(x)
            output.append(x)
        return OrderedDict([(str(i), v) for i, v in enumerate(output)])


def build_stride8_model(
    input_size: int = 640,
    num_classes: int = 2,
    anchor_scales: tuple[float, ...] | list[float] = GEOMETRIC_SCALES,
    pretrained: bool = True,
):
    """Single-class SSDLite MobileNetV3 with the stride-8 level. See module docstring."""
    if len(anchor_scales) != NUM_LEVELS + 1:
        raise ValueError(f"anchor_scales needs {NUM_LEVELS + 1} values (7 levels + 1), got {len(anchor_scales)}")
    if pretrained:
        stock = ssdlite320_mobilenet_v3_large(weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    else:
        stock = ssdlite320_mobilenet_v3_large(weights=None, weights_backbone=None)

    backbone = Stride8FeatureExtractor(stock.backbone)
    anchor_generator = DefaultBoxGenerator([[2, 3]] * NUM_LEVELS, scales=list(anchor_scales))
    out_channels = det_utils.retrieve_out_channels(backbone, (320, 320))
    num_anchors = anchor_generator.num_anchors_per_location()
    norm_layer = nn.BatchNorm2d

    regression_head = SSDLiteRegressionHead(out_channels, num_anchors, norm_layer)
    for level, module in enumerate(stock.head.regression_head.module_list):
        regression_head.module_list[level + 1] = module
    classification_head = SSDLiteClassificationHead(out_channels, num_anchors, num_classes, norm_layer)
    stock.head.regression_head = regression_head
    stock.head.classification_head = classification_head

    return SSD(
        backbone,
        anchor_generator,
        (input_size, input_size),
        num_classes,
        image_mean=list(stock.transform.image_mean),
        image_std=list(stock.transform.image_std),
        head=stock.head,
        score_thresh=stock.score_thresh,
        nms_thresh=stock.nms_thresh,
        detections_per_img=stock.detections_per_img,
        topk_candidates=stock.topk_candidates,
    )
