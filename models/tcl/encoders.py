# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import copy
from typing import Union

import torch
import torch.nn as nn
from einops import rearrange

from models.builder import MODELS
from models.tcl.clip_builder import get_clip_imgenc, get_clip_textenc
from models.tcl.modules import FeatureEncoder, BLCModuleCompatibleBCHW
from models.tcl.masker import MaskerBackbone

from utils import get_logger


class LNProjLayer(BLCModuleCompatibleBCHW):
    """Apply layer norm & projection for 1d or 2d inputs.
    """
    def __init__(self, ln: Union[None, nn.LayerNorm], proj: Union[None, torch.Tensor]):
        super().__init__()
        self.ln = ln
        self.proj = proj

    def forward_blc(self, x):
        if self.ln is not None:
            x = self.ln(x)
        if self.proj is not None:
            x = x @ self.proj

        return x


@MODELS.register_module()
class CLIPImageFeatureEncoder(FeatureEncoder):
    def clone_masker_backbone(self, freeze_idx):
        backbone = MaskerBackbone(self.clip_visual, freeze_idx)

        return backbone

    def clone_proj(self):
        return copy.deepcopy(self.clip_proj)

    def __init__(self, model_name: str, feature_extract_index: int, ignore_last_attn: bool):
        super().__init__()
        # build clip_visual
        clip_visual = get_clip_imgenc(model_name)

        # build clip_proj (ln_post, proj)
        # move both LN & proj
        self.clip_proj = LNProjLayer(clip_visual.ln_post, clip_visual.proj)
        clip_visual.ln_post = nn.Identity()
        clip_visual.proj = None

        self.clip_visual = clip_visual
        self.patch_size = self.clip_visual.patch_size
        self.output_dim = self.clip_visual.output_dim
        self.ignore_last_attn = ignore_last_attn

        # add feature hook
        for resblock in self.clip_visual.transformer.resblocks[feature_extract_index:]:
            resblock.hook_handler = resblock.register_forward_hook(self.hook)

    def _encode(self, x, spatial=True, ignore_last_attn=None):
        if ignore_last_attn is None:
            ignore_last_attn = self.ignore_last_attn

        # x [B, C, H, W]
        H, W = x.shape[-2:]
        x = self.clip_visual(
            x,
            spatial=spatial,
            ignore_last_attn=ignore_last_attn
        )  # [B, L, C]

        if spatial:
            clip_features = x
            _B, L, _C = clip_features.size()
            h = H // self.patch_size
            w = W // self.patch_size
            assert h*w == L-1, f"x {x.shape}, L {L}, h {h}, w{w}"

            if H % self.patch_size or W % self.patch_size:
                logger = get_logger()
                logger.error(
                    f"!!! Input image {x.shape} does not fit to patch size {self.patch_size} !!!"
                )

            x = rearrange(clip_features[:, 1:], "B (H W) C -> B C H W", H=h, W=w)

        return x

    def clip_forward(self, x, ret_feats=False):
        x = self.forward(x, spatial=False, ignore_last_attn=False, ret_feats=ret_feats)
        if ret_feats:
            return self.clip_proj(x[0]), x[1]

        return self.clip_proj(x)

    def maskclip_forward(self, x, ret_feats=False):
        return self.forward(x, spatial=True, ignore_last_attn=True, ret_feats=ret_feats)

    def tcl_forward(self, x, ret_feats=False):
        """This function is same as `forward()` itself.
        """
        return self.forward(
            x,
            spatial=True,
            ignore_last_attn=self.ignore_last_attn,
            ret_feats=ret_feats
        )


@MODELS.register_module()
class CLIPTextEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.clip_text = get_clip_textenc(model_name)

    def forward(self, x):
        return self.clip_text(x)
