# ------------------------------------------------------------------------------
# TCL
# Copyright (c) 2023 Kakao Brain. All Rights Reserved.
# ------------------------------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from models.builder import MODELS
import us


@MODELS.register_module()
class InfoNCE(nn.Module):
    def __init__(self, T_init=0.07, T_learnable=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T_init))
        if not T_learnable:
            self.logit_scale.requires_grad_(False)

    def forward(self, image_emb, text_emb):
        """
        Args:
            image_emb [B, C]: image embedding
            text_emb [B, C]: text embedding
        """
        assert image_emb.ndim == text_emb.ndim == 2

        B = image_emb.shape[0]
        # get label globally
        labels = torch.arange(B, dtype=torch.long, device=image_emb.device) + B * dist.get_rank()

        # [B, C]
        image_emb = us.normalize(image_emb, dim=-1)
        text_emb = us.normalize(text_emb, dim=-1)

        # cosine similarity
        logits_per_img = image_emb @ us.gather_cat(text_emb, grad=True).t()
        logits_per_text = text_emb @ us.gather_cat(image_emb, grad=True).t()

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = F.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = F.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)

        return loss


@MODELS.register_module()
class ExtendedInfoNCE(nn.Module):
    def __init__(self, T_init=0.07, T_learnable=True):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / T_init))
        if not T_learnable:
            self.logit_scale.requires_grad_(False)

    def forward(self, image_emb, text_emb):
        """ExtendedInfoNCE is an InfoNCE function but computes similarity map differently.

        Note:
            InfoNCE: s = einsum("ic,jc->ij", img_emb, txt_emb)
            ExtendedInfoNCE: s = einsum("ijc,jc->ij", img_emb, txt_emb)

            In practice, the implementation of ExtendedInfoNCE becomes rather complicated
            when using multi-gpu with DDP.

        Args:
            image_emb [B, N, C]: extended image embedding where N=B*D
            text_emb [B, C]: text embedding
        """
        B = image_emb.shape[0]
        # get label globally
        labels = torch.arange(B, dtype=torch.long, device=image_emb.device) + B * dist.get_rank()

        # [B, C]
        image_emb = us.normalize(image_emb, dim=-1)
        text_emb = us.normalize(text_emb, dim=-1)

        # cosine similarity
        all_text_emb = us.gather_cat(text_emb, grad=True, contiguous_grad=True)  # [N, C]
        logits_per_img = torch.einsum("bnc,nc->bn", image_emb, all_text_emb)

        n_devices = dist.get_world_size()
        rank = dist.get_rank()
        assert B * n_devices == image_emb.size(1)
        image_emb_here = image_emb.chunk(n_devices, dim=1)[rank].contiguous()  # [B, B, C]
        all_image_emb_here = us.gather_cat(image_emb_here, grad=True, contiguous_grad=True)  # [N, B, C]
        logits_per_text = torch.einsum("nbc,bc->bn", all_image_emb_here, text_emb)

        logit_scale = torch.clamp(self.logit_scale.exp(), max=100)
        loss_img = F.cross_entropy(logits_per_img * logit_scale, labels)
        loss_text = F.cross_entropy(logits_per_text * logit_scale, labels)

        loss = 0.5 * (loss_img + loss_text)

        return loss
