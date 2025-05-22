import os
import open_clip
import clip
import torch
from torch import nn
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from clip import clip

import types
# LaRE Modules
from .modules import (clip_resnet_forward, clip_vit_forward, clip_encode_image, 
                      ChannelAlignLayer, MultiHeadMapAttention, 
                      MultiHeadMapAttentionV2, ROIPooler)
# CLIPping modules
from .modules import (PromptLearner, TextEncoder)

### LaRE Models ###
class CLIPModel(nn.Module):
    def __init__(self, name, pretrained=None):
        super(CLIPModel, self).__init__()
        self.name = name
        if pretrained:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                name, pretrained=pretrained, device="cpu"
            )
        else:
            self.clip_model, self.preprocess = clip.load(name, device="cpu")
        self.text_input = clip.tokenize(["Real Photo", "Fake Photo"])

    def forward(self, image_input, training=True):
        if training:
            logits_per_image, _ = self.clip_model(
                image_input, self.text_input.to(image_input.device)
            )
            return None, logits_per_image
        else:
            image_feats = self.clip_model.encode_image(image_input)
            image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
            return None, image_feats


class CLIPClassifierWMap(nn.Module):
    """
    Version 6 from LaRE model.py
    """

    def __init__(self, name, pretrained=None, num_classes=2, roi_pooling=False):
        super(CLIPClassifierWMap, self).__init__()
        self.multiplier = 3
        if pretrained:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
                name, pretrained=pretrained, device="cpu"
            )
        else:
            self.clip_model, self.preprocess = clip.load(name, device="cpu")
        self.text_input = clip.tokenize(["Real Photo", "Fake Photo"])
        self.name = name
        self.image_proj = nn.Linear(768, 1024) if 'ViT' in self.name else nn.Identity()
        # overriding the methods of clip
        # self.clip_model.visual.forward = clip_forward
        # self.clip_model.encode_image = clip_encode_image
        # bind custom forward properly
        # self.clip_model.visual.forward = types.MethodType(clip_forward, self.clip_model.visual)
        self.clip_model.encode_image = types.MethodType(clip_encode_image, self.clip_model)
        if self.name.__contains__('ViT'):
            self.clip_model.visual.forward = types.MethodType(clip_vit_forward, self.clip_model.visual)
            self.attn_pool = MultiHeadMapAttention(spacial_dim=16)
            # self.attn_pool = MultiHeadMapAttentionV2(spacial_dim=16)
        else:
            self.clip_model.visual.forward = types.MethodType(clip_resnet_forward, self.clip_model.visual)
            self.attn_pool = MultiHeadMapAttention(spacial_dim=14)
            # self.attn_pool = MultiHeadMapAttentionV2(spacial_dim=16)
        self.clip_model.encode_image = types.MethodType(clip_encode_image, self.clip_model)
        # conv. + attention + alignment
        self.conv = nn.Conv2d(4, 8, kernel_size=(1, 1))  # for 8 heads
        self.conv_align = nn.Conv2d(4, 1024, kernel_size=(1, 1))
        self.channel_align = ChannelAlignLayer(4, 128, 1024)
        # roi pooling
        self.roi_pooling = roi_pooling
        if self.roi_pooling:
            self.roi_pool = ROIPooler(output_size=(14, 14), align=True)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.multiplier = 4
        # projection
        self.fc = nn.Linear(1024 * self.multiplier, num_classes)

    def forward(self, image, loss_map, rois=None):
        # image - (B,C,H,W) | loss_map - (B,4,32,32)
        # image_feats, block3_feats = self.clip_model.encode_image(self.clip_model, image)
        image_feats, block3_feats = self.clip_model.encode_image(image)
        # project ViT image features from 768 -> 1024 if needed
        image_feats = self.image_proj(image_feats)

        # block3_feats - RN:50 (B,1024,14,14) | Vit: (B,1024,16,16)
        if self.name.__contains__('ViT'):
            aligned_loss_map = F.adaptive_avg_pool2d(loss_map, (16, 16))  # (B,4,16,16)
        else:
            aligned_loss_map = F.adaptive_avg_pool2d(loss_map, (14, 14))  # (B,4,14,14)
        pooled_loss_map = self.conv(aligned_loss_map)  # (B,8,14/16,14/16)
        pooled_block3_feats = self.attn_pool(block3_feats, pooled_loss_map)  # (B,1024)
        channel_weighted_feats = self.channel_align(block3_feats, loss_map)  # (B,1024)

        # in case BS is 1, for debugging
        if pooled_block3_feats.dim() == 1:
            pooled_block3_feats = pooled_block3_feats.unsqueeze(0)
        if channel_weighted_feats.dim() == 1:
            channel_weighted_feats = channel_weighted_feats.unsqueeze(0)

        if self.roi_pooling:
            # ROI pooling
            roi_feats = self.roi_pool(block3_feats, rois)  # (N_rois, 1024, 14, 14)
            roi_feats = self.pool(roi_feats).squeeze(-1).squeeze(-1)  # (N_rois, 1024)

            # aggregate ROI features per image
            batch_size = image.shape[0]
            aggregated_roi_feats = []
            roi_offset = 0
            for b in range(batch_size):
                num_rois = sum(r[0].item() == b for r in rois)
                if num_rois == 0:
                    aggregated_roi_feats.append(torch.zeros_like(image_feats[b]))
                else:
                    feats = roi_feats[roi_offset : roi_offset + num_rois]
                    aggregated_roi_feats.append(feats.mean(dim=0))
                    roi_offset += num_rois
            roi_feats = torch.stack(aggregated_roi_feats)  # (B, 1024)
            features = torch.cat(
                [image_feats, pooled_block3_feats, channel_weighted_feats, roi_feats],
                dim=1,
            )  # (B, 1024*multiplier)
        else:
            features = torch.cat(
                [image_feats, pooled_block3_feats, channel_weighted_feats], dim=1
            )  # (B, 1024*multiplier)

        logits = self.fc(features)

        return logits

### CLIPping Models ###
class CustomCLIP(nn.Module):
    """
    CLLPing model that uses prompt learning.
    """
    def __init__(self, cfg, name, pretrained=None):
        super().__init__()
        self.name = name
        if pretrained:
            self.clip_model, _, _ = open_clip.create_model_and_transforms(name,
                                                                          pretrained=pretrained,
                                                                          device="cpu")
        else:
            self.clip_model, _ = clip.load(name, device="cpu")
        if cfg.clipping.coop.prec in ["fp32", "amp"]:
            self.clip_model.float()
        self.classes = ["real", "fake"]
        self.prompt_learner = PromptLearner(cfg.clipping, self.classes, self.clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = self.clip_model.visual
        self.text_encoder = TextEncoder(self.clip_model)
        self.logit_scale = self.clip_model.logit_scale
        self.dtype = self.clip_model.dtype

    def forward(self, image):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
