import os
import open_clip
import clip
import torch
from torch import nn
from torch.nn import functional as F
import types
from .modules import (clip_forward, clip_encode_image, ChannelAlignLayer,
                      MultiHeadMapAttention, ROIPooler)


class CLIPModel(nn.Module):
    def __init__(self, name, pretrained=None, num_class=2):
        super(CLIPModel, self).__init__()
        self.name = name
        if pretrained:
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(name,
                                                                                        pretrained=pretrained,
                                                                                        device="cpu")
        else:
            self.clip_model, self.preprocess = clip.load(name, device="cpu")
        self.text_input = clip.tokenize(["Real Photo", "Fake Photo"])

    def forward(self, image_input, training=True):
        if training:
            logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
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
            self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(name,
                                                                                        pretrained=pretrained,
                                                                                        device="cpu")
        else:
            self.clip_model, self.preprocess = clip.load(name, device="cpu")
        self.text_input = clip.tokenize(["Real Photo", "Fake Photo"])
        # overriding the methods of clip
        # self.clip_model.visual.forward = clip_forward
        # self.clip_model.encode_image = clip_encode_image
        # bind custom forward properly
        self.clip_model.visual.forward = types.MethodType(clip_forward, self.clip_model.visual)
        self.clip_model.encode_image = types.MethodType(clip_encode_image, self.clip_model)
        # conv. + attention + alignment
        self.conv = nn.Conv2d(4, 8, kernel_size=(1, 1))  # for 8 heads
        self.conv_align = nn.Conv2d(4, 1024, kernel_size=(1, 1))
        self.attn_pool = MultiHeadMapAttention()
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

        # block3 - (B,1024,14,14)
        aligned_loss_map = F.adaptive_avg_pool2d(loss_map, (14, 14))  # (B,4,14,14)
        pooled_loss_map = self.conv(aligned_loss_map)  # (B,8,14,14)
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
                    feats = roi_feats[roi_offset:roi_offset + num_rois]
                    aggregated_roi_feats.append(feats.mean(dim=0))
                    roi_offset += num_rois
            roi_feats = torch.stack(aggregated_roi_feats)  # (B, 1024)
            features = torch.cat([image_feats, pooled_block3_feats, channel_weighted_feats, roi_feats], dim=1) # (B, 1024*multiplier)
        else:
            features = torch.cat([image_feats, pooled_block3_feats, channel_weighted_feats], dim=1)  # (B, 1024*multiplier)

        logits = self.fc(features)

        return logits


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = CLIPModel().cuda()
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Params: %.2f' % (params / (1024 ** 2)))

    x = torch.zeros([4, 3, 448, 448]).cuda()
    _, logits = model(x)
    print(logits.shape)
