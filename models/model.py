import os
import open_clip
import clip
import torch
from torch import nn
from torch.nn import functional as F

from modules import (clip_forward, encode_image, ChannelAlignLayer,
                     MultiHeadMapAttention)


class CLIPModel(nn.Module):
    def __init__(self, name, pretrained=None, num_class=4):
        super(CLIPModel, self).__init__()
        self.name = name
        if pretrained:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(name,
                                                                                   pretrained=pretrained,
                                                                                   device="cpu")
        else:
            self.model, self.preprocess = clip.load(name, device="cpu")
        # self.output_layer = Sequential(
        #     nn.Linear(1024, 1280),
        #     nn.GELU(),
        #     nn.Linear(1280, 512),
        # )
        # self.fc = nn.Linear(512, num_class)
        # self.text_input = clip.tokenize(['Real', 'Synthetic'])
        self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        # self.text_input = clip.tokenize(['Real-Photo', 'Synthetic-Photo', 'Real-Painting', 'Synthetic-Painting'])
        # self.text_input = clip.tokenize(['a', 'b', 'c', 'd'])

    def forward(self, image_input, training=True):
        if training:
            logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
            return None, logits_per_image
        else:
            image_feats = self.clip_model.encode_image(image_input)
            image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
            return None, image_feats


class CLipClassifierWMap(nn.Module):
    """
    Version 6 from LaRE
    """

    def __init__(self, name, pretrained=None, num_class=4):
        super(CLipClassifierWMap, self).__init__()
        if pretrained:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(name,
                                                                                   pretrained=pretrained,
                                                                                   device="cpu")
        else:
            self.model, self.preprocess = clip.load(name, device="cpu")
        # self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        self.fc = nn.Linear(1024 + 1024 + 1024, 2)
        # self.visual_attnpool = self.clip_model.visual.attnpool
        # Monkey patch
        self.clip_model.visual.forward = clip_forward
        self.clip_model.encode_image = encode_image

        self.conv = nn.Conv2d(4, 8, kernel_size=(1, 1))  # for 8 heads
        self.conv_align = nn.Conv2d(4, 1024, kernel_size=(1, 1))
        self.attn_pool = MultiHeadMapAttention()
        self.channel_align = ChannelAlignLayer(4, 128, 1024)

    def forward(self, image_input, loss_map):
        # loss_map bs * 4 * 32 * 32
        image_feats, block3_feats = self.clip_model.encode_image(self.clip_model, image_input)
        # block3 bs * 1024 * 14 * 14
        aligned_loss_map = F.adaptive_avg_pool2d(loss_map, (14, 14))  # bs * 4 * 14 * 14
        pooled_loss_map = self.conv(aligned_loss_map)  # bs * 8 * 14 * 14
        pooled_block3_feats = self.attn_pool(block3_feats, pooled_loss_map)  # bs * 1024

        channel_weighted_feats = self.channel_align(block3_feats, loss_map)  # bs * 1024
        logits = self.fc(torch.cat([image_feats, pooled_block3_feats, channel_weighted_feats], dim=1))
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
