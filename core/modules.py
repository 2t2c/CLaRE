"""
Helper modules for model.py
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MultiScaleRoIAlign, roi_pool, roi_align

def clip_encode_image(self, image):
    """
    Encodes the image using the visual CLIP model.
    """
    return self.visual(image.type(self.dtype))


def clip_forward(self, x):
    """
    Forward pass through the CLIP model.
    """
    def stem(x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.avgpool(x)
        return x

    x = x.type(self.conv1.weight.dtype)
    x = stem(x)
    x = self.layer1(x)
    x = self.layer2(x)
    x_l3 = self.layer3(x)
    x = self.layer4(x_l3)
    x = self.attnpool(x)

    return x, x_l3


class ROIPooler(nn.Module):
    def __init__(self, output_size=(14, 14), align=True):
        super(ROIPooler, self).__init__()
        self.output_size = output_size
        self.align = align

    def forward(self, feature_map, rois):
        # perform ROI pooling
        if self.align:
            return roi_align(feature_map, rois, output_size=self.output_size, 
                             aligned=self.align)
        else:
            return roi_pool(feature_map, rois, output_size=self.output_size)

class PoolWithMap(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x, loss_map):
        # x bs * 1024 * 14 * 14
        # loss_map bs * 1024 * 14 * 14
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC

        loss_map = loss_map.flatten(start_dim=2).permute(2, 0, 1)
        loss_map = torch.cat([loss_map.mean(dim=0, keepdim=True), loss_map], dim=0)
        loss_map = loss_map + self.positional_embedding[:, None, :].to(loss_map.dtype)  # (HW+1)NC

        queries = torch.cat([x[:1], loss_map[:1]], dim=0)  # 2 * N * C

        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ChannelAlignLayer(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ChannelAlignLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, mid_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, out_dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, feature_map, loss_map):
        b, c, _, _ = feature_map.size()
        pooled_feature_map = self.avg_pool(feature_map).squeeze()  # bs * 1024
        pooled_loss_map = self.avg_pool(loss_map).squeeze()  # bs * 1024
        weights = self.fc(pooled_loss_map)
        out = pooled_feature_map * weights
        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        out_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ScaledDotProductWithMapAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_in, d_out, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductWithMapAttention, self).__init__()
        self.d_in = d_in
        self.fc_q = nn.Linear(d_in, h * d_k)
        self.fc_k = nn.Linear(d_in, h * d_k)
        self.fc_v = nn.Linear(d_in, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_out)
        self.dropout = nn.Dropout(dropout)

        self.d_out = d_out
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, loss_map, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        w_g = loss_map.permute(0, 2, 1).unsqueeze(2)  # bs * 8 * 1 * 197
        w_a = att

        # w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        # print(w_a.shape)
        # print(w_g.shape)
        w_mn = torch.softmax(w_a + w_g, -1)  ## bs * 8 * r * r

        att = self.dropout(w_mn)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class MultiHeadMapAttention(nn.Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_in=1024, d_out=1024, d_k=64, d_v=64, h=8, dropout=.1, identity_map_reordering=False,
                 spacial_dim=14):
        super(MultiHeadMapAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductWithMapAttention(d_in=d_in, d_out=d_out, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, d_in) / d_in ** 0.5)  # 197 * d_in

    def forward(self, feature_map, loss_map, attention_mask=None, attention_weights=None):
        # feature_map bs * 1024 * 14 * 14
        # loss_map bs * 8 * 14 * 14
        x = feature_map.flatten(start_dim=2)  # bs * 1024 * 196
        x = x.permute(0, 2, 1)  # bs * 196 * 1024
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # bs * 197 * 1024

        loss_map = loss_map.flatten(start_dim=2)  # bs * 8 * 196
        loss_map = loss_map.permute(0, 2, 1)  # bs * 196 * 8
        loss_map = torch.cat([loss_map.mean(dim=1, keepdim=True), loss_map], dim=1)  # bs * 197 * 8

        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # bs * 197 * 1024
        queries = x[:, :1]
        keys = x
        values = x
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, loss_map, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, loss_map, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out.squeeze()
