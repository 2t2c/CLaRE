"""
Helper modules for model.py
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import MultiScaleRoIAlign, roi_pool, roi_align
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

def clip_encode_image(self, image, visual_prompts=None):
    """
    Encodes the image using the visual CLIP model.
    """
    if visual_prompts is not None:
        return self.visual(image.type(self.dtype), visual_prompts)
    else:
        return self.visual(image.type(self.dtype))


def clip_resnet_forward(self, x):
    """
    Modified Forward pass through the CLIP ResNet model.
    :returns:
        - Global CLS-like embedding of shape (B, 1024)
        - Spatial feature map from layer3 of shape (B, 1024, 14, 14)
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

def clip_vit_forward(self, x: torch.Tensor, visual_prompts: torch.Tensor = None):
    """
    Modified Forward pass through the CLIP ResNet model.
    :returns:
        - Global CLS token feature (B, 1024)
        - Spatial patch features reshaped to (B, 1024, 14, 14) for map attention
    """
    x = self.conv1(x)  # shape = (B, C, H_patch, W_patch)
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, C, N)
    x = x.permute(0, 2, 1)  # (B, N, C)

    if visual_prompts is not None:
        batch_size = x.shape[0]
        visual_prompts = visual_prompts.expand(batch_size, -1, -1)  # (B, num_prompts, C)
        # prepend visual prompts (B, num_prompts, C)
        num_prompts = visual_prompts.shape[1]
        x = torch.cat([visual_prompts, x], dim=1)

    # CLS token
    cls_token = self.class_embedding.to(x.dtype)
    cls_tokens = cls_token.expand(x.shape[0], 1, -1)
    x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, C)

    # adjust positional embeddings
    pos_embed = self.positional_embedding.to(x.dtype)  # (1 + N, C)
    if visual_prompts is not None:
        cls_pos = pos_embed[:1]
        patch_pos = pos_embed[1:]
        prompt_pos = patch_pos.mean(dim=0, keepdim=True).repeat(num_prompts, 1)
        prompt_pos += torch.randn_like(prompt_pos) * 0.02  # optional noise
        extended_pos_embed = torch.cat([cls_pos, prompt_pos, patch_pos], dim=0)
    else:
        extended_pos_embed = pos_embed

    x = x + extended_pos_embed.unsqueeze(0)
    x = self.ln_pre(x)

    x = x.permute(1, 0, 2)  # NLD -> LND
    x = self.transformer(x)
    x = x.permute(1, 0, 2)  # LND -> NLD

    cls_feats = x[:, 0, :]  # (B, C)
    patch_tokens = x[:, 1 + (visual_prompts.shape[1] if visual_prompts is not None else 0):, :] # (B, N, C)

    # reshape patch tokens to 2D feature map (B, C, H, W)
    num_patches = patch_tokens.shape[1]
    h = w = int(num_patches ** 0.5)
    assert h * w == num_patches, f"Number of patches ({num_patches}) must be a perfect square"
    patch_map = patch_tokens.permute(0, 2, 1).reshape(x.shape[0], -1, h, w)  # (B, C, H, W)

    cls_feats = self.ln_post(cls_feats)
    if self.proj is not None:
        cls_feats = cls_feats @ self.proj

    return cls_feats, patch_map


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
        self.d_in = d_in
        self.d_out = d_out
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductWithMapAttention(d_in=self.d_in, d_out=self.d_out, 
                                                          d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, self.d_in) / self.d_in ** 0.5)  # 197 * d_in

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

        
class ScaledDotProductWithMapAttentionV2(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_in, d_out, d_k, d_v, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        d_in=1024, d_out=1024, d_k=64, d_v=64, h=8, dropout=.1, identity_map_reordering=False,
                 spacial_dim=14
        '''
        super(ScaledDotProductWithMapAttentionV2, self).__init__()
        self.d_in = d_in   # d_in=1024
        self.fc_q = nn.Linear(d_in, h * d_k) # 1024 --> 512
        self.fc_k = nn.Linear(d_in, h * d_k) # 1024 --> 512
        self.fc_v = nn.Linear(d_in, h * d_v) # 1024 --> 512
        self.fc_o = nn.Linear(h * d_v, d_out) # 512 --> 1024
        self.dropout = nn.Dropout(dropout)

        self.d_out = d_out  # d_out=1024
        self.d_k = d_k   # d_k=64
        self.d_v = d_v  # d_v=64
        self.h = h  # h=8

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
        b_s, nq = queries.shape[:2] # queries dim=bs*1*1024 and  so nq=1
        nk = keys.shape[1] # keys dim=bs*197*1024 , so nk=197

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        # bs*1*1024 --> bs*1*512 --> bs*1*8*64 --> bs*8*1*64
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        # bs*197*1024--> bs*197*512 --> bs*197*8*64 --> bs*8*64*197
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        # bs*197*1024 --> bs*1*512 --> bs*197*8*64 --> bs*8*197*64

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        # (bs*8*1*64) X (bs*8*64*197) ---> (bs*8*1*197)
        if attention_weights is not None:
            att = att * attention_weights

        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)

        # w_g = loss_map.permute(0, 2, 1).unsqueeze(2)  # bs * 8 * 1 * 197
        # bs*197*8 --> bs*8*197 --> bs*8*1*197
        # w_a = att

        # w_mn = torch.log(torch.clamp(w_g, min=1e-6)) + w_a
        # print(w_a.shape)
        # print(w_g.shape)
        # w_mn = torch.softmax(w_a + w_g, -1)  ## bs * 8 * r * r
        
        w_mn = torch.softmax(att, -1)

        att = self.dropout(w_mn)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        # (bs*8*1*197) X (bs*8*197*64) --> (bs*8*1*64) --> (bs*1*8*64) --> (bs*1*512)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        # (bs*1*512) --> (bs*1*1024)
        return out


class MultiHeadMapAttentionV2(nn.Module):
    """
    Formula: MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
    where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

    These are the dimensions expected before attention mechanism for feature_map (bs*1024*14*14) and loss_map (bs*8*14*14)
    I mean the image features after image encoder are (bs*1024*14*14) and the loss map are (bs*4*32*32)
    then loss_map feature are passed through adaptive ppoling to get dim as (bs*4*14*14)
    after that we also increase the channel size to 8 and the final dimensions are (bs*8*14*14)
    """
    def __init__(self, d_in=1024, d_out=1024, d_k=64, d_v=64, h=8, dropout=.1, identity_map_reordering=False,
                 spacial_dim=14):
        super(MultiHeadMapAttentionV2, self).__init__()
        
        self.identity_map_reordering = identity_map_reordering
        self.attention = ScaledDotProductWithMapAttentionV2(d_in=d_in, d_out=d_out, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        # self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, d_in) / d_in ** 0.5)  # 197 * d_in
        self.projection_loss_map_1= nn.Conv2d(8,32, kernel_size=(1, 1))
        self.projection_loss_map_2= nn.Conv2d(32,64, kernel_size=(1, 1))
        self.projection_loss_map_3= nn.Conv2d(64,128, kernel_size=(1, 1))
        self.projection_loss_map_4= nn.Conv2d(128,256, kernel_size=(1, 1))
        self.projection_loss_map_5= nn.Conv2d(256,1024, kernel_size=(1, 1))
        # Separate positional embeddings for query (loss map) and key/value (image features)
        self.pos_embed_kv = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, d_in) / d_in ** 0.5)  # 197 x 1024
        self.pos_embed_q = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, d_in) / d_in ** 0.5)    # 197 x 128
        
    def forward(self, feature_map, loss_map, attention_mask=None, attention_weights=None):
        # feature_map bs * 1024 * 14 * 14
        # loss_map bs * 8 * 14 * 14

        # In the next code line i am trying to increase the channels of the loss_map from 8 to 128
        loss_map = self.projection_loss_map_5(self.projection_loss_map_4(self.projection_loss_map_3(self.projection_loss_map_2(self.projection_loss_map_1(loss_map)))))
        # bs*8*14*14 --> bs*32*14*14 --> bs*64*14*14 --> bs*128*14*14 --> bs*256*14*14 --> bs*1024*14*14
        x = feature_map.flatten(start_dim=2)  # bs * 1024 * 196
        x = x.permute(0, 2, 1)  # bs * 196 * 1024
        # In the next line the global vector is added in the begining
        x = torch.cat([x.mean(dim=1, keepdim=True), x], dim=1)  # bs * 197 * 1024

        loss_map = loss_map.flatten(start_dim=2)  # bs * 1024 * 196
        loss_map = loss_map.permute(0, 2, 1)  # bs * 1024 * 128
        loss_map = torch.cat([loss_map.mean(dim=1, keepdim=True), loss_map], dim=1)  # bs * 197 * 1024

        x = x + self.pos_embed_kv[None, :, :].to(x.dtype)  # bs * 197 * 1024
        loss_map = loss_map + self.pos_embed_q[None, :, :].to(loss_map.dtype)  # For queries: bs * 197 * 1024
        queries = loss_map[:, :1] # bs*1*1024 global vector for query
        keys = x  # bs*197*1024
        values = x # bs*197*1024
        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries) 
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, loss_map, attention_mask, attention_weights)
            # out dim = (bs*1*1024)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, loss_map, attention_mask, attention_weights)
            # out dim = (bs*1*1024)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out.squeeze()  # out dim = (bs*1024)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, self.n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = (
                x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)]
                @ self.text_projection
        )

        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.num_classes = len(classnames)
        self.n_ctx = cfg.coop.n_ctx
        self.ctx_init = cfg.coop.ctx_init
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.simple_tokenizer = _Tokenizer()

        if self.ctx_init:
            # use given words to initialize context vectors
            ctx_init = self.ctx_init.replace("_", " ")
            self.n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(self.dtype)
            ctx_vectors = embedding[0, 1: 1 + self.n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.coop.csc:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(self.num_classes, self.n_ctx, self.ctx_dim, dtype=self.dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(self.n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {self.n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(self.simple_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(self.dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + self.n_ctx:, :])  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.coop.class_token_position

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.num_classes, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,  # (n_cls, self.n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.self.n_ctx // 2
            prompts = []
            for i in range(self.num_classes):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i_half1 = ctx[i: i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i: i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i_half1,  # (1, self.n_ctx//2, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i_half2,  # (1, self.n_ctx//2, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.num_classes):
                name_len = self.name_lens[i]
                prefix_i = prefix[i: i + 1, :, :]
                class_i = suffix[i: i + 1, :name_len, :]
                suffix_i = suffix[i: i + 1, name_len:, :]
                ctx_i = ctx[i: i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,  # (1, name_len, dim)
                        ctx_i,  # (1, self.n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            logger.error("Invalid 'class_token_position' defined!")

        return prompts


class VisualPromptLearner(nn.Module):
    def __init__(self,
                 num_prompts: int = 4,   # number of learnable visual tokens
                 prompt_dim: int = 1024,  # dimensionality of each prompt
                 dropout: float = 0.1,
                 condition_on_input: bool = False,  # If True, make prompts image-conditioned
                 condition_dim: int = 512           # Dim of conditional features (e.g., RoI or landmark features)
                 ):
        super().__init__()
        self.num_prompts = num_prompts
        self.prompt_dim = prompt_dim
        self.condition_on_input = condition_on_input

        # learnable prompt tokens (if unconditional)
        self.visual_prompts = nn.Parameter(torch.randn(1, num_prompts, prompt_dim))

        # Optionally condition prompts on input features (e.g., facial landmarks)
        if condition_on_input:
            self.prompt_mlp = nn.Sequential(
                nn.Linear(condition_dim, prompt_dim * num_prompts),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        self.dropout = nn.Dropout(dropout)

    def forward(self, cond_feats: torch.Tensor = None):
        """
        :param:
            cond_feats: [condition_dim], optional conditioning vector (e.g., RoI-pooled or landmarks)

        :return:
            [N+num_prompts, D] sequence with prepended visual prompts
        """

        if self.condition_on_input and cond_feats is not None:
            vps = self.prompt_mlp(cond_feats)
        else:
            vps = self.visual_prompts

        vps = self.dropout(vps)
        return vps
