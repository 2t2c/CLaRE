from yacs.config import CfgNode

from .model import CLIPClassifierWMap, CLIPModel, CustomCLIP

VALID_NAMES = [
    # OpenAI CLIP
    "CLIP:RN50",
    "CLIP:RN101",
    "CLIP:RN50x4",
    "CLIP:RN50x16",
    "CLIP:RN50x64",
    "CLIP:ViT-B/32",
    "CLIP:ViT-B/16",
    "CLIP:ViT-L/14",
    "CLIP:ViT-L/14@336px",
    # OpenCLIP
    "OpenCLIP:ViT-L/14:datacomp_xl_s13b_b90k",
    "OpenCLIP:ViT-H/14:laion2b_s32b_b79k",
    "OpenCLIP:ViT-L/14:dfn2b",
]


def get_model(name: str, cfg: CfgNode):
    if name not in VALID_NAMES:
        raise ValueError(f"Invalid model name '{name}'")

    parts = name.split(":")
    prefix = parts[0]
    clip_type_lower = (cfg.clip_type or "").lower()

    if prefix not in {"CLIP", "OpenCLIP"}:
        raise ValueError(
            f"Unknown model prefix '{prefix}'. Expected 'CLIP' or 'OpenCLIP'."
        )

    model_name = parts[1]
    pretrained = parts[2] if prefix == "OpenCLIP" and len(parts) > 2 else None

    if clip_type_lower == "lare":
        kwargs = {"roi_pooling": cfg.roi_pooling}
        if pretrained:
            kwargs["pretrained"] = pretrained
        return CLIPClassifierWMap(model_name, **kwargs)

    if clip_type_lower == "clipping":
        if pretrained:
            return CustomCLIP(cfg, model_name, pretrained=pretrained)
        return CustomCLIP(cfg, model_name)

    if pretrained:
        return CLIPModel(model_name, pretrained=pretrained)
    return CLIPModel(model_name)
