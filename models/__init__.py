from .model import CLIPModel

VALID_NAMES = [
    # OpenAI CLIP
    'CLIP:RN50',
    'CLIP:RN101',
    'CLIP:RN50x4',
    'CLIP:RN50x16',
    'CLIP:RN50x64',
    'CLIP:ViT-B/32',
    'CLIP:ViT-B/16',
    'CLIP:ViT-L/14',
    'CLIP:ViT-L/14@336px',

    # OpenCLIP
    'OpenCLIP:ViT-L/14:datacomp_xl_s13b_b90k',
    'OpenCLIP:ViT-H/14:laion2b_s32b_b79k',
    'OpenCLIP:ViT-L/14:dfn2b',
]


def get_model(name):
    assert name in VALID_NAMES
    if name.startswith("CLIP:"):
        return CLIPModel(name.split(":")[1])
    elif name.startswith("OpenCLIP:"):
        return CLIPModel(name.split(":")[1], pretrained=name.split(":")[2])
    else:
        assert False
