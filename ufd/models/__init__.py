from .clip_models import CLIPModel
from .imagenet_models import ImagenetModel


VALID_NAMES = [
    'Imagenet:resnet18',
    'Imagenet:resnet34',
    'Imagenet:resnet50',
    'Imagenet:resnet101',
    'Imagenet:resnet152',
    'Imagenet:vgg11',
    'Imagenet:vgg19',
    'Imagenet:swin-b',
    'Imagenet:swin-s',
    'Imagenet:swin-t',
    'Imagenet:vit_b_16',
    'Imagenet:vit_b_32',
    'Imagenet:vit_l_16',
    'Imagenet:vit_l_32',

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
    if name.startswith("Imagenet:"):
        return ImagenetModel(name.split(":")[1]) 
    elif name.startswith("CLIP:"):
        return CLIPModel(name.split(":")[1])  
    elif name.startswith("OpenCLIP:"):
        return CLIPModel(name.split(":")[1], pretrained=name.split(":")[2])  
    else:
        assert False 
