from .clip import clip 
from PIL import Image
import torch.nn as nn
import open_clip


CHANNELS = {
    "RN50" : 1024,
    "ViT-L/14" : 768,
    "ViT-H/14" : 1024,
    "ViT-g/14" : 1024,
}

class CLIPModel(nn.Module):
    def __init__(self, name, pretrained=None, num_classes=1):
        super(CLIPModel, self).__init__()
        self.name = name
        # self.preprecess will not be used during training, which is handled in Dataset class
        if pretrained:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(name, 
                                                                            pretrained=pretrained,
                                                                            device="cpu")
        else:
            self.model, self.preprocess = clip.load(name, device="cpu")

        # add a linear layer to the model (hard-coded for ViT)
        self.project = nn.Linear(1024, 768)
        self.fc = nn.Linear(768, num_classes)
 

    def forward(self, x, return_feature=False):
        features = self.model.encode_image(x)
        if CHANNELS.get(self.name) == 1024: 
            features = self.project(features)
        if return_feature:
            return features
        return self.fc(features)

