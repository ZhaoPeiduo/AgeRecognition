from torch import nn
from torchvision.models.resnet import ResNet, resnet50, resnet101, resnet152
from torchvision.models.vision_transformer import VisionTransformer, vit_b_16, vit_b_32, vit_l_16, vit_l_32, vit_h_14

class ViTAgeRecognizer(nn.Module):
    '''AgeRecongizer using ViT as backbone'''
    def __init__(self, vit_class, output_features=512, weights=None):
        super(ViTAgeRecognizer, self).__init__()
        if weights:
            try:
                self.backbone = vit_class(weights=weights)
            except Exception as e:
                print(e)
                print("There seems to be an issue with the customized weights. Using default weights instead.")
                self.backbone = vit_class(weights='DEFAULT')
        else:
            self.backbone = vit_class(weights='DEFAULT')
        assert isinstance(self.backbone, VisionTransformer), "The specified backbone class does not generate an instance of pytorch's Visiontransformer. \
                                                                Please use the provided models or make sure your customized model is a subclass of pytorch's Visiontransformer"
        input_features = self.backbone.heads.head.in_features
        # Remove original classification head
        self.backbone.heads = nn.Identity()
        self.head = nn.Linear(input_features, output_features)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

class ResNetAgeRecognizer(nn.Module):
    '''AgeRecongizer using ResNet as backbone'''
    def __init__(self, resnet_class, output_features=512, weights=None):
        super(ResNetAgeRecognizer, self).__init__()
        if weights:
            try:
                self.backbone = resnet_class(weights=weights)
            except Exception as e:
                print(e)
                print("There seems to be an issue with the customized weights. Using default weights instead.")
                self.backbone = resnet_class(weights='DEFAULT')
        else:
            self.backbone = resnet_class(weights='DEFAULT')
        assert isinstance(self.backbone, ResNet), "The specified backbone class does not generate an instance of pytorch's ResNet. \
                                                        Please use the provided models or make sure your customized model is a subclass of pytorch's ResNet"
        input_features = self.backbone.fc.in_features
        # Remove original classification head
        self.backbone.fc = nn.Identity()
        self.head = nn.Linear(input_features, output_features)

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x

def resent50_age_recongiser(**kwargs):
    return ResNetAgeRecognizer(resnet50, **kwargs)

def resent101_age_recongiser(**kwargs):
    return ResNetAgeRecognizer(resnet101, **kwargs)

def resent152_age_recongiser(**kwargs):
    return ResNetAgeRecognizer(resnet152, **kwargs)

def vit_b_16_age_recongizer(**kwargs):
    return ViTAgeRecognizer(vit_b_16, **kwargs)

def vit_b_32_age_recongizer(**kwargs):
    return ViTAgeRecognizer(vit_b_32, **kwargs)

def vit_l_16_age_recongizer(**kwargs):
    return ViTAgeRecognizer(vit_l_16, **kwargs)

def vit_l_32_age_recongizer(**kwargs):
    return ViTAgeRecognizer(vit_l_32, **kwargs)

def vit_h_14_age_recongizer(**kwargs):
    return ViTAgeRecognizer(vit_h_14, **kwargs)

