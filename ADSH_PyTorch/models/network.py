import torch.nn as nn
import torch
from torch.hub import load_state_dict_from_url
import torchvision.models as models
import os


class Resnext50(nn.Module):
    def __init__(self, n_classes=90, code_length=48):
        super().__init__()
        self.model = models.resnext50_32x4d(pretrained=True)
        for para in self.model.parameters():
            para.requires_grad = False
        
        self.hash_layer = nn.Sequential(
            nn.Linear(self.model.fc.in_features, self.model.fc.in_features),
            nn.ReLU(),
            nn.Linear(self.model.fc.in_features, code_length),
            nn.Tanh(),
        )
        
        self.model.fc = self.hash_layer
        
        # resnet.fc = nn.Sequential(
        #     nn.Dropout(p=0.2),
        #     nn.Linear(in_features=resnet.fc.in_features, out_features=n_classes)
        # )
        
        # self.base_model = resnet
        # self.sigm = nn.Sigmoid()

    def forward(self, x):
        return self.model(x)


def load_model(code_length=84, load=False):
    """
    Load CNN model.

    Args
        code_length (int): Hashing code length.
        load: eval for lsh, train for ADSH
    Returns
        model (torch.nn.Module): CNN model.
    """
    # model = AlexNet(code_length)
    # state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth')
    # model.load_state_dict(state_dict, strict=False)
    model = Resnext50(code_length=code_length)
    model.train()
    
    if load:
        model = torch.load(os.path.join('ADSH_PyTorch/checkpoints', 'model.t'))
        model.eval()

    return model


class AlexNet(nn.Module):

    def __init__(self, code_length):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096 ,1000),
        )

        self.classifier = self.classifier[:-1]
        self.hash_layer = nn.Sequential(
            nn.Linear(4096, code_length),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        x = self.hash_layer(x)
        return x
