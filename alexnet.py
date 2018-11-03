#pytorch vision version of AlexNet https://github.com/pytorch/vision/blob/master/torchvision/models/alexnet.py

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2), #input 224 (should have been 227)*224*3 (223+2*2-11)/4+1=55 
                                                                   #floor operation 96*55*55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #(55-3)/2+1=27 96*27*27
            nn.Conv2d(64, 192, kernel_size=5, padding=2), #(27+2*2-5)/1+1 256*27*27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #(27-3)/2+1 256*13*13
            nn.Conv2d(192, 384, kernel_size=3, padding=1), #(13+1*2-3)+1 384*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), #(13+1*2-3)+1 384*13*13
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #(13+1*2-3)+1 256*13*13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), #(13-3)/2+1 256*6*6
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096), #256*6*6*4096=37,748,736
            nn.ReLU(inplace=True),
            nn.Dropout(), #half 4096
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x) #conv layers
        x = x.view(x.size(0), 256 * 6 * 6) #x.view() equivalent to numpy .reshape(), reshape for dot product, x.size(0)  
        x = self.classifier(x) #fully connected layers
        return x


def alexnet(pretrained=False, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = AlexNet(**kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['alexnet'])) #use downloaded .pth
return model