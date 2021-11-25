import torch

torch.hub.load('pytorch/vision:v0.10.0',
               'deeplabv3_resnet101', pretrained=True)
