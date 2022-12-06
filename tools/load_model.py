import torch
import torch.nn as nn
import torchvision


def get_model(num_classes): 

    # Detecting available device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', weights='VGG19_Weights.DEFAULT', progress=True)

    # Freezing normal layers
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace last layer of VGG19 classification layer by our customized one.
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # unfreez classification layer parameters.
    for param in model.classifier.parameters():
        param.requires_grad = True


    return model, device
