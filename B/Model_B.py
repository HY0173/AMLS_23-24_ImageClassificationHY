import torch
import torch.nn as nn

# Method 1
class ResnetPath1(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResnetPath1, self).__init__()
        self.resnet = pretrained
        # Freeze all layers of feature extractor
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Replace model classifier
        self.resnet.fc = nn.Linear(2048, num_classes)

        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        logits = self.resnet(x)
        return logits
    
# Method 2 
class ResnetPath2(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResnetPath2, self).__init__()
        self.resnet = pretrained
        # Replace model classifier
        self.resnet.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        logits = self.resnet(x)
        return logits
    
# Method 3
class ResnetPath3(nn.Module):
    def __init__(self, pretrained, num_classes):
        super(ResnetPath3, self).__init__()
        self.resnet = pretrained
        # Replace model classifier
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        logits = self.resnet(x)
        return logits


# CNN From scratch
class CNNPath(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Path, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x