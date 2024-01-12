import torch
import torch.nn as nn

# ======================= CNN from scratch (CNNFS) =======================
class Pneumonia(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Pneumonia, self).__init__()
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

        self.classifier = nn.Sequential(
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
        x = self.classifier(x)
        return x
    

# ======================= Fine-tune Resnet50 =======================
class Pneumonia_resnet(nn.Module):
    def __init__(self, pretrained, in_channels, num_classes):
        super(Pneumonia_resnet, self).__init__()
        # Use a pretrained model
        self.resnet = pretrained
        # Replace model classifier
        self.resnet.fc = nn.Sequential(nn.Linear(2048, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, num_classes))
        # self.resnet.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        logits = self.resnet(x)
        return logits
    
# If input is not duplicated
class Pneumonia_resnet2(nn.Module):
    def __init__(self, pretrained, in_channels, num_classes):
        super(Pneumonia_resnet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=3),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        # Use a pretrained model
        self.resnet = pretrained
        
        for param in self.resnet.parameters():
            param.requires_grad = False
            
        # Replace model classifier
        self.resnet.fc = nn.Sequential(nn.Linear(2048, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, num_classes))

        # self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        logits = self.resnet(x)
        #logits = self.classifier(x)
        return logits
