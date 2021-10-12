import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.container import Sequential
import torch.functional as f

class ConvCell(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvCell, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding = 1, stride=stride, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1, padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, X):
        return self.conv_stack(X)

def make_upscale_downscale_pair(downscale_in_features, downscale_out_features, upscale_in_features):
    downscale = ConvCell(downscale_in_features, downscale_out_features)
    upscale = nn.ModuleDict({  "upsample_conv":nn.ConvTranspose2d(upscale_in_features, downscale_out_features, 2, 2), 
                "conv_cell":ConvCell(downscale_out_features*2, downscale_in_features)})
    return downscale, upscale

class BaseUNet(nn.Module):
    def __init__(self, features=[128, 256, 512]):
        assert(len(features) >= 2), "Features len should be at least 2"
        super(BaseUNet, self).__init__()
        self.bottleneck = ConvCell(features[-1], features[-1]*2)
        self.downscale_layers = nn.ModuleList([])
        in_channels = features[0]
        for feature in features[1:]:
            self.downscale_layers.append( ConvCell(in_channels, feature) )
            in_channels = feature

        self.upscale_layers = nn.ModuleList([nn.ModuleDict({  "upsample_conv":nn.ConvTranspose2d(features[-1]*2, features[-1], 2, 2), 
                "conv_cell":ConvCell(features[-1]*2, features[-2])})])

        for index in range(len(features)-2, 0, -1):
            self.upscale_layers.append(nn.ModuleDict({  "upsample_conv":nn.ConvTranspose2d(features[index], features[index], 2, 2), 
                "conv_cell":ConvCell(features[index]*2, features[index-1])}))

        self.downscale = nn.MaxPool2d(2)

    def downscale_pass(self, X):
        scales = []
        for layer in self.downscale_layers:
            X = layer(X)
            scales.append(X)
            X = self.downscale(X)

        return scales, X

    def upscale_pass(self, X, scales):
        for matching_res, item in zip(reversed(scales), self.upscale_layers):
            upscale = item["upsample_conv"]
            cell = item["conv_cell"]

            X = upscale(X)
            X = torch.cat((matching_res, X), dim=1)
            X = cell(X)
        return X

    def forward(self, X):
        downscale_res, X = self.downscale_pass(X)
        X = self.bottleneck(X)
        X = self.upscale_pass(X, downscale_res)
        return X

class DynamicUNet(nn.Module):
    def __init__(self, in_channels, out_classes, base_features):
        super(DynamicUNet, self).__init__()
        
        self.base_unet = BaseUNet(base_features)
        self.features = base_features
        
        self.in_channels = in_channels
        self.out_channels = out_classes

        self.downscale = nn.MaxPool2d(2)
        self.base_mapper_classifier = self.make_mapper_classifier(in_channels, out_classes, base_features[0])

    def make_mapper_classifier(self, in_channels, out_classes, num_features):
        mapper = nn.Sequential( 
            nn.Conv2d(in_channels, num_features, 3, 1, 1), 
            nn.InstanceNorm2d(num_features), 
            nn.LeakyReLU(0.1) 
        )
        classifier = nn.Conv2d(num_features, out_classes, 1)
        # downscale, upscale = make_upscale_downscale_pair(num_features, self.features[0], self.features[0])
        return nn.ModuleDict({"mapper":mapper, "classifier":classifier})

    # def downscale_pass(self, X):
    #     scales = []
    #     for layer in self.downscale_layers:
    #         X = layer(X)
    #         scales.append(X)
    #         X = self.downscale(X)

    #     return scales, X

    # def upscale_pass(self, X, scales):
    #     for matching_res, item in zip(reversed(scales), self.upscale_layers):
    #         upscale = item["upsample_conv"]
    #         cell = item["conv_cell"]

    #         X = upscale(X)
    #         X = torch.cat((matching_res, X), dim=1)
    #         X = cell(X)
    #     return X

    # def forward(self, X):
    #     downscale_res, X = self.downscale_pass(X)
    #     X = self.bottleneck(X)
    #     X = self.upscale_pass(X, downscale_res)
    #     return self.classifier(X)

    # def calculate_train_loss(self, X, y, loss_fn=nn.CrossEntropyLoss()):
    #     scales, X = self.downscale_pass(X)
    #     bottleneck_res = self.bottleneck(X)
    #     X = self.upscale_pass(bottleneck_res, scales)
    #     y_hat = self.classifier(X)
    #     loss = loss_fn(y_hat, y)
    #     return loss

    def forward(self, X):
        mapped = self.base_mapper_classifier["mapper"](X)
        X = self.base_unet(mapped)
        return self.base_mapper_classifier["classifier"](X)