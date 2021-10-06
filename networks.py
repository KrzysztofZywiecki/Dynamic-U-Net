import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.container import Sequential
import torch.functional as f

class ConvCell(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvCell, self).__init__()

        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding = 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding = 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, X):
        return self.conv_stack(X)

class DynamicUNet(nn.Module):
    def __init__(self, in_channels, out_classes, features=[64, 128, 256]):
        super(DynamicUNet, self).__init__()
        self.features = features
        self.bottleneck_size = features[-1]*2

        self.downscale_layers = nn.ModuleList([])
        for feature in features:
            self.downscale_layers.append( ConvCell(in_channels, feature) )
            in_channels = feature

        self.downsample = nn.MaxPool2d(2)
        self.bottleneck = ConvCell(features[-1], features[-1]*2)

        self.upscale_layers = nn.ModuleList([nn.ModuleDict({  "upsample_conv":nn.ConvTranspose2d(features[-1]*2, features[-1], 2, 2), 
                "conv_cell":ConvCell(features[-1]*2, features[-1])})])
        for index in range(len(features)-1, 0, -1):
            self.upscale_layers.append(nn.ModuleDict({  "upsample_conv":nn.ConvTranspose2d(features[index], features[index-1], 2, 2), 
                "conv_cell":ConvCell(features[index-1]*2, features[index-1])}))

        self.classifier = nn.Conv2d(features[0], out_classes, 1)

    def make_additional_step(self, num_features):
        in_features = self.features[-1]

        bottleneck = ConvCell(num_features, num_features*2)
        downscale_layer = ConvCell(in_features, num_features)
        upscale_layer = nn.ModuleDict({"upscale_conv":nn.ConvTranspose2d(num_features*2, num_features, 2, 2),
            "conv_cell":ConvCell(num_features*2, in_features*2)})
        return {"bottleneck":bottleneck, "downscale":downscale_layer, "upscale":upscale_layer}

    def downscale_pass(self, X):
        scales = []
        for layer in self.downscale_layers:
            scale_res = layer(X)
            scales.append(scale_res)
            X = self.downsample(scale_res)

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
        return self.classifier(X)

    def calculate_train_loss(self, X, y, additional_step = None, loss_fn=nn.CrossEntropyLoss(), alpha = 1.0):
        scales, X = self.downscale_pass(X)
        bottleneck_res = self.bottleneck(X)

        if additional_step is None:
            X = self.upscale_pass(bottleneck_res, scales)
        else:
            downscale_forward = additional_step["downscale"](X)
            bottleneck_forward = additional_step["bottleneck"](self.downsample(downscale_forward))
            upsampled = additional_step["upscale"]["upscale_conv"](bottleneck_forward)
            catenation = torch.cat((upsampled, downscale_forward), dim=1)

            augmented_res = additional_step["upscale"]["conv_cell"](catenation)

            res = alpha*bottleneck_res + (1-alpha) * augmented_res

            X = self.upscale_pass(res, scales)

        y_hat = self.classifier(X)
        loss = loss_fn(y_hat, y)
        return loss
