import torch
import torch.nn as nn
from torch.nn.modules import padding
from torch.nn.modules.container import Sequential
import torch.functional as f
from torch.nn.modules.instancenorm import InstanceNorm2d
from torchmetrics import IoU
import numpy as np

class ScaleAttentionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, scale, side_length):
        super(ScaleAttentionBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, scale, scale),
            nn.InstanceNorm2d(out_channels)
        )
        self.multihead_attention_layer = nn.MultiheadAttention(out_channels, 4, batch_first=True)
        self.unflatten = nn.Unflatten(2, (side_length//scale, side_length//scale))
        self.norm = nn.InstanceNorm2d(out_channels)

    def forward(self, X):
        scaled_patch = self.conv_layer(X)
        sequence_patch = torch.flatten(scaled_patch, 2).swapaxes(1, 2)
        attenuated = self.multihead_attention_layer(sequence_patch, sequence_patch, sequence_patch, need_weights=False)[0]
        result = self.unflatten(attenuated.swapaxes(1, 2))
        return self.norm(result)

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

def make_mapper_classifier(in_channels, out_classes, num_features):
    mapper = nn.Sequential( 
        nn.Conv2d(in_channels, num_features, 3, 1, 1, bias=False), 
        nn.InstanceNorm2d(num_features), 
        nn.LeakyReLU(0.1) 
    )
    classifier = nn.Conv2d(num_features, out_classes, 1)
    return nn.ModuleDict({"mapper":mapper, "classifier":classifier})

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
    def __init__(self, in_channels, out_classes, base_features, additional_features):
        super(DynamicUNet, self).__init__()
        
        self.features = base_features
        self.in_channels = in_channels
        self.out_channels = out_classes
        self.level = 0
        
        self.downscale = nn.MaxPool2d(2)

        self.base_unet = BaseUNet(base_features)
        self.base_mapper_classifier = nn.ModuleDict({"mapper":
            nn.Sequential( 
                nn.Conv2d(in_channels, base_features[0], 3, 1, 1, bias=False), 
                nn.InstanceNorm2d(base_features[0]), 
                nn.LeakyReLU(0.1) 
            ), "classifier":nn.Conv2d(base_features[0], out_classes, 1)})

        self.additional_mapper_classifiers = nn.ModuleList([])
        self.additional_upscale = nn.ModuleList([])
        self.additional_downscale = nn.ModuleList([])

        input_size = base_features[0]
        for feature in additional_features:
            downscale, upscale = make_upscale_downscale_pair(feature, input_size, input_size)
            mappers = make_mapper_classifier(in_channels, out_classes, feature)

            self.additional_upscale.append(upscale)
            self.additional_downscale.append(downscale)
            self.additional_mapper_classifiers.append(mappers)

            input_size = feature

    def freeze_base_unet(self):
        self.base_mapper_classifier.requires_grad_(False)
        self.base_unet.requires_grad_(False)
    def unfreeze_base_unet(self):
        self.base_mapper_classifier.requires_grad_(True)
        self.base_unet.requires_grad_(True)

    def use_layers(self, number):
        if number > len(self.additional_downscale):
            number = len(self.additional_downscale)
        self.level = number

    def use_higher_layer(self): # Method returns True if trying to use level that is higher than num of levels: if training is finished
        if self.level == 0:
            self.base_unet.requires_grad_(False)
            self.base_mapper_classifier.requires_grad_(False)
            self.level += 1
            return False
        elif self.level < len(self.additional_downscale):
            self.additional_downscale[self.level-1].requires_grad_(False)
            self.additional_upscale[self.level-1].requires_grad_(False)
            self.additional_mapper_classifiers[self.level-1].requires_grad_(False)
            self.level += 1
            return False
        return True

    def forward(self, X):
        if self.level != 0:
            X = self.additional_mapper_classifiers[self.level - 1]["mapper"](X)
        else:
            X = self.base_mapper_classifier["mapper"](X)

        results = []
        for i in range(self.level, 0, -1):
            X = self.additional_downscale[i - 1](X)
            results.append(X)
            X = self.downscale(X)

        X = self.base_unet(X)

        for i in range(self.level):
            X = self.additional_upscale[i]["upsample_conv"](X)
            X = torch.cat([X, results[self.level - i - 1]], dim=1)
            X = self.additional_upscale[i]["conv_cell"](X)

        if self.level != 0:
            return self.additional_mapper_classifiers[self.level - 1]["classifier"](X)
        else:
            return self.base_mapper_classifier["classifier"](X)

class LearnableScaleDynamicUNet(nn.Module):
    def __init__(self, in_channels, out_classes, base_features, base_scale, additional_features):
        super(LearnableScaleDynamicUNet, self).__init__()
        
        self.features = base_features
        self.in_channels = in_channels
        self.out_channels = out_classes
        self.level = 0
        
        self.downscale = nn.MaxPool2d(2)

        self.base_scale = base_scale
        self.base_unet = BaseUNet(base_features)
        self.base_mapper_classifier = nn.ModuleDict({"mapper":
        ScaleAttentionBlock(in_channels, base_features[0], base_scale, 128),
            # nn.Sequential( 
            #     nn.Conv2d(in_channels, base_features[0], base_scale, stride=base_scale, bias=False), 
            #     nn.InstanceNorm2d(base_features[0]), 
            #     nn.LeakyReLU(0.1) 
            # ), 
            "classifier":nn.Conv2d(base_features[0], out_classes, 1)})

        self.additional_upscale = nn.ModuleList([])
        self.additional_downscale = nn.ModuleList([])
        self.additional_scale = nn.ModuleList([])
        self.additional_classifiers = nn.ModuleList([])
        self.scales = []

        input_size = base_features[0]
        for feature in additional_features:
            downscale, upscale = make_upscale_downscale_pair(feature[0], input_size, input_size)
            self.additional_scale.append(nn.Sequential(
                nn.Conv2d(in_channels, feature[0], feature[1], feature[1]),
                nn.InstanceNorm2d(feature[0]), 
                nn.LeakyReLU(0.1) 
            ) if feature[1] > 1 else nn.Sequential(
                nn.Conv2d(in_channels, feature[0], 3, 1, 1),
                nn.InstanceNorm2d(feature[0]), 
                nn.LeakyReLU(0.1) 
            )) 
            self.additional_classifiers.append(nn.Conv2d(feature[0], out_classes, 1))

            self.additional_upscale.append(upscale)
            self.additional_downscale.append(downscale)
            self.scales.append(feature[1])

            input_size = feature[0]

    def freeze_base_unet(self):
        self.base_mapper_classifier.requires_grad_(False)
        self.base_unet.requires_grad_(False)
    def unfreeze_base_unet(self):
        self.base_mapper_classifier.requires_grad_(True)
        self.base_unet.requires_grad_(True)

    def use_layers(self, number):
        if number > len(self.additional_downscale):
            number = len(self.additional_downscale)
        self.level = number

    def use_higher_layer(self): # Method returns True if trying to use level that is higher than num of levels: if training is finished
        if self.level == 0:
            self.base_unet.requires_grad_(False)
            self.base_mapper_classifier.requires_grad_(False)
            self.level += 1
            return False
        elif self.level < len(self.additional_downscale):
            self.additional_downscale[self.level-1].requires_grad_(False)
            self.additional_upscale[self.level-1].requires_grad_(False)
            self.additional_classifiers[self.level-1].requires_grad_(False)
            self.additional_scale[self.level-1].requires_grad_(False)
            self.level += 1
            return False
        return True

    def forward(self, X):
        if self.level != 0:
            X = self.additional_scale[self.level - 1](X)
        else:
            X = self.base_mapper_classifier["mapper"](X)

        results = []
        for i in range(self.level, 0, -1):
            X = self.additional_downscale[i - 1](X)
            results.append(X)
            X = self.downscale(X)

        X = self.base_unet(X)

        for i in range(self.level):
            X = self.additional_upscale[i]["upsample_conv"](X)
            X = torch.cat([X, results[self.level - i - 1]], dim=1)
            X = self.additional_upscale[i]["conv_cell"](X)

        if self.level != 0:
            return self.additional_classifiers[self.level - 1](X)
        else:
            return self.base_mapper_classifier["classifier"](X)


def dice_score(y_pred, y_true, reduction = "mean"): # Calculates dice score for binary prediction (expects prediction to be two values per pixel)
    pred_label = y_pred[:, 1] > y_pred[:, 0]
    iou = IoU(num_classes=2)
    scores =  np.array([iou(pred_label[i], y_true[i]) for i in range(pred_label.size()[0])  ]) # iou(pred_label.long(), y_true)


    if reduction == "mean":
        return scores.mean()
    if reduction == "sum":
        return scores.sum()
    else:
        return scores