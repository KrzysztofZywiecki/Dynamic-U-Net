import numpy as np
from torchmetrics import IoU
from torch.utils.data import Dataset
import albumentations as A
import os
import cv2 as cv
import torch

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

class SegmentationDataset(Dataset):
    def __init__(self, ids, out_scale=1, in_scale=1, patch_folder="./segmentation/patches", mask_folder="./segmentation/masks"):
        self.ids = ids
        self.mask_folder = mask_folder
        self.patch_folder = patch_folder
        self.in_scale = in_scale
        self.out_scale = out_scale
    def __len__(self):
        return len(self.ids)
    def __getitem__(self, index):
        patch_path = os.path.join(self.patch_folder, "org_{}.bmp".format(self.ids[index]))
        mask_path = os.path.join(self.mask_folder, "mask_{}.bmp".format(self.ids[index]))

        patch = cv.imread(patch_path, cv.IMREAD_GRAYSCALE)/255.0
        patch = cv.resize(patch, (int(patch.shape[0]*self.in_scale), int(patch.shape[1]*self.in_scale))) if self.in_scale != 1 else patch
        mask = cv.imread(mask_path, cv.IMREAD_GRAYSCALE)
        mask = cv.resize(mask, (int(mask.shape[0]*self.out_scale), int(mask.shape[1]*self.out_scale))) if self.out_scale != 1 else mask
        patch = np.expand_dims(patch, 0)

        return torch.tensor(patch, dtype=torch.float), torch.tensor(mask, dtype=torch.long)