# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import scipy.ndimage as ndimage
import torch
import os
import SimpleITK as sitk
from tqdm import tqdm
from scipy.ndimage import label as connect_label


def read(img, transpose=False):
    img = sitk.ReadImage(img)
    direction = img.GetDirection()
    origin = img.GetOrigin()
    Spacing = img.GetSpacing()

    img = sitk.GetArrayFromImage(img)
    if transpose:
        img = img.transpose(1, 2, 0)

    return img, direction, origin, Spacing


def check_acc_volume_level(pred_path, label_path):
    ls = os.listdir(pred_path)
    total_num = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    for i in tqdm(ls):
        if i.endswith('.nii.gz'):
            pred = read(os.path.join(pred_path, i))[0]
            label = read(os.path.join(label_path, i))[0]
            total_num += 1

            if label.sum() > 0:
                if (pred*label).sum() > 0:
                    TP += 1
                else:
                    FN += 1
            else:
                if pred.sum() > 125:
                    FP += 1
                else:
                    TN += 1
    print('TP, TN, FP, FN:', TP, TN, FP, FN)
    accuracy = (TP+TN)/total_num
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    print('accuracy, sensitivity, specificity, precision, recall, f1_score: ',
          accuracy, sensitivity, specificity, precision, recall, f1_score)


def check_acc_case_level(pred_path, label_path):
    ls = os.listdir(pred_path)
    total_num = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    for i in tqdm(ls):
        if i.endswith('.nii.gz'):
            pred = read(os.path.join(pred_path, i))[0]
            label = read(os.path.join(label_path, i))[0]

            if label.sum() > 0:
                labeled_matrix, num_features = connect_label(label)
                for i in range(1, num_features + 1):
                    total_num += 1

                    cur_case = labeled_matrix.copy()
                    cur_case[labeled_matrix == i] = 1
                    cur_case[labeled_matrix != i] = 0
                    # print('cur_case: ', cur_case.shape, np.unique(cur_case))

                    if (pred*cur_case).sum() > 0:
                        TP += 1
                    else:
                        FN += 1

            else:
                total_num += 1
                if pred.sum() > 125:
                    FP += 1
                else:
                    TN += 1

    print('TP, TN, FP, FN:', TP, TN, FP, FN)
    accuracy = (TP+TN)/total_num
    sensitivity = TP / (TP + FN + 1e-6)
    specificity = TN / (TN + FP + 1e-6)
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
    print('accuracy, sensitivity, specificity, precision, recall, f1_score: ',
          accuracy, sensitivity, specificity, precision, recall, f1_score)


def check_each_dataset_dice(pred_path, label_path):
    ls = os.listdir(pred_path)
    dataset_keys = {
                'Adrenal': [],
                'Chest_coronacases': [],
                'Chest_LIDC-IDRI': [],
                'Chest_MSD_lung': [],
                'Chest_NSCLC-Radiogenomics': [],
                'Chest_NSCLC-Radiomics': [],
                'Chest_volume-covid19': [],
                'Chest_NSCLCPleuralEffusion': [],
                'HCC': [],
                'KiTS23': [],
                'MSD_colon': [],
                'MSD_hepaticvessel': [],
                'MSD_liver': [],
                'MSD_pancreas': [],
                'Panorama': [],
                'WAWTACE': []
        }

    for i in tqdm(ls):
        pred = read(os.path.join(pred_path, i))[0]
        label = read(os.path.join(label_path, i))[0]

        if label.sum() > 0:
            case_dice = dice(pred, label)
            print(i, case_dice)

            for key in dataset_keys:
                if key in i:
                    dataset_keys[key].append(case_dice)

    for key in dataset_keys:
        dice_list = dataset_keys[key]
        print(key, np.mean(dice_list))


def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

        cmap[19] = np.array([0, 0, 0])
        cmap[255] = np.array([0, 0, 0])

    return cmap


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load(model, model_dict):
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    elif "network_weights" in model_dict.keys():
        state_dict = model_dict["network_weights"]
    elif "net" in model_dict.keys():
        state_dict = model_dict["net"]
    else:
        state_dict = model_dict

    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)

    if "backbone." in list(state_dict.keys())[0]:
        print("Tag 'backbone.' found in state dict - fixing!")
    for key in list(state_dict.keys()):
        state_dict[key.replace("backbone.", "")] = state_dict.pop(key)

    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

    current_model_dict = model.state_dict()
    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}

    model.load_state_dict(new_state_dict, strict=True)
    print("Using VoCo pretrained backbone weights !!!!!!!")

    return model