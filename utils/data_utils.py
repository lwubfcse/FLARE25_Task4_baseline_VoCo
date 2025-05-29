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

import math
import os
from copy import deepcopy
import numpy as np
import torch
import pickle
from monai import data, transforms
from monai.data import *
from monai.transforms import *
from torch.utils.data import DataLoader, ConcatDataset
from utils.data_trans import *
from utils.voco_trans import VoCoAugmentation


data_dir = "./data/"
cache_dir = './data/cache/'

FLARE25_dir = data_dir + 'FLARE-Task4-CT-FM/'
FLARE25_json = "./jsons/FLARE25_Task4.json"
FLARE25_list = load_decathlon_datalist(FLARE25_json, True, "training", base_dir=FLARE25_dir)
FLARE25_cache_dir = cache_dir + 'FLARE25'


def get_loader(args):
    train_ds = get_ds(args)

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    return train_loader


def get_ds(args):
    if args.distributed:
        print_cond = (args.rank == 0)
    else:
        print_cond = True

    if print_cond:
        print("Total number of data: {}".format(len(FLARE25_list)))
        print('---' * 20)

    base_trans_without_label = get_abdomen_trans_without_label(args)

    if args.cache:
        FLARE25_ds = PersistentDataset(data=FLARE25_list,
                                    transform=base_trans_without_label,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=FLARE25_cache_dir)
    else:
        FLARE25_ds = Dataset(data=FLARE25_list,transform=transforms.Compose(base_trans_without_label))


    return FLARE25_ds
