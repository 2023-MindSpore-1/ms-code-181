# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Dataset for train and eval."""
import os
import copy
import cv2
import numpy as np

import mindspore.dataset as de
from mindspore.communication.management import init, get_rank, get_group_size

from .augmemtation import preproc
from .utils import bbox_encode


class WiderFace():
    def __init__(self, label_path):
        self.images_list = []  # 所有图片路径组成的列表
        # FIXME
        self.labels_list = []  # 三维度列表，每张图片的数字组成的二维列表叠加
        f = open(label_path, 'r')
        lines = f.readlines()
        First = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if First is True:
                    First = False
                else:
                    c_labels = copy.deepcopy(labels)
                    self.labels_list.append(c_labels)  # 三维度列表，每张图片的数字组成的二维列表叠加
                    labels.clear()  # labels = []
                # remove '# '
                path = line[2:]
                path = label_path.replace('label.txt', 'images/') + path

                assert os.path.exists(path), 'image path is not exists.'

                self.images_list.append(path)  # 所有图片路径组成的列表
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)  # 二维列表，把每一行数字作为一个列表添加到二维列表中。当前图片的每行数字
        # add the last label
        self.labels_list.append(labels)

        # del bbox which width is zero or height is zero
        for i in range(len(self.labels_list) - 1, -1, -1):  # n-1, n-2, ... , 1, 0。n为图片数量
            labels = self.labels_list[i]  # 当前图片的数字组成的二维列表
            for j in range(len(labels) - 1, -1, -1):  # 依次取当前图片的每一行数字
                label = labels[j]  # 当前图片的当前行数字
                if label[2] <= 0 or label[3] <= 0:  # 若该行中第3个或第4个数字为非正数，则删除该行数字
                    labels.pop(j)
            if not labels:  # 若没有数字，则删除当前图片的相关信息
                self.images_list.pop(i)
                self.labels_list.pop(i)
            else:
                self.labels_list[i] = labels

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, item):
        return self.images_list[item], self.labels_list[item]

def read_dataset(img_path, annotation):
    cv2.setNumThreads(2)  # ////////////////////////////2

    if isinstance(img_path, str):
        img = cv2.imread(img_path)
    else:
        img = cv2.imread(img_path.tostring().decode("utf-8"))

    labels = annotation
    anns = np.zeros((0, 15))
    if labels.shape[0] <= 0:
        return anns
    for _, label in enumerate(labels):
        ann = np.zeros((1, 15))

        # get bbox
        ann[0, 0:2] = label[0:2]  # x1, y1
        ann[0, 2:4] = label[0:2] + label[2:4]  # x2, y2

        # get landmarks
        ann[0, 4:14] = label[[4, 5, 7, 8, 10, 11, 13, 14, 16, 17]]

        # set flag
        if (ann[0, 4] < 0):
            ann[0, 14] = -1
        else:
            ann[0, 14] = 1

        anns = np.append(anns, ann, axis=0)
    target = np.array(anns).astype(np.float32)

    return img, target


def create_dataset(data_dir, cfg, batch_size=32, repeat_num=1, shuffle=True, multiprocessing=True, num_worker=16):
    dataset = WiderFace(data_dir)

    # init("nccl")  # ///////////////////////////////////
    device_num, rank_id = _get_rank_info()
    if device_num == 1:
        de_dataset = de.GeneratorDataset(dataset, ["image", "annotation"],
                                         shuffle=shuffle,
                                         num_parallel_workers=num_worker)
    else:
        de_dataset = de.GeneratorDataset(dataset, ["image", "annotation"],
                                         shuffle=shuffle,
                                         num_parallel_workers=num_worker,
                                         num_shards=device_num,
                                         shard_id=rank_id)

    aug = preproc(cfg['image_size'])
    encode = bbox_encode(cfg)

    def union_data(image, annot):
        i, a = read_dataset(image, annot)
        i, a = aug(i, a)
        out = encode(i, a)

        return out

    de_dataset = de_dataset.map(input_columns=["image", "annotation"],
                                output_columns=["image", "truths", "conf", "landm"],
                                column_order=["image", "truths", "conf", "landm"],
                                operations=union_data,
                                python_multiprocessing=multiprocessing,
                                num_parallel_workers=num_worker)

    de_dataset = de_dataset.batch(batch_size, drop_remainder=True)
    de_dataset = de_dataset.repeat(repeat_num)


    return de_dataset


def _get_rank_info():
    """
    get rank size and rank id
    """
    rank_size = int(os.environ.get("RANK_SIZE", 1))

    if rank_size > 1:
        from mindspore.communication.management import get_rank, get_group_size
        rank_size = get_group_size()
        rank_id = get_rank()
    else:
        rank_size = rank_id = None

    return rank_size, rank_id
