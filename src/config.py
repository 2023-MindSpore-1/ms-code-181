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
"""Config for train and eval."""
cfg_res50 = {
    'device_target': "GPU", # "Ascend",
    'device_id': 0,  # ////
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'class_weight': 1.0,
    'landm_weight': 1.0,
    'batch_size': 8,  # 8
    'num_workers': 16,
    'num_anchor': 29126,  # 29126
    # FIXME nNPU
    'nnpu': 1,
    'ngpu': 1,  # 8P
    'image_size': 840,
    'match_thresh': 0.35,

    # opt
    'optim': 'sgd',  # 'sgd'. 'momentum'
    'momentum': 0.9,
    'weight_decay': 5e-4,  # 5e-4
    'loss_scale': 1,  # 1, 1024 will make loss 40+

    # seed
    'seed': 1,

    # lr
    'epoch': 80,  # 100
    'T_max': 40,
    'eta_min': 0.0,
    'decay1': 35,  # 70
    'decay2': 60,  # 90
    'lr_type': 'dynamic_lr',  # 'dynamic_lr', cosine_annealing
    'initial_lr': 0.01,  # 0.001
    'warmup_epoch': 5,  # dynamic_lr: -1, cosine_annealing:0
    'gamma': 0.1,

    # checkpoint
    # FIXME 保存路径
    'ckpt_path': '/model/checkpoint/',
    # 'save_checkpoint_steps': 2000,
    'keep_checkpoint_max': 5,
    'resume_net': None,

    # dataset 
    # FIXME train_dataset_path
    'training_dataset': '/dataset/WIDER_train/label.txt', # '/home/ms8p/q/widerface/train/label.txt'
    'pretrain': True,
    # FIXME resnet50 pretrain_path
    'pretrain_path': '/code/resnet50_ascend_v130_imagenet2012_official_cv_bs32_acc77.06.ckpt', #'/home/ms8p/q/RetinaFace_ResNet50_combine/resnet/scripts/train_parallel0/ckpt_0/resnet-90_625.ckpt',

    # val
    # FIXME 
    'val_model': '/code/RetinaFace-50_1609.ckpt',
    # FIXME val_dataset_path
    'val_dataset_folder': '/dataset/WIDER_val/WIDER_val/', # '/home/ms8p/q/widerface/val/',
    'val_origin_size': False,
    'val_confidence_threshold': 0.02,
    'val_nms_threshold': 0.4,
    'val_iou_threshold': 0.5,
    'val_save_result': True, # False,
    # FIXME
    'val_predict_save_folder': '/model/widerface_result',
    'val_gt_dir': '/code/val_gt/',
}
