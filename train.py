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
"""Train Retinaface_resnet50."""
from __future__ import print_function
import math
import mindspore
import os

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.train import Model
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor, SummaryCollector
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.serialization import load_checkpoint, load_param_into_net

from src.config import cfg_res50
from src.network import RetinaFace, RetinaFaceWithLossCell, TrainingWrapper, resnet50
from src.loss import MultiBoxLoss
from src.dataset import create_dataset
from src.lr_schedule import adjust_learning_rate, warmup_cosine_annealing_lr
from shutil import copyfile

def train(cfg):

    context.set_context(mode=context.GRAPH_MODE, device_target=cfg['device_target'])
    # context.set_context(enable_graph_kernel=True)
    # context.set_context(mode=context.PYNATIVE_MODE, device_target=cfg['device_target'])
    device_num = cfg['nnpu']
    rank = 0
    if cfg['device_target'] == "Ascend":
        if device_num > 1:
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num, parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            init()
            rank = get_rank()
        else:
            context.set_context(device_id=cfg['device_id'])
    elif cfg['device_target'] == "GPU":
        # Enable graph kernel
        mindspore.set_context(enable_graph_kernel=True, graph_kernel_flags="--enable_parallel_fusion")
        if cfg['ngpu'] > 1:
            init("nccl")
            context.set_auto_parallel_context(device_num=get_group_size(), parallel_mode=ParallelMode.DATA_PARALLEL,
                                              gradients_mean=True)
            rank = get_rank()
        #else:
            # raise ValueError('cfg_num_gpu <= 1')

    batch_size = cfg['batch_size']
    max_epoch = cfg['epoch']

    momentum = cfg['momentum']
    lr_type = cfg['lr_type']
    weight_decay = cfg['weight_decay']
    loss_scale = cfg['loss_scale']
    initial_lr = cfg['initial_lr']
    gamma = cfg['gamma']
    T_max = cfg['T_max']
    eta_min = cfg['eta_min']
    training_dataset = cfg['training_dataset']
    num_classes = 2
    negative_ratio = 7
    stepvalues = (cfg['decay1'], cfg['decay2'])

    ds_train = create_dataset(training_dataset, cfg, batch_size, multiprocessing=True, num_worker=cfg['num_workers'])
    print('dataset size is : \n', ds_train.get_dataset_size())

    steps_per_epoch = math.ceil(ds_train.get_dataset_size())

    multibox_loss = MultiBoxLoss(num_classes, cfg['num_anchor'], negative_ratio, cfg['batch_size'])
    backbone = resnet50(1001)
    backbone.set_train(True)

    if cfg['pretrain'] and cfg['resume_net'] is None:
        pretrained_res50 = cfg['pretrain_path']
        param_dict_res50 = load_checkpoint(pretrained_res50)
        load_param_into_net(backbone, param_dict_res50)
        print('Load resnet50 from [{}] done.'.format(pretrained_res50))
    
    # STEP network
    net = RetinaFace(phase='train', backbone=backbone)
    net.set_train(True)

    if cfg['resume_net'] is not None:
        pretrain_model_path = cfg['resume_net']
        param_dict_retinaface = load_checkpoint(pretrain_model_path)
        load_param_into_net(net, param_dict_retinaface)
        print('Resume Model from [{}] Done.'.format(cfg['resume_net']))

    net = RetinaFaceWithLossCell(net, multibox_loss, cfg)
    
    # STEP learning rate
    if lr_type == 'dynamic_lr':
        lr = adjust_learning_rate(initial_lr, gamma, stepvalues, steps_per_epoch, max_epoch,
                                  warmup_epoch=cfg['warmup_epoch'])
    elif lr_type == 'cosine_annealing':
        lr = warmup_cosine_annealing_lr(initial_lr, steps_per_epoch, cfg['warmup_epoch'], max_epoch, T_max, eta_min)
    
    # STEP optim
    if cfg['optim'] == 'momentum':
        opt = mindspore.nn.Momentum(net.trainable_params(), lr, momentum, weight_decay, loss_scale)
    elif cfg['optim'] == 'sgd':
        opt = mindspore.nn.SGD(params=net.trainable_params(), learning_rate=lr, momentum=momentum,
                               weight_decay=weight_decay, loss_scale=loss_scale)
    else:
        raise ValueError('optim is not define.')

    net = TrainingWrapper(net, opt)
    
    # 实例化模型
    model = Model(net)
    # model = Model(net, optimizer=opt, amp_level="O2")

    config_ck = CheckpointConfig(save_checkpoint_steps=ds_train.get_dataset_size() * 2,
                                 keep_checkpoint_max=cfg['keep_checkpoint_max'])
    # config_ck = CheckpointConfig(save_checkpoint_steps=cfg['save_checkpoint_steps'],
                                 # keep_checkpoint_max=cfg['keep_checkpoint_max'])
    cfg['ckpt_path'] = cfg['ckpt_path'] + "ckpt_" + str(rank) + "/"
    ckpoint_cb = ModelCheckpoint(prefix="RetinaFace", directory=cfg['ckpt_path'], config=config_ck)
    
    # Init a SummaryCollector callback instance, and use it in model.train or model.eval
    summary_collector = SummaryCollector(summary_dir='/model/summary_dir', collect_freq=1)

    time_cb = TimeMonitor(data_size=ds_train.get_dataset_size())
    callback_list = [LossMonitor(), time_cb, ckpoint_cb, summary_collector]
  

    print("============== Starting Training ==============")
    model.train(max_epoch, ds_train, callbacks=callback_list)


if __name__ == '__main__':

    config = cfg_res50
    mindspore.common.seed.set_seed(config['seed'])
    print('train config:\n', config)

    train(cfg=config)
