# 目录

<!-- TOC -->

- [目录](#目录)
- [RetianFace描述](#RetianFace描述)
- [模型架构](#模型架构)
- [预训练ResNet50](#预训练ResNet50)
- [数据集](#数据集)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
        - [分布式训练](#分布式训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [性能](#性能)
        - [评估性能](#评估性能)
            - [WIDER FACE上的RetianFace](#WIDER FACE上的RetianFace)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

# RetianFace描述

RetianFace是一种单级人脸检测器，这里基于ResNet50骨干网络对WIDER FACE数据集进行人脸检测，可以做到对各种尺度条件下的人脸像素级别的定位。

[论文](https://arxiv.org/abs/1905.00641v2) ：Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, Stefanos Zafeiriou. "RetinaFace: Single-stage Dense Face Localisation in the Wild". 2019.

# 模型架构

RetianFace的总体网络架构如下：[链接](https://arxiv.org/abs/1905.00641v2)

# 预训练ResNet50

这里的RetinaFace使用ResNet50作为骨干网络来提取图像特征进行检测，需要先在ImageNet2012数据集上训练ResNet50得到预训练模型。在resnet文件夹下进行。

```python
# 训练示例
bash run_standalone_train.sh resnet50 imagenet2012 [DATASET_PATH]

# 分布式训练示例
bash run_distribute_train.sh resnet50 imagenet2012 [RANK_TABLE_FILE] [DATASET_PATH]

# 评估示例
bash run_eval.sh resnet50 imagenet2012 [DATASET_PATH] [CHECKPOINT_PATH]
```

# 数据集

使用的数据集：[WIDERFACE](<http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/WiderFace_Results.html>)

获取数据集：

1. 点击[此处](<https://github.com/peteryuX/retinaface-tf2>)获取数据集和标注。
2. 点击[此处](<https://github.com/peteryuX/retinaface-tf2/tree/master/widerface_evaluate/ground_truth>)获取评估地面真值标签。

- 数据集大小：3.42G，32203张彩色图像
    - 训练集：1.36G，12800张图像
    - 验证集：345.95M，3226张图像
    - 测试集：1.72G，16177张图像

# 环境要求

- 硬件（Ascend）
    - 使用Ascend来搭建硬件环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorial/training/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/doc/api_python/en/master/index.html)

# 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```python
  # 运行训练示例，将./src/config.py中的'nnpu'设置成1，然后
  python train.py > train.log 2>&1 &

  # 运行分布式训练示例，将./src/config.py中的'nnpu'设置成8，然后
  bash ./scripts/run_distribute_ms_train.sh hccl_ms.json

  # 运行评估示例，设置./src/config.py中的'val_model'，然后
  python eval.py > ./eval.log 2>&1 &
  ```

  对于分布式训练，需要提前创建JSON格式的hccl配置文件。

  请遵循以下链接中的说明：

 <https://gitee.com/mindspore/mindspore/tree/master/model_zoo/utils/hccl_tools.>

# 脚本说明

## 脚本及样例代码

```bash
├── model_zoo
    ├── README.md                          // 所有模型相关说明
    ├── RetianFace
        ├── README_CN.md                   // RetianFace相关说明
        ├── resnet                         // 预训练ResNet50模型所需的文件
        ├── scripts
            ├──run_distribute_ms_train.sh  // 分布式到Ascend的shell脚本
        ├── src
            ├──augmemtation.py             // 数据增强
            ├──config.py                   // 参数配置
            ├──dataset.py                  // 创建数据集
            ├──loss.py                     // 损失函数
            ├──lr_schedule.py              // 学习率衰减策略
            ├──network.py                  // RetianFace架构
            ├──utils.py                    // 数据预处理
        ├── eval.py                        // 评估脚本
        ├── train.py                       // 训练脚本
```

## 脚本参数

在config.py中可以同时配置训练参数和评估参数。

- 配置RetianFace和WIDER FACE数据集。

  ```python
    'device_target': 'Ascend',                                # 处理器环境
    'device_id': 0,                                           # Ascend单卡训练时选择的卡号
    'variance': [0.1, 0.2],                                   # 方差
    'clip': False,                                            # 裁剪
    'loc_weight': 2.0,                                        # Bbox回归损失权重
    'class_weight': 1.0,                                      # 置信度/类回归损失权重
    'landm_weight': 1.0,                                      # 地标回归损失权重
    'batch_size': 8,                                          # 训练批次大小
    'num_workers': 16,                                         # 数据集加载数据的线程数量
    'num_anchor': 29126,                                      # 矩形框数量，取决于图片大小
    'nnpu': 8,                                                # 并行训练的NPU数量
    'image_size': 840,                                        # 训练图像大小
    'match_thresh': 0.35,                                     # 匹配框阈值
    'optim': 'sgd',                                           # 优化器类型
    'momentum': 0.9,                                          # 优化器动量
    'weight_decay': 1e-4,                                     # 优化器权重衰减
    'loss_scale': 1,                                          # 优化器损失等级
    'epoch': 50,                                              # 训练轮次数量
    'lr_type': 'dynamic_lr',                                  # 学习率衰减策略
    'initial_lr': 0.04,                                       # 初始学习率
    'decay1': 15,                                             # 首次权重衰减的轮次数
    'decay2': 30,                                             # 二次权重衰减的轮次数
    'warmup_epoch': -1,                                       # 热身大小，-1表示无热身
    'gamma': 0.1,                                             # 学习率衰减比
    'ckpt_path': './checkpoint/',                             # 模型保存路径
    'keep_checkpoint_max': 5,                                 # 预留检查点数量
    'resume_net': None,                                       # 重启网络，默认为None
    'training_dataset': '',                                   # 训练数据集标签路径，如data/widerface/train/label.txt
    'pretrain': True,                                         # 是否基于预训练骨干进行训练
    'pretrain_path': './resnet/resnet-90_625.ckpt',           # 预训练的骨干ResNet50检查点路径
    # 验证
    'val_model': './checkpoint/ckpt_0/RetinaFace-50_201.ckpt', # 验证模型路径
    'val_dataset_folder': './data/widerface/val/',            # 验证数据集路径
    'val_origin_size': False,                                 # 是否使用全尺寸验证
    'val_confidence_threshold': 0.02,                         # 验证置信度阈值
    'val_nms_threshold': 0.4,                                 # 验证NMS阈值
    'val_iou_threshold': 0.5,                                 # 验证IOU阈值
    'val_save_result': False,                                 # 是否保存结果
    'val_predict_save_folder': './widerface_result',          # 结果保存路径
    'val_gt_dir': './data/widerface/ground_truth/',           # 验证集ground_truth路径
  ```

更多配置细节请参考脚本`config.py`。

## 训练过程

### 训练

- Ascend处理器环境运行

  ```bash
  python train.py > train.log 2>&1 &
  ```

### 分布式训练

- Ascend处理器环境运行

  ```bash
  bash ./scripts/run_distribute_ms_train.sh hccl_ms.json
  ```

  上述shell脚本将在后台运行分布训练。

## 评估过程

### 评估

- 在Ascend环境运行时评估WIDER FACE数据集

  设置./src/config.py中的'val_model'，然后

  ```bash
  python eval.py > ./eval.log 2>&1 &
  ```

# 模型描述

## 性能

### 评估性能

#### WIDER FACE上的RetianFace

| 参数                 | Ascend                                                      |
| -------------------------- | ----------------------------------------------------------- |
| 模型版本              | RetianFace_ResNet50                                                |
| 资源                   | Ascend 910               |
| 上传日期              | 2021-07-31                                 |
| MindSpore版本          | 1.2.0                                                 |
| 数据集                    | WIDER FACE                                                |
| 训练参数        | epoch=50, batch_size=8, lr_init=0.04（八卡并行训练）             |
| 优化器                  | SGD                                                    |
| 损失函数              | MultiBoxLoss + Softmax交叉熵                                       |
| 输出                    | 边界框 + 置信度 + 地标                                                 |
| 准确率             | 八卡：Easy：94.10%, Medium：93.17%, Hard：88.84%               |
| 速度                      | 八卡：430毫秒/步                        |
| 总时长                 | 八卡：1.2小时/50轮                                             |

# ModelZoo主页  

 请浏览官网[主页](https://gitee.com/mindspore/mindspore/tree/master/model_zoo)。
