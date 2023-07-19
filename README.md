# 嵌入式芯片与系统设计大赛海面溢油检测分割模型

## 使用方法

### 
### 1. 数据集准备

Please go to ["./datasets/README.md"](datasets/README.md) for details, or please send an Email to zhangyitian2017 AT 163.com to request the preprocessed data. If you would like to use the preprocessed data, please use it for research purposes and do not redistribute it.

### 2. 环境配置

Please prepare an environment with python=3.7, and then use the command "pip install -r requirements.txt" for the dependencies.

### 3. 训练/测试

- Run the train script on synapse dataset. The batch size can be reduced to 12 or 6 to save memory (please also decrease the base_lr linearly), and both can reach similar performance.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Synapse --vit_name R50-ViT-B_16
```

- Run the test script on synapse dataset. It supports testing for both 2D images and 3D volumes.

```bash
python test.py --dataset Synapse --vit_name R50-ViT-B_16
```

## Reference
* [Google ViT](https://github.com/google-research/vision_transformer)
* [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch)
* [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch)

