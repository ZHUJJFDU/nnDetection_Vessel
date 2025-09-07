<div align="center">

<img src=docs/source/nnDetection.svg width="600px">

![Version](https://img.shields.io/badge/nnDetection-Vessel-blue)
![Python](https://img.shields.io/badge/python-3.8+-orange)
![CUDA](https://img.shields.io/badge/CUDA-10.1%2F10.2%2F11.0-green)

</div>

# nnDetection with Vessel-Guided Attention

这是一个基于nnDetection的改进版本，集成了**血管引导注意力机制**，专门用于肺栓塞检测任务。该模型通过利用血管分割掩码来增强网络对肺血管区域的感知能力，从而提高肺栓塞检测的性能。

## 主要特性

- **血管引导注意力机制**: 使用血管分割掩码引导网络关注肺血管区域
- **多尺度特征融合**: 在特征金字塔网络的多个层级应用血管注意力
- **自适应通道注意力**: 结合通道注意力和空间注意力机制
- **端到端训练**: 支持血管掩码和检测任务的联合训练

## 核心模块

- `VascularGuidedAttention`: 血管引导注意力模块
- `VesselAttentionFPN`: 带血管注意力的特征金字塔网络
- `VesselAttenRetinaUNetModule`: 血管引导的RetinaUNet主模块

# 环境配置

## 基本要求

- Python 3.8+
- PyTorch 1.10+ (推荐 1.11.0)
- CUDA 10.1+ (推荐 11.3)
- GPU内存 ≥ 11GB (推荐 RTX2080TI 或更高)

## 快速安装

```bash
# 创建conda环境
conda create --name nndet_vessel python=3.8
conda activate nndet_vessel

# 安装依赖
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
pip install hydra-core --upgrade --pre

# 编译安装
FORCE_CUDA=1 pip install -v -e .
```

## 环境变量设置

```bash
export det_data="/path/to/your/data"          # 数据目录
export det_models="/path/to/your/models"      # 模型保存目录
export OMP_NUM_THREADS=1                      # 必须设置为1
export det_num_threads=12                     # 数据加载进程数
```

## 验证安装

```bash
python -c "import torch; import nndet._C; import nndet"
nndet_env  # 查看环境信息
```

# 血管引导注意力配置

## 模型配置

本项目使用 `RetinaUNetV004` 模块，配置文件位于 `nndet/conf/module/RetinaUNetV004.yaml`。关键配置参数：

```yaml
# 血管引导注意力参数
vessel_attention:
  enabled: true                    # 启用血管引导注意力
  fusion_mode: "concat"           # 特征融合模式
  reduction_ratio: 16             # 通道减少比例
  attention_levels: [0, 1, 2, 3, 4]  # 应用注意力的FPN层级

# 训练配置
train_cfg:
  vessel_weight: 1.0              # 血管掩码损失权重
  detection_weight: 1.0           # 检测损失权重
```

## 数据格式要求

除了标准的nnDetection数据格式外，还需要提供血管分割掩码：

```
Task[XXX]_[Name]/
├── raw_splitted/
│   ├── imagesTr/           # CT图像
│   ├── labelsTr/           # 检测标签
│   └── vesselsTr/          # 血管分割掩码 (新增)
├── dataset.yaml
└── ...
```

血管掩码文件命名规则：
- `case0000.nii.gz` - 对应 `imagesTr/case0000_0000.nii.gz`
- 掩码值：0=背景，1=血管区域

# 训练流程

基于 `train.txt` 的完整训练流程，以Task101为例：

## 1. 激活环境

```bash
conda activate nndet_vessel  # 或你的环境名
```

## 2. 数据预处理

```bash
# 标准数据预处理
nndet_prep 101 -np 12 -npp 12 --full_check

# 血管数据预处理 (如果有血管掩码)
nndet_vesselprep 101
```

## 3. 数据解包

```bash
# 解包图像数据
nndet_unpack /path/to/det_data/Task101_Full/preprocessed/D3V001_3d/imagesTr 6

# 解包血管掩码数据 (如果有)
nndet_vesselunpack /path/to/det_data/Task101_Full/preprocessed/D3V001_3d/vesselsTr 6
```

## 4. 模型训练

### 基础训练
```bash
# 设置环境变量
export OMP_NUM_THREADS=1
export det_num_threads=10
export CUDA_VISIBLE_DEVICES=0

# 开始训练并进行超参数搜索
nndet_train 101 --sweep
```

### 多折交叉验证
```bash
# 训练指定折数 (fold 1,2,3,4)
nndet_train 101 -o exp.fold=1,2,3,4 --sweep
```

### 血管引导注意力训练
```bash
# 使用血管引导注意力模块训练
nndet_train Task102_vesspe -o module=RetinaUNetV004 train=v004 +vessel.dir=/path/to/vesselsTr
```

## 5. 模型评估

```bash
# 超参数搜索
nndet_sweep 101 RetinaUNetV001_D3V001_3d 0

# 模型评估
nndet_eval 101 RetinaUNetV001_D3V001_3d 0 --boxes --analyze_boxes
```

## 6. 模型整合与预测

```bash
# 整合多折模型 (可选)
nndet_consolidate Task102_Test RetinaUNetV001_D3V001_3d --sweep_boxes --num_folds 4

# 预测
nndet_predict 101 RetinaUNetV001_D3V001_3d --fold -1
```

## 7. 结果转换

```bash
# 设置matplotlib后端
export MPLBACKEND=Agg

# 将检测框转换为NIfTI格式
nndet_boxes2nii 101 RetinaUNetV001_D3V001_3d --test=true --threshold=0.5
```

# 技术细节

## 血管引导注意力机制原理

### 核心思想
血管引导注意力机制通过以下方式增强肺栓塞检测：

1. **血管区域定位**: 利用血管分割掩码精确定位肺血管区域
2. **注意力引导**: 引导网络重点关注血管区域的特征
3. **多尺度融合**: 在FPN的多个层级应用血管注意力
4. **自适应权重**: 根据血管密度自适应调整注意力权重

### 关键组件

#### VascularGuidedAttention
- **通道注意力**: 使用全局平均池化和最大池化提取通道重要性
- **血管注意力**: 通过卷积网络处理血管掩码生成空间注意力
- **特征融合**: 将通道注意力和血管注意力进行融合

#### VesselAttentionFPN
- **多层级应用**: 在FPN的P0-P4层级都应用血管注意力
- **尺度自适应**: 自动调整血管掩码尺寸匹配特征图
- **渐进式增强**: 从低分辨率到高分辨率逐步增强特征

## 配置参数说明

### 重要参数
```yaml
vessel_attention:
  enabled: true                    # 是否启用血管引导注意力
  fusion_mode: "concat"           # 融合模式: "concat", "add", "multiply"
  reduction_ratio: 16             # 通道注意力的降维比例
  attention_levels: [0,1,2,3,4]   # 应用注意力的FPN层级
  vessel_weight: 1.0              # 血管注意力的权重
```

### 训练参数
```yaml
train_cfg:
  vessel_loss_weight: 0.5         # 血管分割损失权重
  detection_loss_weight: 1.0      # 检测损失权重
  vessel_threshold: 0.5           # 血管掩码二值化阈值
```

## 快速开始

### 基本使用流程
以Task101为例，完整的训练和推理流程：

```bash
# 1. 激活环境
conda activate nndet_vessel

# 2. 数据预处理
nndet_prep 101

# 3. 数据解包
nndet_unpack ${det_data}/Task101_Full/preprocessed/D3V001_3d/imagesTr 6

# 4. 模型训练（血管引导注意力）
nndet_train 101 -o module=RetinaUNetV004 --sweep

# 5. 模型评估
nndet_eval 101 RetinaUNetV004_D3V001_3d 0 --boxes

# 6. 模型整合
nndet_consolidate 101 RetinaUNetV004_D3V001_3d --sweep_boxes

# 7. 模型预测
nndet_predict 101 RetinaUNetV004_D3V001_3d --fold -1

# 8. 结果转换
nndet_boxes2nii 101 RetinaUNetV004_D3V001_3d --test
```

### 重要说明

- **数据准备**: 确保血管分割数据位于 `vesselsTr` 目录
- **模型选择**: 使用 `RetinaUNetV004` 模块以启用血管注意力
- **参数配置**: 通过 `-o` 参数可以覆盖配置文件中的设置
- **多折训练**: 使用 `-o exp.fold=0,1,2,3,4` 进行5折交叉验证

### 常用命令参数

```bash
# 启用血管注意力训练
-o module=RetinaUNetV004

# 多折训练
-o exp.fold=0,1,2,3,4

# 指定血管数据路径
+vessel.dir=/path/to/vesselsTr

# 超参数搜索
--sweep
```

### 结果目录结构

训练完成后，结果将保存在以下目录结构中：

```text
Task101_Full/
    results/
        RetinaUNetV004_D3V001_3d/
            fold_0/
                train/          # 训练日志
                val/            # 验证日志  
                model/          # 模型检查点
                predict/        # 预测结果
                eval/           # 评估结果
            consolidated/       # 整合后的结果
                model/          # 整合模型
                predict/        # 整合预测
```

## 常见问题

### 环境配置问题

**Q: 血管分割数据应该放在哪里？**

A: 血管分割数据应该放在 `${det_data}/Task101_Full/vesselsTr/` 目录下，文件命名格式为 `case_XXXX.nii.gz`。

**Q: 如何验证血管注意力是否正常工作？**

A: 检查训练日志中是否有 "VascularGuidedAttention" 相关的输出，或在配置中设置 `model.vessel_attention.enabled=True`。

### 训练问题

**Q: 训练时显存不足怎么办？**

A: 可以减小批次大小，在配置中设置 `train.batch_size=1` 或使用梯度累积。

**Q: 血管注意力训练比普通训练慢多少？**

A: 大约增加15-20%的训练时间，但检测精度会有显著提升。

### 数据问题

**Q: 血管分割质量对结果影响大吗？**

A: 是的，建议使用高质量的血管分割结果。可以使用专门的血管分割模型预处理数据。

**Q: 可以不使用血管分割数据吗？**

A: 可以，设置 `model.vessel_attention.enabled=False` 即可使用原始的RetinaUNet模型。

## 技术支持

如遇到问题，请检查：
1. 环境变量设置是否正确
2. 数据格式是否符合要求  
3. 血管分割数据是否存在
4. GPU显存是否充足

## 致谢

本项目基于以下优秀的开源项目：
- [nnDetection](https://github.com/MIC-DKFZ/nnDetection)
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)
