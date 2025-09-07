 # nnDetection模型的网络架构组织
nnDetection的网络架构采用了模块化设计，主要基于RetinaNet和U-Net的结合（RetinaUNet），它的网络架构组织如下：
**1. 整体架构设计**
nnDetection的网络架构遵循了模块化和可配置的设计理念，主要包含以下组件：
编码器(Encoder): 提取图像特征的下采样路径
解码器(Decoder): 融合多尺度特征的上采样路径
检测头(Heads): 包括分类和回归头，用于目标检测
分割器(Segmenter): 可选组件，用于语义分割任务
核心模型是BaseRetinaNet，它从AbstractModel继承而来，定义了网络的基本结构和功能。
**2. 模块化组织结构**
*2.1 编码器(Encoder)模块*
编码器负责提取图像特征，它是一个多阶段的特征提取器：
采用Encoder类实现，它由多个阶段(stages)组成，每个阶段代表一个分辨率层级
每个阶段包含多个卷积块(blocks)，可以是简单卷积块或残差块等
编码器支持2D和3D数据处理
可配置参数包括卷积类型、核大小、步长、通道数等
*2.2 解码器(Decoder)模块*
解码器采用特征金字塔网络(FPN)结构，主要有以下实现：
BaseUFPN: 基础解码器类，提供FPN的基本功能
UFPNModular: 模块化的FPN实现，支持更灵活的特征融合
PAUFPN: 扩展版FPN，支持自适应特征融合
解码器通过自底向上和自顶向下的路径连接编码器的不同层级特征，形成多尺度特征表示。
*2.3 检测头(Heads)模块*
检测头负责从特征图预测目标的类别和边界框：
分类器(Classifier): 包括CEClassifier(交叉熵)和BCEClassifier(二元交叉熵)等实现
回归器(Regressor): 包括L1Regressor和GIoURegressor等实现
组合检测头(Combined Head): 如DetectionHeadHNM(带困难负样本挖掘的检测头)、DetectionHeadHNMNative等
这些头通过从解码器获取的多尺度特征预测目标的位置和类别。
*2.4 分割器(Segmenter)模块*
分割模块是可选的，用于语义分割任务：
主要实现如DiCESegmenter、DiCESegmenterFgBg等
使用Dice损失函数进行分割任务的优化
**3. 网络配置的自动调整**
nnDetection最独特的特点是能够自动适应不同的医学检测问题：
规划器(Planner): 自动确定网络参数，包括层数、通道数、卷积核大小等
数据分析: 根据数据特性(如图像尺寸、对象大小)自动调整网络架构
锚点生成: 根据训练数据中目标的尺寸分布自动生成适合的锚点
**4. 模型变体**
nnDetection提供了不同的模型变体，每个变体针对特定需求进行了优化：
RetinaUNetV001: 基础版本，使用BCE分类器、GIoU回归器和ATSS匹配器
RetinaUNetV002: 改进版本，可能包含性能或架构上的优化
**5. 训练和推理流程**
网络通过以下方法实现训练和推理：
train_step: 实现一个训练步骤，包括前向传播和损失计算
inference_step: 实现推理步骤，生成最终预测结果
后处理: 通过非极大值抑制(NMS)和阈值筛选完成最终检测框的生成
**6. 注册器(Registry)机制**
nnDetection使用注册器机制来管理不同模块的实现：
MODULE_REGISTRY: 注册不同的模型实现
PLANNER_REGISTRY: 注册不同的规划策略
DATALOADER_REGISTRY: 注册不同的数据加载器
AUGMENTATION_REGISTRY: 注册不同的数据增强策略
这种注册机制使得模型可以在配置文件中灵活组合不同的组件，而不需要修改代码。
总结来说，nnDetection的网络架构采用了高度模块化和自适应的设计，能够自动根据不同医学数据集的特性调整网络结构参数，这使得它能够在不需要手动调参的情况下，适应各种医学目标检测任务。


1.**预处理部分**
scripts/preprocess.py

2.**训练部分**

scripts/train.py

module = MODULE_REGISTRY 从cfg中注册网络

cfg拼接 config.yaml->v001.yaml->base_more.yaml->数据集config.yaml



***新建网络*** 

1.在nndet/ptmodule/retinaunet下新建v002.py

2.nndet/ptmodule/retinaunet/__init__.py 中from nndet.ptmodule.retinaunet.v002 import RetinaUNetV002

3.在nndet/conf/train下新建v002.yaml

4.在nndet/conf/config.yaml中改train: v002



网络部分

nndet/arch/

其中网络的基本组件(blocks)、编码器(encoder)、解码器(decoder)和网络头部(heads)



修改卷积nndet\arch\conv.py



修改检测头部 nndet\arch\heads\comb.py

修改分类器nndet\arch\heads\classifier.py

修改回归器nndet\arch\heads\regressor.py



nndet/core/retina.py

定义BaseRetinaNet类

nndet/ptmodule/retinaunet/base.py

实例化BaseRetinaNet

nndet/ptmodule/retinaunet/v001.py

实例化BaseRetinaNet



3.**后处理部分**

**背景知识**
IOU（Intersection over Union，交并比）是一个重要的评价指标。它用于衡量预测的边界框（bounding box）与实际的边界框（ground truth）之间的重叠程度。IOU的计算公式是预测框和真实框的交集面积除以它们的并集面积，其取值范围在0到1之间。IOU值越高，表示预测框与真实框的重叠程度越高，模型的检测或分割性能越好。

非极大值抑制（Non-Maximum Suppression，NMS）是目标检测中常用的后处理技术，用于消除多余的边界框，保留最优的检测结果。在目标检测任务中，一个物体可能会被多个边界框检测到，导致同一个物体出现多个检测框。NMS的目的是从这些重叠的检测框中筛选出最优的一个或几个框，提高检测结果的准确性。从所有检测框中选出置信度（confidence score）最高的框作为当前最优框。计算该最优框与其他所有框的IOU值。将IOU值大于设定阈值的框剔除，因为它们与最优框重叠度高，可能是重复的检测结果。从未被剔除的框中再次选择置信度最高的框，重复上述步骤，直到所有框处理完毕。
1. **分类损失 (Classification Loss)**
位置：`nndet/losses/classification.py`
```python
class FocalLossWithLogits(nn.Module):
    def __init__(self,
                 gamma: float = 2,  # 调节难易样本的权重
                 alpha: float = -1,  # 调节正负样本的权重
                 reduction: str = "sum",
                 loss_weight: float = 1.,
                 ):
```
作用：
- 用于判断检测到的物体属于哪个类别
- gamma参数：调节难易样本的权重，增大gamma会更关注困难样本
- alpha参数：调节正负样本的权重，增大alpha会提高对前景类的关注度
修改建议：
- 如果检测效果对小目标不敏感，可以增大gamma值（默认2）
- 如果样本严重不平衡，可以调整alpha值

2. **回归损失 (Regression Loss)**
位置：`nndet/losses/regression.py`
```python
class SmoothL1Loss(torch.nn.Module):
    def __init__(self,
                 beta: float,  # L1到L2的转换点
                 reduction: Optional[str] = None,
                 loss_weight: float = 1.,
                 ):
```
作用：
- 用于预测边界框的位置和大小
- beta参数：控制L1和L2损失的过渡点
修改建议：
- 如果边界框预测不够精确，可以减小beta值
- 如果训练不稳定，可以增大beta值

3. **GIoU损失**
位置：`nndet/losses/regression.py`
```python
class GIoULoss(torch.nn.Module):
    def __init__(self,
                 reduction: Optional[str] = None,
                 eps: float = 1e-7,
                 loss_weight: float = 1.,
                 ):
```
作用：
- 提供比传统IoU更好的边界框回归
- 考虑了边界框的重叠度和距离
修改建议：
- 可以通过调整loss_weight来改变其在总损失中的权重
- eps参数用于数值稳定性，一般不需要修改

4. **Dice损失**
位置：`nndet/losses/segmentation.py`
```python
class SoftDiceLoss(nn.Module):
    def __init__(self,
                 nonlin: Callable = None,
                 batch_dice: bool = False, 
                 do_bg: bool = False,
                 smooth_nom: float = 1e-5,
                 smooth_denom: float = 1e-5,
                 ):
```
作用：
- 用于医学图像分割任务
- 特别适合处理类别不平衡问题
修改建议：
- do_bg参数：是否考虑背景类
- smooth_nom和smooth_denom：数值稳定性参数

5. **检测头中的损失组合**
位置：`nndet/arch/heads/comb.py`
```python
class DetectionHeadHNM(DetectionHead):
    def compute_loss(self,
                     prediction: Dict[str, Tensor],
                     target_labels: List[Tensor],
                     matched_gt_boxes: List[Tensor],
                     anchors: List[Tensor],
                     ):
```
这里组合了分类和回归损失：
```python
losses = {}
if sampled_pos_inds.numel() > 0:
    losses["reg"] = self.regressor.compute_loss(
        box_deltas[sampled_pos_inds],
        target_deltas_sampled,
        ) / max(1, sampled_pos_inds.numel())

losses["cls"] = self.classifier.compute_loss(
    box_logits[sampled_inds], target_labels[sampled_inds])
```

常见的调优建议：

1. **对于小目标检测问题**：
- 增大Focal Loss的gamma值（如2->4）
- 增加分类损失的权重

2. **对于边界框不够精确的问题**：
- 增加GIoU损失的权重
- 减小SmoothL1Loss的beta值

3. **对于类别严重不平衡的问题**：
- 调整Focal Loss的alpha值
- 考虑启用do_bg参数在Dice损失中

4. **对于训练不稳定的问题**：
- 增大smooth参数

- 调整loss_weight的比例

我来帮你找到相关的修改位置。让我们分步骤来查找：

1. **Anchor配置**
位置：`nndet/core/boxes/anchors.py`
```python
class AnchorGenerator2D(torch.nn.Module):
    def __init__(self, 
                 sizes: Sequence[Union[int, Sequence[int]]] = (128, 256, 512),
                 aspect_ratios: Sequence[Union[float, Sequence[float]]] = (0.5, 1.0, 2.0),
                 **kwargs):
```

2. **置信度阈值和NMS阈值**
位置：`nndet/core/retina.py`
```python
class BaseRetinaNet(AbstractModel):
    def __init__(self,
                 # ...
                 score_thresh: float = None,  # 置信度阈值
                 detections_per_img: int = 100,
                 topk_candidates: int = 10000,
                 remove_small_boxes: float = 1e-2,
                 nms_thresh: float = 0.9,  # NMS阈值
                 ):
```

3. **配置文件**
位置：`nndet/conf/train/v001.yaml` 或其他配置文件中：
```yaml
model_cfg:
  plan_anchors_overwrites:  # 这里可以覆盖anchor的配置
    width: [[2.0, 3.0, 4.0], [4.0, 6.0, 8.0], [8.0, 12.0, 16.0], [8.0, 12.0, 16.0]]
    height: [[3.0, 4.0, 5.0], [6.0, 8.0, 10.0], [12.0, 16.0, 20.0], [24.0, 32.0, 40.0]]
    depth: [[3.0, 4.0, 5.0], [6.0, 8.0, 10.0], [12.0, 16.0, 20.0], [24.0, 32.0, 40.0]]
```

4. **推理时的阈值设置**
位置：`nndet/inference/ensembler/detection.py`
```python
@classmethod
def get_default_parameters(cls):
    return {
        # single model
        "model_iou": 0.1,
        "model_score_thresh": 0.0,  # 置信度阈值
        "model_topk": 1000,
        "model_detections_per_image": 100,
        
        # ensemble multiple models
        "ensemble_iou": 0.5,
        "ensemble_score_thresh": 0.0,
        "remove_small_boxes": 1e-2,
    }
```

建议的修改方案：

1. **增大Anchor尺寸**：
在配置文件中添加或修改：
```yaml
model_cfg:
  plan_anchors_overwrites:
    width: [[4.0, 6.0, 8.0], [8.0, 12.0, 16.0], [16.0, 24.0, 32.0], [32.0, 48.0, 64.0]]
    height: [[4.0, 6.0, 8.0], [8.0, 12.0, 16.0], [16.0, 24.0, 32.0], [32.0, 48.0, 64.0]]
    depth: [[4.0, 6.0, 8.0], [8.0, 12.0, 16.0], [16.0, 24.0, 32.0], [32.0, 48.0, 64.0]]
```

2. **降低置信度阈值**：
修改BaseRetinaNet的初始化参数或配置文件：
```python
score_thresh = 0.05  # 降低置信度阈值
nms_thresh = 0.7    # 适当降低NMS阈值以保留更多大框
```

3. **调整推理参数**：
```python
parameters = {
    "model_score_thresh": 0.05,  # 降低置信度阈值
    "model_iou": 0.1,           # 降低IoU阈值以保留更多框
    "remove_small_boxes": 1e-2,  # 可以适当增大以过滤掉小框
}
```

4. **增加大框的权重**：
在检测头的损失函数中增加对大框的权重，位置在`nndet/arch/heads/comb.py`中的`DetectionHeadHNM`类。

这些修改会让模型：
1. 倾向于生成更大的检测框
2. 保留更多可能的大框检测结果
3. 降低对小目标的敏感度
4. 提高大目标的检测召回率

需要注意的是，这些修改可能会导致：
1. 假阳性检测增加
2. 计算开销增加
3. 对小目标的检测性能下降

建议你先在验证集上测试这些修改的效果，然后根据实际效果进行微调。

