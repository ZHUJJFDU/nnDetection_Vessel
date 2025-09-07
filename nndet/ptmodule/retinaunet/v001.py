from nndet.ptmodule.retinaunet.base import RetinaUNetModule

from nndet.core.boxes.matcher import ATSSMatcher
from nndet.arch.heads.classifier import BCECLassifier
from nndet.arch.heads.regressor import GIoURegressor
from nndet.arch.heads.comb import DetectionHeadHNMNative
from nndet.arch.heads.segmenter import DiCESegmenterFgBg
from nndet.arch.conv import ConvInstanceRelu, ConvGroupRelu

from nndet.ptmodule import MODULE_REGISTRY


@MODULE_REGISTRY.register
class RetinaUNetV001(RetinaUNetModule):
    base_conv_cls = ConvInstanceRelu
    head_conv_cls = ConvGroupRelu

    head_cls = DetectionHeadHNMNative
    head_classifier_cls = BCECLassifier
    head_regressor_cls = GIoURegressor
    matcher_cls = ATSSMatcher
    segmenter_cls = DiCESegmenterFgBg
