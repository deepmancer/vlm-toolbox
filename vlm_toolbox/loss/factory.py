from config.enums import LossType, LossWrappers
from loss.coarsely_supervised import CoarselySupervisedLoss
from loss.contrastive_loss import ContrastiveLoss
from loss.elm_loss import EnlargedLargeMarginLoss
from loss.l1_loss import WeightedL1Loss
from loss.l2_loss import WeightedL2Loss
from loss.label_smoothing_loss import LabelSmoothingLoss
from loss.margin_metric_softmax import MarginMetricLoss


class LossFactory:
    loss_class_mapping = {
        LossType.CONTRASTIVE_LOSS: ContrastiveLoss,
        LossType.WEIGHTED_L2_LOSS: WeightedL2Loss,
        LossType.WEIGHTED_L1_LOSS: WeightedL1Loss,
        LossType.LABEL_SMOOTHING_LOSS: LabelSmoothingLoss,
        LossType.MARGIN_METRIC_LOSS: MarginMetricLoss,
        LossType.ENLARGED_LARGE_MARGIN_LOSS: EnlargedLargeMarginLoss,
    }
    loss_wrapper_mapping = {
        LossWrappers.COARSELY_SUPERVISED_LOSS: CoarselySupervisedLoss,
    }

    @staticmethod
    def create_loss(loss_class_name, wrapper_class=None, fine_to_coarse_mapping=None, **kwargs):
        loss_class = LossFactory.loss_class_mapping.get(loss_class_name)
        loss_fn = loss_class(**kwargs)
        
        wrapper_class = LossFactory.loss_wrapper_mapping.get(wrapper_class, None)
        if wrapper_class is not None:
            return wrapper_class(loss_fn=loss_fn, fine_to_coarse_mapping=fine_to_coarse_mapping, **kwargs)
        else:
            return loss_fn
