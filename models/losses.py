import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.bbox import bbox_overlaps, min_area_square
from utils.box_coder import BoxCoder
from utils.overlaps.rbox_overlaps import rbox_overlaps


def xyxy2xywh_a(query_boxes):
    out_boxes = query_boxes.copy()
    out_boxes[:, 0] = (query_boxes[:, 0] + query_boxes[:, 2]) * 0.5
    out_boxes[:, 1] = (query_boxes[:, 1] + query_boxes[:, 3]) * 0.5
    out_boxes[:, 2] = query_boxes[:, 2] - query_boxes[:, 0]
    out_boxes[:, 3] = query_boxes[:, 3] - query_boxes[:, 1]
    return out_boxes

# cuda_overlaps
class IntegratedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, func = 'lmr5p', seg_loss_fn=None, seg_loss_weight=1.0, num_seg_classes=1):
        super(IntegratedLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.box_coder = BoxCoder()
        if func == 'smooth':
            self.criteron = smooth_l1_loss
        elif func == 'mse':
            self.criteron = F.mse_loss
        elif func == 'balanced':
            self.criteron = balanced_l1_loss
        elif func == 'lmr5p':
            self.criteron = lmr5p
        else:
            raise ValueError(func)

        # 新增：分割损失
        self.seg_loss_fn = nn.BCEWithLogitsLoss(reduction='none')   # seg_loss_fn  # e.g. nn.BCEWithLogitsLoss() 或 nn.CrossEntropyLoss(ignore_index=...)
        self.seg_loss_weight = float(seg_loss_weight)
        self.num_seg_classes = int(num_seg_classes)

    def forward(self, classifications, regressions, anchors, annotations,iou_thres=0.5, seg_logits=None, seg_masks=None):

        device = classifications.device

        cls_losses = []
        reg_losses = []
        batch_size = classifications.shape[0]
        for j in range(batch_size):
            classification = classifications[j, :, :]
            regression = regressions[j, :, :]
            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, -1] != -1]
            if bbox_annotation.shape[0] == 0:
                cls_losses.append(torch.tensor(0).float().cuda())
                reg_losses.append(torch.tensor(0).float().cuda())
                continue

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            indicator = bbox_overlaps(
                min_area_square(anchors[j, :, :]),
                min_area_square(bbox_annotation[:, :-1])
            )
            ious = rbox_overlaps(
                anchors[j, :, :].cpu().numpy(),
                bbox_annotation[:, :-1].cpu().numpy(),
                indicator.cpu().numpy(),
                thresh=1e-1
            )
            if not torch.is_tensor(ious):
                ious = torch.from_numpy(ious).cuda()

            iou_max, iou_argmax = torch.max(ious, dim=1)

            positive_indices = torch.ge(iou_max, iou_thres)

            max_gt, argmax_gt = ious.max(0)
            if (max_gt < iou_thres).any():
                positive_indices[argmax_gt[max_gt < iou_thres]]=1

            # cls loss
            cls_targets = (torch.ones(classification.shape) * -1).cuda()
            cls_targets[torch.lt(iou_max, iou_thres - 0.1), :] = 0
            num_positive_anchors = positive_indices.sum()
            assigned_annotations = bbox_annotation[iou_argmax, :]
            cls_targets[positive_indices, :] = 0
            cls_targets[positive_indices, assigned_annotations[positive_indices, -1].long()] = 1
            alpha_factor = torch.ones(cls_targets.shape).cuda() * self.alpha
            alpha_factor = torch.where(torch.eq(cls_targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(cls_targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            bin_cross_entropy = -(cls_targets * torch.log(classification+1e-6) + (1.0 - cls_targets) * torch.log(1.0 - classification+1e-6))
            cls_loss = focal_weight * bin_cross_entropy
            cls_loss = torch.where(torch.ne(cls_targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            cls_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))
            # reg loss
            if positive_indices.sum() > 0:
                all_rois = anchors[j, positive_indices, :]
                gt_boxes = assigned_annotations[positive_indices, :]
                reg_targets = self.box_coder.encode(all_rois, gt_boxes)
                # reg_loss = self.criteron(regression[positive_indices, :], reg_targets)

                if self.criteron == lmr5p:
                    reg_loss = lmr5p(
                        regression[positive_indices, :],  # inputs
                        reg_targets,  # targets
                        all_rois,  # anchors (正样本)
                        self.box_coder
                    )
                else:
                    reg_loss = self.criteron(regression[positive_indices, :], reg_targets)
                reg_losses.append(reg_loss)

                if not torch.isfinite(reg_loss) :
                    import ipdb; ipdb.set_trace()
            else:
                reg_losses.append(torch.tensor(0).float().cuda())
        loss_cls = torch.stack(cls_losses).mean(dim=0, keepdim=True)
        loss_reg = torch.stack(reg_losses).mean(dim=0, keepdim=True)

        # ===== 新增：分割损失（可选）=====
        if (seg_logits is not None) and (seg_masks is not None) and (self.seg_loss_fn is not None):
            # 统一到同一设备/类型
            seg_logits = seg_logits.to(device)  # (B,1,H,W) or (B,C,H,W)
            seg_masks = seg_masks.to(device)

            if self.num_seg_classes == 1:
                # 二值：mask→float，形状对齐
                # seg_masks: (B,H,W) -> (B,1,H,W)
                if seg_masks.ndim == 3:
                    seg_masks = seg_masks.unsqueeze(1)
                seg_masks = seg_masks.float()
                loss_seg = self.seg_loss_fn(seg_logits, seg_masks)
            else:
                # 多类：CrossEntropyLoss，seg_masks: Long, shape (B,H,W)
                if seg_masks.ndim == 4 and seg_masks.size(1) == 1:
                    seg_masks = seg_masks.squeeze(1)
                loss_seg = self.seg_loss_fn(seg_logits, seg_masks)

            loss_seg = loss_seg.view(1) * self.seg_loss_weight
        else:
            loss_seg = torch.zeros(1, device=device)

        return loss_cls, loss_reg, loss_seg
        # return loss_cls, loss_reg

    
def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True,
                   weight = None):
    """
    https://github.com/facebookresearch/maskrcnn-benchmark
    """
    diff = torch.abs(inputs - targets)
    if  weight is  None:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        )
    else:
        loss = torch.where(
            diff < beta,
            0.5 * diff ** 2 / beta,
            diff - 0.5 * beta
        ) * weight.max(1)[0].unsqueeze(1).repeat(1,5)
    if size_average:
        return loss.mean()
    return loss.sum()


def balanced_l1_loss(inputs,
                     targets,
                     beta=1. / 9,
                     alpha=0.5,
                     gamma=1.5,
                     size_average=True):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """
    assert beta > 0
    assert inputs.size() == targets.size() and targets.numel() > 0

    diff = torch.abs(inputs - targets)
    b = np.e**(gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    if size_average:
        return loss.mean()
    return loss.sum()

def lmr5p(inputs, targets, anchors, boxcoder, beta=1./9, size_average=True):
    """
    WakeNet / L5p_mr 风格：两条等价回归路径取 min
    inputs : (N,5) 网络输出的回归量（编码空间）
    targets: (N,5) BoxCoder.encode 输出的回归量（编码空间）
    anchors: (N,5) 正样本anchor（注意 anchors[:,4] 要是 degree）
    boxcoder: 需要提供 lmr5pangle(anchors, inputs[:,4]) -> 等价路径的 dt'
    """
    assert inputs.shape[-1] == 5 and targets.shape[-1] == 5

    x2, x1 = inputs[:, 0], targets[:, 0]
    y2, y1 = inputs[:, 1], targets[:, 1]
    w2, w1 = inputs[:, 2], targets[:, 2]
    h2, h1 = inputs[:, 3], targets[:, 3]
    t2, t1 = inputs[:, 4], targets[:, 4]

    # 等价路径的角度编码（dt'）
    t2_equiv = boxcoder.lmr5pangle(anchors, t2)

    # diff：两条路径
    diff1 = torch.abs(x1 - x2)
    diff2 = torch.abs(y1 - y2)
    diff3 = torch.abs(w1 - w2)
    diff4 = torch.abs(h1 - h2)
    diff5 = torch.abs(w1 - h2)  # w<->h
    diff6 = torch.abs(h1 - w2)
    diff7 = torch.abs(t1 - t2)
    diff8 = torch.abs(t1 - t2_equiv)

    def _smooth(d):
        return torch.where(d < beta, 0.5 * d * d / beta, d - 0.5 * beta)

    loss_a = _smooth(diff1) + _smooth(diff2) + _smooth(diff3) + _smooth(diff4) + _smooth(diff7)
    loss_b = _smooth(diff1) + _smooth(diff2) + _smooth(diff5) + _smooth(diff6) + _smooth(diff8)

    loss = torch.min(loss_a, loss_b)

    if size_average:
        return (loss / 5.0).mean()
    return loss.sum()
