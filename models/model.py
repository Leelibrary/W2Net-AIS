import numpy as np
from models.fpn import LastLevelP6P7
from models.FPAN import FPAN, LastLevelP6P7
from models.heads import CLSHead, REGHead, FPNUNet5SegHead #, MultiHead
from models.anchors import Anchors
from models.losses import IntegratedLoss #, KLLoss
import torch.nn.functional as F
from models.FcaNet import fcanet34, fcanet50, fcanet101, fcanet152
from modules.GBC import *
from utils.nms_wrapper import nms
from utils.SPPF import SPPF
# from utils.cuda_rnms.r_nms import r_nms
from utils.box_coder import BoxCoder
from utils.bbox import clip_boxes
from modules.cpam import LightWeightWaveletCPAM, WaveletCPAM, local_att, ConvLocal



class RetinaNet(nn.Module):
    def __init__(self,backbone='fca101', hyps=None):
        super(RetinaNet, self).__init__()
        self.num_classes  = int(hyps['num_classes']) + 1
        self.anchor_generator = Anchors(
            ratios = np.array([0.2, 0.5, 1.0, 2.0, 5.0]),
        ) 
        self.num_anchors = self.anchor_generator.num_anchors
        self.init_backbone(backbone)
        self.sppf = SPPF(c_in=self.fpn_in_channels[-1], c_out=self.fpn_in_channels[-1], k=5)

        self.fpan = FPAN(
            in_channels_list=self.fpn_in_channels,  # 这里默认是 [C3,C4,C5] 的通道数
            out_channels=256,
            top_blocks=LastLevelP6P7(self.fpn_in_channels[-1], 256)
        )

        self.cls_head = CLSHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,      
            num_anchors=self.num_anchors,
            num_classes=self.num_classes
        )
        self.reg_head = REGHead(
            in_channels=256,
            feat_channels=256,
            num_stacked=4,
            num_anchors=self.num_anchors,
            num_regress=5   # xywha
        )

        # ---- 新增：分割头 + 分割损失（logits 上用 BCEWithLogits）----
        self.seg_head = FPNUNet5SegHead(fpn_channels=256, out_channels=1, bilinear=True)

        # --- 分割损失：BCEWithLogits(正样本权重) + Dice ---
        self.seg_lambda = float(hyps.get('seg_lambda', 1.0))

        # ======== seg 专属 adapter（用于 seg_head 前的特征重塑/解耦）========
        use_wavelet = bool(hyps.get('seg_use_wavelet', True))  # 可消融
        reduction = int(hyps.get('seg_att_reduction', 16))

        # 你 P1~P5 都是 256 通道（从你的代码看是这样）
        self.seg_adapters = nn.ModuleList([
            LightWeightWaveletCPAM(256, reduction=reduction, use_wavelet=use_wavelet),
            LightWeightWaveletCPAM(256, reduction=reduction, use_wavelet=use_wavelet),
            LightWeightWaveletCPAM(256, reduction=reduction, use_wavelet=use_wavelet),
            LightWeightWaveletCPAM(256, reduction=reduction, use_wavelet=use_wavelet),
            LightWeightWaveletCPAM(256, reduction=reduction, use_wavelet=use_wavelet),
        ])

        self.w_p2 = nn.Parameter(torch.ones(2))
        self.w_p1 = nn.Parameter(torch.ones(2))
        self.eps = 1e-4
        # 1212

        # 正样本权重（缓解前景稀疏），可在 hyp.py 里调
        self.register_buffer(
            'seg_pos_weight',
            torch.tensor(float(hyps.get('seg_pos_weight', 8.0)), dtype=torch.float32)
        )
        self.seg_bce = nn.BCEWithLogitsLoss(pos_weight=self.seg_pos_weight)

        # 两个分量的权重（可在 hyp.py 调整，比如 0.5/0.5）
        self.seg_bce_w = float(hyps.get('seg_bce_w', 0.5))
        self.seg_dice_w = float(hyps.get('seg_dice_w', 0.5))

        self.loss = IntegratedLoss(func='smooth')
        # self.loss_var = KLLoss()
        self.box_coder = BoxCoder()

        self.p2_reduce = nn.Conv2d(256, 256, 1, bias=False)
        self.p2_smooth = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # === 新增：P1 降维/平滑（由 C1 -> 256，并与上采样后的 P2 融合） ===
        self.p1_reduce = nn.Conv2d(self.c1_channels, 256, 1, bias=False)
        self.p1_smooth = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.stop_grad_seg_backbone = bool(hyps.get('stop_grad_seg_backbone', True))

        # ===== Seg-as-Attention (Residual) ===== 1215
        self.seg_att_alpha = float(hyps.get('seg_att_alpha', 1.0))

        # (1) 分层强弱门控：P3 强 -> P7 弱（可学习，但用 sigmoid 约束到 (0,1)）
        # 这里保存的是 “logits”，forward 时再 sigmoid 成 (0,1) 的 k_i
        init_ks = torch.tensor([0.7, 0.6, 0.5, 0.4, 0.3], dtype=torch.float32)
        eps = 1e-6
        init_ks = init_ks.clamp(eps, 1 - eps)
        init_logits = torch.log(init_ks / (1 - init_ks))
        self.seg_gate_ks = nn.Parameter(init_logits)  # shape: (5,)

        # (4) 注意力只放在 P3/P4：P3/P4 用 Conv+local_att；P5~P7 仅 Conv
        reduction_gate = int(hyps.get('seg_gate_reduction', 16))  # 可选超参（不给也行）
        self.seg_att_convs = nn.ModuleList([
            ConvLocal(256, reduction=reduction_gate),  # P3
            ConvLocal(256, reduction=reduction_gate),  # P4
            nn.Sequential(  # P5 only Conv
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(  # P6 only Conv
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
            nn.Sequential(  # P7 only Conv
                nn.Conv2d(256, 256, 3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            ),
        ])

    def init_backbone(self, backbone):
        if backbone == 'fca34':
            self.backbone = fcanet34(pretrained=True)
            self.fpn_in_channels = [128, 256, 512]
            del self.backbone.avgpool
            del self.backbone.fc
        elif backbone == 'fca50':
            self.backbone = fcanet50(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
            del self.backbone.avgpool
            del self.backbone.fc
        elif backbone == 'fca101':
            self.backbone = fcanet101(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
            del self.backbone.avgpool
            del self.backbone.fc
        elif backbone == 'fca152':
            self.backbone = fcanet152(pretrained=True)
            self.fpn_in_channels = [512, 1024, 2048]
            del self.backbone.avgpool
            del self.backbone.fc
        else:
            raise NotImplementedError

        # 记录 C1 的通道数（conv1 输出通道；如不存在该属性则回退 64）
        self.c1_channels = getattr(self.backbone.conv1, 'out_channels', 64)

    def ims_2_features(self, ims):
        c1 = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(ims)))
        c2 = self.backbone.layer1(self.backbone.maxpool(c1))
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        c5 = self.sppf(c5)
        # 返回 c1 供 P1 使用
        return c1, c2, c3, c4, c5

    def forward(self, ims, gt_boxes=None, gt_seg=None,test_conf=None, return_seg=False, process=None):  #
        anchors_list, offsets_list, cls_list, var_list = [], [], [], []
        original_anchors = self.anchor_generator(ims)   # (bs, num_all_achors, 5)
        anchors_list.append(original_anchors)

        c1, c2, c3, c4, c5 = self.ims_2_features(ims)

        # 只把 C3~C5 给 FPAN（检测分支不变）
        p_feats = self.fpan([c3, c4, c5])  # -> [P3,P4,P5,P6,P7]
        P3, P4, P5, P6, P7 = p_feats

        # FPAN
        # features = self.fpan(self.ims_2_features(ims))

        # cls_score = torch.cat([self.cls_head(feature) for feature in p_feats], dim=1)
        # bbox_pred = torch.cat([self.reg_head(feature) for feature in p_feats], dim=1)
        # bboxes = self.box_coder.decode(anchors_list[-1], bbox_pred, mode='xywht').detach()

        # 分割 logits（使用 P3~P5，分辨率更高）
        B, _, H, W = ims.shape

        w2 = F.relu(self.w_p2)
        w2 = w2 / (w2.sum() + self.eps)
        P2 = w2[0] * self.p2_reduce(c2) + w2[1] * F.interpolate(P3, size=c2.shape[-2:], mode='bilinear', align_corners=False)
        P2 = self.p2_smooth(P2)

        w1 = F.relu(self.w_p1)
        w1 = w1 / (w1.sum() + self.eps)
        P1 = w1[0] * self.p1_reduce(c1) + w1[1] * F.interpolate(P2, size=c1.shape[-2:], mode='bilinear', align_corners=False)
        P1 = self.p1_smooth(P1)

        # seg_inputs = [P1, P2, P3, P4, P5, P6]
        seg_inputs = [P1, P2, P3, P4, P5]

        # 再做分割分支专属 adapter（注意：不覆盖检测分支用的 P3~P7）
        seg_inputs = [self.seg_adapters[i](seg_inputs[i], seg_inputs[i]) for i in range(5)] # 1-11 这是FSAM模块

        seg_logits = self.seg_head(seg_inputs[0], seg_inputs[1], seg_inputs[2],
                                   seg_inputs[3], seg_inputs[4],
                                   out_hw=(H, W))

        A_full = torch.sigmoid(seg_logits)  # (B,1,H,W)

        # ---------- 再用 A_full 去 gate 检测特征 ----------
        att_p_feats = []
        for i, P in enumerate(p_feats):  # P3~P7  (i=0..4)
            # (3) detach：不让检测梯度反推分割分支（更稳）
            Ai = F.interpolate(A_full.detach(), size=P.shape[-2:], mode='bilinear', align_corners=False)

            # (1) 分层强弱门控：k_i ∈ (0,1)；P3 强 -> P7 弱
            k_i = torch.sigmoid(self.seg_gate_ks[i]).view(1, 1, 1, 1)

            # (2) 软门控：不掐死原特征，只在 Ai 高的区域增强
            Pg = P * (1.0 + k_i * Ai)

            # (4) P3/P4: Conv+local_att；P5~P7: 仅 Conv
            att = self.seg_att_convs[i](Pg)

            # residual
            P = P + self.seg_att_alpha * att
            att_p_feats.append(P)

        p_feats = att_p_feats # 1.11 消融实验

        # ---------- 最后才是检测 head ----------
        cls_score = torch.cat([self.cls_head(feature) for feature in p_feats], dim=1)
        bbox_pred = torch.cat([self.reg_head(feature) for feature in p_feats], dim=1)

        if self.training:
            # 兼容 self.loss 返回多种形式：tuple/list/dict
            loss_out = self.loss(cls_score, bbox_pred, anchors_list[-1], gt_boxes, iou_thres=0.5)

            if isinstance(loss_out, dict):
                loss_cls = loss_out.get('loss_cls', None)
                loss_reg = loss_out.get('loss_reg', None)
                if loss_cls is None or loss_reg is None:
                    # 若 dict 用别的键名，取前两个数值兜底
                    vals = [v for v in loss_out.values() if torch.is_tensor(v)]
                    loss_cls, loss_reg = vals[0], vals[1]
            else:
                # tuple / list / 单个张量
                if isinstance(loss_out, (list, tuple)):
                    loss_cls, loss_reg = loss_out[0], loss_out[1]
                else:
                    # 极端情况：只返回一个，总之别崩
                    loss_cls = loss_out
                    loss_reg = torch.zeros_like(loss_cls)

            if gt_seg is not None:
                if gt_seg.dim() == 3:
                    gt_seg = gt_seg.unsqueeze(1)  # (B,1,H,W)
                gt_seg = gt_seg.to(seg_logits.device).float()

                loss_bce = self.seg_bce(seg_logits, gt_seg)  # 带 pos_weight 的 BCE
                loss_dice = RetinaNet.dice_loss_from_logits(seg_logits, gt_seg)  # 1 - Dice
                loss_seg = (self.seg_bce_w * loss_bce + self.seg_dice_w * loss_dice) * self.seg_lambda
            else:
                loss_seg = torch.zeros_like(loss_cls)

            losses = dict(
                loss_cls=loss_cls,
                loss_reg=loss_reg,
                loss_seg=loss_seg,
                loss_total=loss_cls + loss_reg + loss_seg,
                seg_logits=seg_logits.detach()
            )
            return losses

        # ------------------- 测试/评估分支 -------------------
        det = self.decoder(ims, anchors_list[-1], cls_score, bbox_pred, test_conf=test_conf)

        if return_seg:
            return {
                'det': det,  # [scores, labels, boxes]
                'seg_logits': seg_logits  # (B,1,H,W), 未过 sigmoid
            }
        else:
            return det

    def decoder(self, ims, anchors, cls_score, bbox_pred, thresh=0.4, nms_thresh=0.3, test_conf=None):  # 0.2 0.3
        if test_conf is not None:
            thresh = test_conf
        bboxes = self.box_coder.decode(anchors, bbox_pred, mode='xywht')
        bboxes = clip_boxes(bboxes, ims)
        scores = torch.max(cls_score, dim=2, keepdim=True)[0]
        keep = (scores >= thresh)[0, :, 0]
        if keep.sum() == 0:
            return [torch.zeros(1), torch.zeros(1), torch.zeros(1, 5)]
        scores = scores[:, keep, :]
        anchors = anchors[:, keep, :]
        cls_score = cls_score[:, keep, :]
        bboxes = bboxes[:, keep, :]
        # NMS
        anchors_nms_idx = nms(torch.cat([bboxes, scores], dim=2)[0, :, :], nms_thresh)
        nms_scores, nms_class = cls_score[0, anchors_nms_idx, :].max(dim=1)
        output_boxes = torch.cat([
            bboxes[0, anchors_nms_idx, :],
            anchors[0, anchors_nms_idx, :]],
            dim=1
        )
        return [nms_scores, nms_class, output_boxes]

    @staticmethod
    def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
        """
        logits: (B,1,H,W) 未过sigmoid
        targets: (B,1,H,W) {0,1}
        返回 batch 平均 1-Dice
        """
        probs = torch.sigmoid(logits)
        dims = (1, 2, 3)
        inter = (probs * targets).sum(dims)
        den = probs.sum(dims) + targets.sum(dims)
        dice = (2 * inter + eps) / (den + eps)
        return 1.0 - dice.mean()

    def freeze_bn(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()



if __name__ == '__main__':
    pass
