import torch
import numpy as np
import numpy.random as npr
import cv2
from torchvision.transforms import Compose
from utils.utils import Rescale, Normailize, Reshape

# TODO: keep_ratio

# class Collater(object):
#     """"""
#     def __init__(self, scales, keep_ratio=False, multiple=32):
#         if isinstance(scales, (int, float)):
#             self.scales = np.array([scales], dtype=np.int32)
#         else:
#             self.scales = np.array(scales, dtype=np.int32)
#         self.keep_ratio = keep_ratio
#         self.multiple = multiple
#
#     def __call__(self, batch):
#         random_scale_inds = npr.randint(0, high=len(self.scales))
#         target_size = self.scales[random_scale_inds]
#         target_size = int(np.floor(float(target_size) / self.multiple) * self.multiple)
#         rescale = Rescale(target_size=target_size, keep_ratio=self.keep_ratio)
#         transform = Compose([Normailize(), Reshape(unsqueeze=False)])
#
#         images = [sample['image'] for sample in batch]
#         bboxes = [sample['boxes'] for sample in batch]
#         batch_size = len(images)
#         max_width, max_height = -1, -1
#         for i in range(batch_size):
#             im, _ = rescale(images[i])
#             height, width = im.shape[0], im.shape[1]
#             max_width = width if width > max_width else max_width
#             max_height = height if height > max_height else max_height
#
#         padded_ims = torch.zeros(batch_size, 3, max_height, max_width)
#
#         num_params = bboxes[0].shape[-1]
#         max_num_boxes = max(bbox.shape[0] for bbox in bboxes)
#         padded_boxes = torch.ones(batch_size, max_num_boxes, num_params) * -1
#         for i in range(batch_size):
#             im, bbox = images[i], bboxes[i]
#             im, im_scale = rescale(im)
#             height, width = im.shape[0], im.shape[1]
#             padded_ims[i, :, :height, :width] = transform(im)
#             if num_params < 9:
#                 bbox[:, :4] = bbox[:, :4] * im_scale
#             else:
#                 bbox[:, :8] = bbox[:, :8] * np.hstack((im_scale, im_scale))
#             padded_boxes[i, :bbox.shape[0], :] = torch.from_numpy(bbox)
#         return {'image': padded_ims, 'boxes': padded_boxes}

class Collater(object):
    """"""
    def __init__(self, scales, keep_ratio=False, multiple=32):
        if isinstance(scales, (int, float)):
            self.scales = np.array([scales], dtype=np.int32)
        else:
            self.scales = np.array(scales, dtype=np.int32)
        self.keep_ratio = keep_ratio
        self.multiple = multiple

    def __call__(self, batch):
        random_scale_inds = npr.randint(0, high=len(self.scales))
        target_size = self.scales[random_scale_inds]
        target_size = int(np.floor(float(target_size) / self.multiple) * self.multiple)

        rescale = Rescale(target_size=target_size, keep_ratio=self.keep_ratio)
        transform = Compose([Normailize(), Reshape(unsqueeze=False)])

        images = [sample['image'] for sample in batch]
        bboxes = [sample['boxes'] for sample in batch]
        masks  = [sample.get('mask', None) for sample in batch]  # 可能没有mask，兜底

        batch_size = len(images)
        max_width, max_height = -1, -1

        # 先走一遍，得到各图像在缩放后的尺寸的最大值，用于padding框大小
        for i in range(batch_size):
            im_rescaled, _ = rescale(images[i])
            h, w = im_rescaled.shape[0], im_rescaled.shape[1]
            if w > max_width:  max_width  = w
            if h > max_height: max_height = h

        # 分配padding后的容器
        padded_ims   = torch.zeros(batch_size, 3, max_height, max_width)
        num_params   = bboxes[0].shape[-1]
        max_num_boxes = max(bbox.shape[0] for bbox in bboxes)
        padded_boxes = torch.ones(batch_size, max_num_boxes, num_params) * -1

        # 新增：mask 的容器（长整型更适合语义分割类别索引；二值也OK）
        has_mask = all(m is not None for m in masks)
        padded_masks = None
        if has_mask:
            # 用 0 作为pad值（背景）
            padded_masks = torch.zeros(batch_size, max_height, max_width, dtype=torch.long)

        # 第二遍：真正填充图像/框/掩膜
        for i in range(batch_size):
            im, bbox = images[i], bboxes[i]
            im_rescaled, im_scale = rescale(im)  # im_scale 通常是标量
            h, w = im_rescaled.shape[0], im_rescaled.shape[1]

            # 图像：归一化、CHW，并放入大画布
            padded_ims[i, :, :h, :w] = transform(im_rescaled)

            # 框：根据参数个数决定缩放前 4 或前 8 个数值
            if bbox.size > 0:
                if num_params < 9:
                    bbox[:, :4] = bbox[:, :4] * im_scale
                else:
                    # OBB 多边形的情况（:8 为4个点的xy）
                    # 若 Rescale 返回标量 im_scale，堆叠成 (scale_x, scale_y)
                    if np.isscalar(im_scale):
                        sx, sy = im_scale, im_scale
                    else:
                        # 如果你的 Rescale 返回 (sx, sy)，这里直接使用
                        sx, sy = im_scale[0], im_scale[1]
                    bbox[:, :8] = bbox[:, :8] * np.array([sx, sy, sx, sy, sx, sy, sx, sy], dtype=bbox.dtype)

                padded_boxes[i, :bbox.shape[0], :] = torch.from_numpy(bbox)

            # 掩膜：用相同的 scale 缩放，注意使用最近邻插值，避免类别值被破坏
            if has_mask:
                m = masks[i]
                if m is None:
                    # 保持全零（背景）
                    continue
                # 计算缩放后的尺寸（与 im_rescaled 一致）
                # 若 Rescale 可能返回非等比例缩放，则应使用 im_rescaled 的 w、h
                mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                # 放入大画布
                padded_masks[i, :h, :w] = torch.from_numpy(mask_resized).long()

        out = {'image': padded_ims, 'boxes': padded_boxes}
        if has_mask:
            out['mask'] = padded_masks  # 形状: (B, Hmax, Wmax)
        return out