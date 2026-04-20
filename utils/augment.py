import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
import sys
import os
import math
import imgaug.augmenters as iaa
import torch
from utils.bbox import quad_2_rbox, rbox_2_quad, mask_valid_boxes


def _norm180(a_deg):
    # 归一化到 (-180, 180]
    return ((a_deg + 180.0) % 360.0) - 180.0

def _hflip_img_mask(img, mask):
    img = img[:, ::-1, :]
    if mask is not None:
        mask = mask[:, ::-1]
    return img, mask

def _vflip_img_mask(img, mask):
    img = img[::-1, :, :]
    if mask is not None:
        mask = mask[::-1, :]
    return img, mask

def _warp_affine_img_mask(img, M, dsize, mask):
    img = cv2.warpAffine(img, M, dsize=dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))
    if mask is not None:
        # 最近邻 + 边界填 0
        mask = cv2.warpAffine(mask, M, dsize=dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, mask


class HSV(object):
    def __init__(self , saturation=0, brightness=0, p=0.):
        self.saturation = saturation 
        self.brightness = brightness
        self.p = p

    def __call__(self, img, labels, seg_mask=None, mode=None):
        if random.random() < self.p:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value
            a = random.uniform(-1, 1) * self.saturation + 1
            b = random.uniform(-1, 1) * self.brightness + 1
            S *= a
            V *= b
            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img, labels, seg_mask


class HSV_pos(object):
    def __init__(self , saturation=0, brightness=0, p=0.):
        self.saturation = saturation 
        self.brightness = brightness
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
            S = img_hsv[:, :, 1].astype(np.float32)  # saturation
            V = img_hsv[:, :, 2].astype(np.float32)  # value
            a = random.uniform(-1, 1) * self.saturation + 1
            b = random.uniform(0, 1) * self.brightness + 1
            S *= a
            V *= b
            img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
            img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
            cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img, labels    

class Blur(object):
    def __init__(self, sigma=0 ,p=0.):
        self.sigma = sigma 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            blur_aug = iaa.GaussianBlur(sigma=(0,self.sigma))
            img = blur_aug.augment_image(img)
        return img, labels


class Grayscale(object):
    def __init__(self, grayscale=0. ,p=0.):
        self.alpha = random.uniform(grayscale,1.0)
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            gray_aug = iaa.Grayscale(alpha=(self.alpha, 1.0))
            img = gray_aug.augment_image(img)
        return img, labels


class Gamma(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            gm = random.uniform(1-self.intensity,1+self.intensity)
            img = np.uint8(np.power(img/float(np.max(img)), gm)*np.max(img))
        return img, labels


class Noise(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            noise_aug = iaa.AdditiveGaussianNoise(scale=(0, self.intensity * 255))
            img = noise_aug.augment_image(img)
        return img, labels



class Sharpen(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            sharpen_aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(1 - self.intensity,1 + self.intensity))
            img = sharpen_aug.augment_image(img)
        return img, labels


class Contrast(object):
    def __init__(self, intensity=0 ,p=0.):
        self.intensity = intensity 
        self.p = p

    def __call__(self, img, labels, mode=None):
        if random.random() < self.p:
            contrast_aug = aug = iaa.contrast.LinearContrast((1 - self.intensity, 1 + self.intensity))
            img=contrast_aug.augment_image(img)
        return img, labels


####################################
# class HorizontalFlip(object):
#     def __init__(self, p=0.):
#         self.p = p
#
#     def __call__(self, img, labels, seg_mask, mode=None):
#         if random.random() < self.p:
#             img = np.fliplr(img)
#             if mode == 'cxywha':
#                 labels[:, 1] = img.shape[1] - labels[:, 1]
#                 labels[:, 5] = -labels[:, 5]
#             if mode == 'xyxyxyxy':
#                 labels[:, [0,2,4,6]] = img.shape[1] - labels[:, [0,2,4,6]]
#             if mode == 'xywha':
#                 labels[:, 0] = img.shape[1] - labels[:, 0]
#                 labels[:, -1] = -labels[:, -1]
#         return img, labels, seg_mask

class HorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, labels, seg_mask=None, mode=None):
        """
        img:  HxWx3
        labels: (N,5)[cx,cy,w,h,a] for 'xywha'
                (N,6)[c,x,y,w,h,a] for 'cxywha'
                (N,8)[x1,y1,x2,y2,x3,y3,x4,y4] for 'xyxyxyxy'
        mask:
            - 若是 HxW（二值或概率），直接 np.fliplr
            - 若是点集/折线 (K,2) 或 (...,2)，仅 x 维做 W-1-x
            - 若带方向 (vx,vy) 作为最后两列，只翻转 vx 的符号
        """
        if random.random() >= self.p:
            return img, labels, seg_mask

        H, W = img.shape[:2]
        img = np.fliplr(img).copy()

        # ---- 处理 mask ----
        if seg_mask is not None:
            if seg_mask.ndim == 2:  # 二值/概率 mask
                seg_mask = np.fliplr(seg_mask).copy()
            else:
                # 假定为点集/折线：[..., 2]
                seg_mask = seg_mask.copy()
                # x -> W-1-x
                seg_mask[..., 0] = (W - 1) - seg_mask[..., 0]
                # 如果最后两列是 (vx, vy) 方向向量（可选）：
                # 仅当你确实这样存时再打开下一行
                # mask[..., -2] = -mask[..., -2]  # 只翻转 vx

        # ---- 处理 labels ----
        if labels is not None and len(labels) > 0:
            labels = labels.copy()
            if mode == 'xywha':
                # [cx, cy, w, h, a]
                labels[:, 0] = (W - 1) - labels[:, 0]
                labels[:, 4] = _norm180(180.0 - labels[:, 4])
            elif mode == 'cxywha':
                # [c, x, y, w, h, a]
                labels[:, 1] = (W - 1) - labels[:, 1]
                labels[:, 5] = _norm180(180.0 - labels[:, 5])
            elif mode == 'xyxyxyxy':
                # 8 点中的 x 分量镜像
                labels[:, [0, 2, 4, 6]] = (W - 1) - labels[:, [0, 2, 4, 6]]
                # 如需保持顺时针/逆时针顺序一致，可在此处重排点序（按需要）
            else:
                # 未指定模式则不改 labels
                pass

        return img, labels, seg_mask


# class VerticalFlip(object):
#     def __init__(self ,p=0.):
#         self.p = p
#
#     def __call__(self, img, labels, seg_mask, mode=None):
#         if random.random() < self.p:
#             img = np.flipud(img)
#             if mode == 'cxywha':
#                 labels[:, 2] = img.shape[0] - labels[:, 2]
#                 labels[:, 5] = -labels[:, 5]
#             if mode == 'xyxyxyxy':
#                 labels[:, [1,3,5,7]] = img.shape[0] - labels[:, [1,3,5,7]]
#             if mode == 'xywha':
#                 labels[:, 1] = img.shape[0] - labels[:, 1]
#                 labels[:, -1] = -labels[:, -1]
#         return img, labels, seg_mask

class VerticalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, labels, seg_mask=None, mode=None):
        """
        img:  HxWx3
        labels:
          - 'xywha'   -> (N,5): [cx,cy,w,h,a_deg]
          - 'cxywha'  -> (N,6): [c,x,y,w,h,a_deg]
          - 'xyxyxyxy'-> (N,8): [x1,y1,x2,y2,x3,y3,x4,y4]
        mask:
          - HxW（二值/概率）：直接上下翻转
          - Kx2（点集/折线）：y -> H-1-y；若最后两列为方向向量(vx,vy)，仅翻转 vy 的符号（按需打开）
        """
        if random.random() >= self.p:
            return img, labels, seg_mask

        H, W = img.shape[:2]
        img = np.flipud(img).copy()

        # ---- 处理 mask ----
        if seg_mask is not None:
            if seg_mask.ndim == 2:                 # 二值/概率图
                seg_mask = np.flipud(seg_mask).copy()
            else:                               # 点集/折线 [...,2]
                seg_mask = seg_mask.copy()
                seg_mask[..., 1] = (H - 1) - seg_mask[..., 1]
                # 若末尾还带方向向量 (vx, vy)，可只翻转 vy（按你数据结构决定是否启用）
                # mask[..., -1] = -mask[..., -1]

        # ---- 处理 labels ----
        if labels is not None and len(labels) > 0:
            labels = labels.copy()
            if mode == 'xywha':
                labels[:, 1] = (H - 1) - labels[:, 1]     # cy
                labels[:, 4] = _norm180(-labels[:, 4])    # a' = -a
            elif mode == 'cxywha':
                labels[:, 2] = (H - 1) - labels[:, 2]     # y
                labels[:, 5] = _norm180(-labels[:, 5])    # a' = -a
            elif mode == 'xyxyxyxy':
                labels[:, [1,3,5,7]] = (H - 1) - labels[:, [1,3,5,7]]
                # 如需保持点序(顺/逆时针)一致，可在此处按面积正负或极角排序重排
            else:
                pass

        return img, labels, seg_mask


# class Affine(object):
#     def __init__(self, degree = 0., translate = 0., scale = 0., shear = 0., p=0.):
#         self.degree = degree
#         self.translate = translate
#         self.scale = scale
#         self.shear = shear
#         self.p = p
#
#     def __call__(self, img, labels, seg_mask,mode=None):
#         if random.random() < self.p:
#             if mode == 'xywha':
#                 labels = rbox_2_quad(labels, mode = 'xywha')
#                 img, labels = random_affine(img, labels,
#                             degree=self.degree,translate=self.translate,
#                             scale=self.scale,shear=self.shear )
#                 labels = quad_2_rbox(labels, mode = 'xywha')
#
#             else:
#                 img, labels = random_affine(img, labels,
#                                 degree=self.degree,translate=self.translate,
#                                 scale=self.scale,shear=self.shear )
#         return img, labels, seg_mask

def _warp_points(M, pts):
    """将 Nx2 点集按 2x3 仿射矩阵 M 变换"""
    pts = np.asarray(pts, dtype=np.float32)
    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.hstack([pts, ones])        # [x,y,1]
    out = homo @ M.T                     # [x',y']
    return out


class Affine(object):
    def __init__(self, degree=0., translate=0., scale=0., shear=0., p=0.5):
        self.degree = degree
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.p = p

    def __call__(self, img, labels, seg_mask=None, mode=None):
        """
        要求 random_affine 支持 return_M=True，并返回 (img2, labels2, M)。
        - 若 mode == 'xywha'：先 rbox->quad，仿射后再 quad->rbox（与原实现一致）
        - mask:
            * HxW：cv2.warpAffine，同一 M
            * Kx2 点集：用同一 M 变换
        """
        if random.random() >= self.p:
            return img, labels, seg_mask

        H, W = img.shape[:2]

        if mode == 'xywha':
            # rbox -> quad（与原实现一致）
            from utils.bbox import rbox_2_quad, quad_2_rbox
            quads = rbox_2_quad(labels, mode='xywha') if labels is not None and len(labels) > 0 else labels
            img2, quads2, M = random_affine(
                img, quads,
                degree=self.degree, translate=self.translate,
                scale=self.scale, shear=self.shear,
                return_M=True
            )
            # quad -> rbox
            labels2 = quad_2_rbox(quads2, mode='xywha') if quads2 is not None and len(quads2) > 0 else quads2
        else:
            img2, labels2, M = random_affine(
                img, labels,
                degree=self.degree, translate=self.translate,
                scale=self.scale, shear=self.shear,
                return_M=True
            )

        # ---- mask 用同一 M ----
        mask2 = seg_mask
        if seg_mask is not None:
            if seg_mask.ndim == 2:
                # 对二值/概率图：最近邻；边界补 0
                mask2 = cv2.warpAffine(seg_mask.astype(np.float32), M, dsize=(W, H),
                                       flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                # 若是概率图也可用 LINEAR；二值建议 NEAREST
                # mask2 = (mask2 > 0.5).astype(mask.dtype)  # 如需硬阈值
                if seg_mask.dtype == np.uint8:
                    mask2 = mask2.astype(np.uint8)
            else:
                # 点集/折线：按同一 M 变换
                shp = seg_mask.shape
                pts = seg_mask.reshape(-1, 2)
                pts2 = _warp_points(M, pts)
                mask2 = pts2.reshape(shp)

        return img2, labels2, mask2



class CopyPaste(object):
    def __init__(self, mean = 0 , sigma = 0, p=0.):
        self.mean = mean
        self.sigma = sigma
        self.p = np.clip(p, 0, 0.5)
        # 遵循3sigma原则，在船体侧边一个h位置为mean=0，偏移的范围约束在0+3*sigma内(2*sigma就够了)
        self.pos = abs(np.random.normal(self.mean, self.sigma))


    def __call__(self, img, labels, mode=None):
        boxes_w = labels[:,3]
        boxes_h = labels[:,4]
        boxes_a = labels[:,5]
        areas = boxes_w * boxes_h
        object_coors = [get_rotated_coors(i).reshape(-1,2).astype(np.int32)  for i in labels[:,1:]]
        pasted_img=img.copy()
        for i,coor in enumerate(object_coors):
            a=boxes_a[i]; w=boxes_w[i]; h=boxes_h[i]; area = areas[i]
            area_ratio = areas[i]/img.shape[0]/img.shape[1]
            # vis验证bbox计算无误
            # img = cv2.polylines(img,[coor],True,(0,0,255),2)	# 后三个参数为：是否封闭/color/thickness
            # cv2.imshow('drawbox',img)
            # cv2.moveWindow('drawbox',100,100)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            M_up   = np.float32([[1, 0, -h*(1+self.pos)*np.cos(math.pi*0.5+a)], [0, 1, -h*(1+self.pos)*np.sin(math.pi*0.5+a)]])
            M_down = np.float32([[1, 0,  h*(1+self.pos)*np.cos(math.pi*0.5+a)], [0, 1,  h*(1+self.pos)*np.sin(math.pi*0.5+a)]])
            # 分别获得bbox上下邻域的梯度
            sobel_up  , up_masked_img   , up_pos_mask   = cal_sobel(M_up, coor,img)
            sobel_down, down_masked_img , down_pos_mask = cal_sobel(M_down, coor,img)
            up_masked_img   = cv2.cvtColor(up_masked_img,   cv2.COLOR_BGR2GRAY)
            down_masked_img = cv2.cvtColor(down_masked_img, cv2.COLOR_BGR2GRAY)
            # 获取gt_mask
            gt_mask = np.zeros(img.shape[:-1], np.uint8)
            cv2.fillConvexPoly(gt_mask, coor, (1, 1))

            if  area_ratio<0.01:   self.p *= 1.2  # 小目标尤其丢的厉害，加倍加倍
            # 两侧都不越界，为了考虑海面反光导致的梯度骤增采用作差法。适合场景：陆海/海面/陆地
            if not sobel_up.all() and not sobel_down.all() and random.random() < self.p:	
                grad_diff = ((sobel_up>20).sum()-(sobel_down>20).sum())/area	# thre: 0.1
                pix_diff = abs((up_masked_img).sum()/(area*255) - (down_masked_img).sum()/(area*255)) # 防止模糊图像的梯度都平滑带来误操作
                if grad_diff < 0.15 and pix_diff < 0.15:	# 两侧环境一致，均为海面，两边等概率paste
                    if random.random()<0.7:
                        pasted_img = copy_paste(pasted_img,gt_mask,up_pos_mask)
                        labels = np.row_stack((labels,generate_label(M_up,labels[i])))
                    if random.random()<0.7:
                        pasted_img = copy_paste(pasted_img,gt_mask,down_pos_mask)
                        labels = np.row_stack((labels,generate_label(M_down,labels[i])))
                else: 		# 半海半陆地，选海面paste
                    if up_masked_img.sum()<down_masked_img.sum() : 
                        pos_mask = up_pos_mask
                        M = M_up
                    else:
                        pos_mask = down_pos_mask
                        M = M_down
                    pasted_img = copy_paste(pasted_img,gt_mask,pos_mask)
                    labels = np.row_stack((labels,generate_label(M,labels[i])))
            else:		# 越界增强有风险，没有差分对比，容易误判，暂时不做增强
                pass


        # vis:可视化检查正确性
        # fig = plt.figure(figsize=(10, 10))   
        # ax1 = fig.add_subplot(121)
        # ax1.imshow(img)
        # plt.title('img')
        # plt.axis('off')

        # ax4 = fig.add_subplot(122)
        # ax4.imshow(pasted_img)
        # plt.title('pasted_img')
        # plt.axis('off')
        # plt.show()

        return pasted_img, labels

class Augment(object):
    def __init__(self, augmentations, probs=1, box_mode=None):
        self.augmentations = augmentations
        self.probs = probs
        self.mode = box_mode
        
    def __call__(self, img, labels, seg_mask=None):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs
                
            if random.random() < prob:
                img, labels = augmentation(img, labels, seg_mask, self.mode)

        return img, labels, seg_mask







def random_affine(img,  targets=(), degree=10, translate=.1, scale=.1, shear=10):
    # torchvision.transforms.RandomAffine(degree=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    
    if targets is None:
        targets = []
    border = 0  # width of added border (optional)
    height = img.shape[0] + border * 2
    width = img.shape[1] + border * 2

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    # # # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
    T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)


    M =  T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
    imw = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_AREA,
                         borderValue=(128, 128, 128))  # BGR order borderValue

    # Return warped points also
    targets[:, [0,2,4,6]] = targets[:, [0,2,4,6]] * M[0,0] + targets[:, [1,3,5,7]] * M[0,1] + M[0,2]
    targets[:, [1,3,5,7]] = targets[:, [0,2,4,6]] * M[1,0] + targets[:, [1,3,5,7]] * M[1,1] + M[1,2]
    for x in range(0,8,2):
        targets[:,x] = targets[:,x].clip(0, width)
    for y in range(1,8,2):
        targets[:,y] = targets[:,y].clip(0, height)
    return imw, targets



def cutout(image, labels):
    # https://arxiv.org/abs/1708.04552
    # https://github.com/hysts/pytorch_cutout/blob/master/dataloader.py
    # https://towardsdatascience.com/when-conventional-wisdom-fails-revisiting-data-augmentation-for-self-driving-cars-4831998c5509
    h, w = image.shape[:2]

    def bbox_ioa(box1, box2, x1y1x2y2=True):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                     (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

        # box2 area
        box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + 1e-16

        # Intersection over box2 area
        return inter_area / box2_area

    # random mask_size up to 50% image size
    mask_h = random.randint(1, int(h * 0.5))
    mask_w = random.randint(1, int(w * 0.5))

    # box center
    cx = random.randint(0, h)
    cy = random.randint(0, w)

    xmin = max(0, cx - mask_w // 2)
    ymin = max(0, cy - mask_h // 2)
    xmax = min(w, xmin + mask_w)
    ymax = min(h, ymin + mask_h)

    # apply random color mask
    mask_color = [random.randint(0, 255) for _ in range(3)]
    image[ymin:ymax, xmin:xmax] = mask_color

    # return unobscured labels
    if len(labels):
        box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
        ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
        labels = labels[ioa < 0.90]  # remove >90% obscured labels
    return labels



def coor_trans(M,coor):
    tcoor = [0 for i in range(8)]
    coor_x = coor[:,0]
    coor_y = coor[:,1]
    tx = M[0,0]*coor_x + M[0,1]*coor_y + M[0,2]
    ty = M[1,0]*coor_x + M[1,1]*coor_y + M[1,2]
    tcoor[::2] = tx
    tcoor[1::2] = ty
    return np.array(tcoor).reshape(4,2).astype(np.int32)

def cal_sobel(M,coor,img):
    mask = np.zeros(img.shape[:-1], np.uint8)
    tcoor = coor_trans(M,coor)
    if (tcoor>0).all() and (tcoor[:,0]<img.shape[1]).all() and (tcoor[:,1]<img.shape[0]).all() :
        cv2.fillConvexPoly(mask, tcoor, (1, 1))
        masked_img = img * np.expand_dims(mask,-1)
        sobel = filter(masked_img)[...,0]	
        pos_mask = mask.copy()
        cv2.fillConvexPoly(pos_mask, tcoor, (1, 1))

        return sobel,masked_img,pos_mask
    else:
        mask.fill(255)
        return mask, img,mask

def filter(img):
	img_gry = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	x = cv2.Sobel(img_gry, cv2.CV_16S, 1, 0)
	y = cv2.Sobel(img_gry, cv2.CV_16S, 0, 1)
	xy = cv2.Sobel(img_gry,cv2.CV_16S, 1 , 1)
	absX = cv2.convertScaleAbs(x)
	absY = cv2.convertScaleAbs(y)
	xy = cv2.convertScaleAbs(xy)
	sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
	sobel = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)

	return sobel

def copy_paste(img,gt_mask,pos_mask):
	pasted = img.copy() 
	pasted[pos_mask!=0]=img[gt_mask!=0]  
	return pasted 

def generate_label(M,label):
    new_label = label.copy()
    cx = label[1]; cy = label[2]; 
    tx = M[0,0]*cx + M[0,1]*cy + M[0,2]
    ty = M[1,0]*cx + M[1,1]*cy + M[1,2]
    new_label[1] = tx
    new_label[2] = ty
    return new_label






# if __name__ == "__main__":
    

#     path = '/py/datasets/HRSC2016/yolo-dataset/train'
#     img_files = os.listdir(path)
#     img_files = [i for i in img_files if i.endswith('jpg')]
#     for img_file in img_files:
#         img = cv2.imread(os.path.join(path,img_file),1)
#         labels = np.loadtxt(os.path.join(path,img_file)[:-4]+'.txt')
#         if len(labels.shape)<2:
#             labels = np.array([labels])
#         labels[:,[1,3]] *= img.shape[1]
#         labels[:,[2,4]] *= img.shape[0]

#         cp = CopyPaste(sigma= 0.1)
#         cp(img,labels)



def get_rotated_coors(box):
    assert len(box) > 0 , 'Input valid box!'
    cx = box[0]; cy = box[1]; w = box[2]; h = box[3]; a = box[4]
    xmin = cx - w*0.5; xmax = cx + w*0.5; ymin = cy - h*0.5; ymax = cy + h*0.5
    t_x0=xmin; t_y0=ymin; t_x1=xmin; t_y1=ymax; t_x2=xmax; t_y2=ymax; t_x3=xmax; t_y3=ymin
    R = np.eye(3)
    R[:2] = cv2.getRotationMatrix2D(angle=-a*180/math.pi, center=(cx,cy), scale=1)
    x0 = t_x0*R[0,0] + t_y0*R[0,1] + R[0,2] 
    y0 = t_x0*R[1,0] + t_y0*R[1,1] + R[1,2] 
    x1 = t_x1*R[0,0] + t_y1*R[0,1] + R[0,2] 
    y1 = t_x1*R[1,0] + t_y1*R[1,1] + R[1,2] 
    x2 = t_x2*R[0,0] + t_y2*R[0,1] + R[0,2] 
    y2 = t_x2*R[1,0] + t_y2*R[1,1] + R[1,2] 
    x3 = t_x3*R[0,0] + t_y3*R[0,1] + R[0,2] 
    y3 = t_x3*R[1,0] + t_y3*R[1,1] + R[1,2] 

    if isinstance(x0,torch.Tensor):
        r_box=torch.cat([x0.unsqueeze(0),y0.unsqueeze(0),
                         x1.unsqueeze(0),y1.unsqueeze(0),
                         x2.unsqueeze(0),y2.unsqueeze(0),
                         x3.unsqueeze(0),y3.unsqueeze(0)], 0)
    else:
        r_box = np.array([x0,y0,x1,y1,x2,y2,x3,y3])
    return r_box