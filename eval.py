from __future__ import print_function

import os
import cv2
import torch
import codecs
import zipfile
import shutil
import argparse
import sys
sys.path.append('datasets/DOTA_devkit')

from tqdm import tqdm
from datasets import *
from models.model import RetinaNet
from utils.detect import im_detect
from utils.bbox import rbox_2_aabb, rbox_2_quad
from utils.utils import sort_corners, is_image, hyp_parse
from utils.map import eval_mAP
import numpy as np
import torch.nn.functional as F
from datasets.DOTA_devkit.ResultMerge_multi_process import ResultMerge
from datasets.DOTA_devkit.dota_evaluation_task1 import task1_eval
from utils.im_segment import im_segment


DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'IC13': IC13Dataset,
            'HRSC2016': HRSCDataset,
            'DOTA':DOTADataset,
            'UCAS_AOD':UCAS_AODDataset,
            'NWPU_VHR':NWPUDataset
            }


def _bin_pred_from_logits(seg_logits, thr=0.25):
    # seg_logits: (1,1,H,W) 或 (B,1,H,W)
    probs = torch.sigmoid(seg_logits)
    return (probs > thr).long()

def _iou_dice(pred, target, eps=1.0):
    # pred/target: (1,1,H,W) 或 (1,H,W) 的 0/1
    if pred.ndim == 4:
        pred = pred.squeeze(0)
    if target.ndim == 4:
        target = target.squeeze(0)
    if target.ndim == 3:
        target = target[0]  # (1,H,W)->(H,W)
    inter = (pred & target).float().sum()
    union = (pred | target).float().sum()
    iou  = (inter + eps) / (union + eps)
    dice = (2 * inter + eps) / (pred.float().sum() + target.float().sum() + eps)
    return float(iou), float(dice)


def make_zip(source_dir, output_filename):
    zipf = zipfile.ZipFile(output_filename, 'w')
    # pre_len = len(os.path.dirname(source_dir))
    for parent, dirnames, filenames in os.walk(source_dir):
        for filename in filenames:
            pathfile = os.path.join(parent, filename)
            # arcname = pathfile[pre_len:].strip(os.path.sep)
            zipf.write(pathfile, filename)
    zipf.close()


def icdar_evaluate(model, 
                   target_size, 
                   test_path, 
                   dataset):
    if dataset == 'IC15':
        output = './datasets/IC_eval/icdar15'
    elif dataset == 'IC13':
        output = './datasets/IC_eval/icdar13'
    else:
        raise NotImplementedError

    ims_dir = test_path
    out_dir = './temp'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    ims_list = [x for x in os.listdir(ims_dir) if is_image(x)]
    s = ('%20s' + '%10s' * 8) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean', 'mIoU', 'Dice')
    nt = 0

    for idx, im_name in enumerate(tqdm(ims_list, desc=s)):
        im_path = os.path.join(ims_dir, im_name)
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model, im, target_sizes=target_size)
        nt += len(dets)
        out_file = os.path.join(out_dir, 'res_' + im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                continue
            if dataset == 'IC15':
                res = sort_corners(rbox_2_quad(dets[:, 2:]))
                for k in range(dets.shape[0]):
                    f.write('{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f},{:.0f}\n'.format(
                        res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                        res[k, 4], res[k, 5], res[k, 6], res[k, 7])
                    )
            if dataset == 'IC13':
                res = rbox_2_aabb(dets[:, 2:])
                for k in range(dets.shape[0]):
                    f.write('{:.0f},{:.0f},{:.0f},{:.0f}\n'.format(
                        res[k, 0], res[k, 1], res[k, 2], res[k, 3])
                    )


    zip_name = 'submit.zip'
    make_zip(out_dir, zip_name)
    shutil.move(os.path.join('./', zip_name), os.path.join(output, zip_name))
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    result = os.popen('cd {0} && python script.py -g=gt.zip -s=submit.zip '.format(output)).read()
    sep = result.split(':')
    precision = sep[1][:sep[1].find(',')].strip()
    recall = sep[2][:sep[2].find(',')].strip()
    f1 = sep[3][:sep[3].find(',')].strip()
    map = 0
    p = eval(precision)
    r = eval(recall)
    hmean = eval(f1)
    # display result
    pf = '%20s' + '%10.3g' * 8  # print format
    print(pf % ('all', len(ims_list), nt, p, r, 0, hmean))
    return p, r, map, hmean 



# def data_evaluate(model,
#                   target_size,
#                   test_path,
#                   conf = 0.01,
#                   dataset=None):
#     root_dir = 'datasets/evaluate'
#     out_dir = os.path.join(root_dir,'detection-results')
#     if  os.path.exists(out_dir):
#         shutil.rmtree(out_dir)
#     os.makedirs(out_dir)
#
#     ds = DATASETS[dataset]()
#
#     with open(test_path,'r') as f:
#         if dataset == 'VOC':
#             im_dir = test_path.replace('/ImageSets/Main/test.txt','/JPEGImages')
#             ims_list = [os.path.join(im_dir, x.strip('\n')+'.jpg') for x in f.readlines()]
#         else:
#             ims_list = [x.strip('\n') for x in f.readlines() if is_image(x.strip('\n'))]
#     s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean')
#     nt = 0
#     for idx, im_path in enumerate(tqdm(ims_list, desc=s)):
#         im_name = os.path.split(im_path)[1]
#         im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
#         dets = im_detect(model, im, target_sizes=target_size, conf = conf)
#         nt += len(dets)
#         out_file = os.path.join(out_dir,  im_name[:im_name.rindex('.')] + '.txt')
#         with codecs.open(out_file, 'w', 'utf-8') as f:
#             if dets.shape[0] == 0:
#                 f.close()
#                 continue
#             res = sort_corners(rbox_2_quad(dets[:, 2:]))
#             for k in range(dets.shape[0]):
#                 f.write('{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
#                     ds.return_class(dets[k, 0]), dets[k, 1],
#                     res[k, 0], res[k, 1], res[k, 2], res[k, 3],
#                     res[k, 4], res[k, 5], res[k, 6], res[k, 7])
#                 )
#         assert len(os.listdir(os.path.join(root_dir,'ground-truth'))) != 0, 'No labels found in test/ground-truth!! '
#     mAP = eval_mAP(root_dir, use_07_metric=False)
#     # display result
#     pf = '%20s' + '%10.3g' * 6  # print format
#     print(pf % ('all', len(ims_list), nt, 0, 0, mAP, 0))
#     return 0, 0, mAP, 0
def data_evaluate(model,
                  target_size,
                  test_path,
                  conf=0.01,
                  dataset=None):
    root_dir = 'datasets/evaluate'
    gt_dir = os.path.join(root_dir, 'ground-truth')
    out_dir = os.path.join(root_dir, 'detection-results')

    # 重新生成 detection-results
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    ds = DATASETS[dataset]()  # 只为 return_class 用

    # 读取评测图片清单
    with open(test_path, 'r', encoding='utf-8') as f:
        if dataset == 'VOC':
            im_dir = test_path.replace('/ImageSets/Main/test.txt', '/JPEGImages')
            ims_list = [os.path.join(im_dir, x.strip() + '.jpg') for x in f if x.strip()]
        else:
            ims_list = [x.strip() for x in f if x.strip()]

    # 保险：扩展名判断（is_image 可能不覆盖所有大小写）
    def _is_image(p):
        ext = os.path.splitext(p)[1].lower()
        return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    ims_list = [p for p in ims_list if _is_image(p)]
    assert len(ims_list) > 0, f'Empty test list from {test_path}'

    header = ('%20s' + '%10s' * 8) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean', 'mIoU', 'Dice')
    nt = 0

    print(f'[eval] images to run: {len(ims_list)}')
    for idx, im_path in enumerate(tqdm(ims_list, desc=header)):
        im_name = os.path.basename(im_path)
        base = os.path.splitext(im_name)[0]  # 20013
        out_file = os.path.join(out_dir, base + '.txt')

        src = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if src is None:
            print(f'[warn] cannot read image: {im_path}')
            # 也创建空文件，保证评测不缺文件
            open(out_file, 'w', encoding='utf-8').close()
            continue

        im = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        dets = im_detect(model, im, target_sizes=target_size, conf=conf)

        # 记录一下数量
        n_this = 0 if dets is None else int(dets.shape[0])
        nt += n_this
        # 关键：无论有没有 dets，都要创建出文件
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if n_this == 0:
                # 留空文件即可
                pass
            else:
                res = sort_corners(rbox_2_quad(dets[:, 2:]))
                for k in range(dets.shape[0]):
                    f.write('{} {:.2f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f}\n'.format(
                        ds.return_class(dets[k, 0]), dets[k, 1],
                        res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                        res[k, 4], res[k, 5], res[k, 6], res[k, 7])
                    )
        # # 调试输出：哪张图生成了哪个文件、dets 数
        # if base == '20013':
        #     print(f'[eval] touched {out_file}, dets={n_this}')

    # 基础健壮性：确保有 GT
    assert os.path.isdir(gt_dir) and len(os.listdir(gt_dir)) != 0, \
        f'No labels found in {gt_dir} !! '

    # 交叉检查：哪些 GT 没有对应 DR（最直观定位 20013）
    gt_bases = {os.path.splitext(x)[0] for x in os.listdir(gt_dir) if x.endswith('.txt')}
    dr_bases = {os.path.splitext(x)[0] for x in os.listdir(out_dir) if x.endswith('.txt')}
    missing = sorted(list(gt_bases - dr_bases))
    if missing:
        print('[warn] DR missing for these GT files:', missing[:20], '... total', len(missing))

    mAP = eval_mAP(root_dir, use_07_metric=False)

    # === 新增：分割评估（若不想这里评，就在外层 evaluate 里评；二者选一个避免重复）===
    try:
        miou, dice = seg_evaluate(model, target_size, test_path, dataset, thr=0.25)
    except Exception as e:
        print('[seg_eval] failed in data_evaluate:', e)
        miou, dice = 0.0, 0.0

    pf = '%20s' + '%10.3g' * 8
    print(pf % ('all', len(ims_list), nt, 0, 0, mAP, 0, miou, dice))
    return 0, 0, mAP, 0, miou, dice



def dota_evaluate(model, 
                  target_size, 
                  test_path,
                  conf = 0.01):
    # 
    root_data, evaldata = os.path.split(test_path)
    splitdata = evaldata + 'split'
    ims_dir = os.path.join(root_data, splitdata + '/' + 'images')
    root_dir = 'outputs'
    res_dir = os.path.join(root_dir, 'detections')          # 裁剪图像的检测结果   
    integrated_dir = os.path.join(root_dir, 'integrated')   # 将裁剪图像整合后成15个txt的结果
    merged_dir = os.path.join(root_dir, 'merged')           # 将整合后的结果NMS

    if  os.path.exists(root_dir):
        shutil.rmtree(root_dir)
    os.makedirs(root_dir)

    for f in [res_dir, integrated_dir, merged_dir]: 
        if os.path.exists(f):
            shutil.rmtree(f)
        os.makedirs(f)

    ds = DOTADataset()
    # loss = torch.zeros(3)
    ims_list = [x for x in os.listdir(ims_dir) if is_image(x)]
    s = ('%20s' + '%10s' * 8) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'Hmean', 'mIoU', 'Dice')
    nt = 0
    for idx, im_name in enumerate(tqdm(ims_list, desc=s)):
        im_path = os.path.join(ims_dir, im_name)
        im = cv2.cvtColor(cv2.imread(im_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        dets = im_detect(model, im, target_sizes=target_size, conf = conf)
        nt += len(dets)
        out_file = os.path.join(res_dir,  im_name[:im_name.rindex('.')] + '.txt')
        with codecs.open(out_file, 'w', 'utf-8') as f:
            if dets.shape[0] == 0:
                f.close()
                continue
            res = sort_corners(rbox_2_quad(dets[:, 2:]))
            for k in range(dets.shape[0]):
                f.write('{:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {} {} {:.2f}\n'.format(
                    res[k, 0], res[k, 1], res[k, 2], res[k, 3],
                    res[k, 4], res[k, 5], res[k, 6], res[k, 7],
                    ds.return_class(dets[k, 0]), im_name[:-4], dets[k, 1],)
                )
    ResultMerge(res_dir, integrated_dir, merged_dir)
    ## calc mAP
    mAP, classaps = task1_eval(merged_dir, test_path)

    try:
        miou, mdice = seg_evaluate(model, target_size, test_path, 'DOTA', thr=0.25)
    except Exception as e:
        print('[seg_eval] failed in dota_evaluate:', e)
        miou, mdice = 0.0, 0.0

    # # display result
    pf = '%20s' + '%10.3g' * 8  # print format
    print(pf % ('all', len(ims_list), nt, 0, 0, mAP, 0, miou, mdice))
    return 0, 0, mAP, 0, miou, mdice

def seg_evaluate(model, target_size, test_path, dataset, thr=0.25):
    """
    返回 (mIoU, mDice)
    依赖：DATASETS[dataset] 的 _load_mask(img_path) 能取到对应 mask
    """
    if dataset not in DATASETS:
        print(f'[seg_eval] dataset {dataset} not registered; skip.')
        return 0.0, 0.0

    # 用数据集类来按规则定位 mask
    # ds = DATASETS[dataset]()  # 若你需要自定义 mask_root，可在这里传入
    ds = DATASETS[dataset](
        mask_root='/home/lab/libr/obb-RetinaNet/wave_dataset/Segmentation_masks',  # 你的mask根目录
        mask_binary=True,  # 二值化mask
        mask_positive_values=(255,)  # 白色为前景
    )

    # 收集测试图片清单（和 data_evaluate 保持一致）
    with open(test_path, 'r', encoding='utf-8') as f:
        if dataset == 'VOC':
            im_dir = test_path.replace('/ImageSets/Main/test.txt', '/JPEGImages')
            ims_list = [os.path.join(im_dir, x.strip() + '.jpg') for x in f if x.strip()]
        else:
            ims_list = [x.strip() for x in f if x.strip()]

    def _is_image(p):
        ext = os.path.splitext(p)[1].lower()
        return ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

    ims_list = [p for p in ims_list if _is_image(p)]
    if not ims_list:
        print('[seg_eval] empty test list; skip.')
        return 0.0, 0.0

    device = next(model.parameters()).device
    iou_sum = 0.0
    dice_sum = 0.0
    n_valid = 0

    header = ('%20s' + '%10s' * 2) % ('SegEval', 'mIoU', 'Dice')
    for im_path in tqdm(ims_list, desc=header):
        bgr = cv2.imread(im_path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W = rgb.shape[:2]

        # 1) 取 GT mask（失败就跳过）
        try:
            gt = ds._load_mask(im_path, target_hw=(H, W))
        except Exception as e:
            # 某些数据集未实现 _load_mask 或找不到对应 mask
            # print(f'[seg_eval] skip {os.path.basename(im_path)}: {e}')
            continue
        if gt is None or gt.size == 0:
            continue
        gt = (gt > 0).astype(np.uint8)  # 二值
        gt_t = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)

        # 2) 模型前向，显式取 seg_logits；失败就跳过
        # 与训练一致的归一化（ImageNet）
        im = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        im = (im - mean) / std
        im_t = torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float().to(device)

        with torch.no_grad():
            outs = model(im_t, return_seg=True)
            if isinstance(outs, dict) and ('seg_logits' in outs):
                seg_logits = outs['seg_logits']  # (1,1,h',w')
            else:
                # 兼容某些实现直接返回张量
                if torch.is_tensor(outs):
                    seg_logits = outs
                else:
                    # print(f'[seg_eval] no seg_logits for {os.path.basename(im_path)}, skip.')
                    continue

            # 上采样到原图尺寸
            seg_logits = F.interpolate(seg_logits, size=(H, W), mode='bilinear', align_corners=False)

            # 3) 阈值化得到预测
            prob = torch.sigmoid(seg_logits)
            pred = (prob > thr).to(gt_t.dtype)  # (1,1,H,W) uint8/byte

            # 4) IoU / Dice（同设备）
            inter = (pred & gt_t).float().sum()
            union = (pred | gt_t).float().sum().clamp_min(1.0)
            iou   = (inter / union).item()

            denom = (pred.float().sum() + gt_t.float().sum()).clamp_min(1.0)
            dice  = (2.0 * inter / denom).item()

        iou_sum += iou
        dice_sum += dice
        n_valid += 1

    if n_valid == 0:
        print('[seg_eval] no valid masks; skip.')
        return 0.0, 0.0

    miou  = iou_sum / n_valid
    mdice = dice_sum / n_valid
    # print(f'[seg_eval] mIoU={miou:.4f}, Dice={mdice:.4f}  (over {n_valid} images)')
    return miou, mdice


def evaluate(target_size,
             test_path,
             dataset,
             backbone=None, 
             weight=None, 
             model=None,
             hyps=None,
             conf=0.3):
    if model is None:
        model = RetinaNet(backbone=backbone,hyps=hyps)
        if torch.cuda.is_available():
            model.cuda()
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model).cuda()
            
        if weight.endswith('.pth'):
            chkpt = torch.load(weight)
            # load model
            if 'model' in chkpt.keys():
                model.load_state_dict(chkpt['model'])
            else:
                model.load_state_dict(chkpt)

    model.eval()

    if 'IC' in dataset :
        results = icdar_evaluate(model, target_size, test_path, dataset)
    elif dataset in ['HRSC2016', 'UCAS_AOD', 'VOC', 'NWPU_VHR']:
        results = data_evaluate(model, target_size, test_path, conf, dataset)
    elif dataset == 'DOTA':
        results = dota_evaluate(model, target_size, test_path, conf)
    else:
        raise RuntimeError('Unsupported dataset!')

    # # === 新增：分割评估（可选）===
    # try:
    #     miou, mdice = seg_evaluate(model, target_size, test_path, dataset, thr=0.5)
    #     results = (*results, miou, mdice)  # 现在是 6 个数
    # except Exception as e:
    #     print('[seg_eval] failed:', e)
    #     results = (*results, 0.0, 0.0)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--backbone', dest='backbone', default='fca101', type=str)
    parser.add_argument('--weight', type=str, default='weights/best.pth')
    parser.add_argument('--target_size', dest='target_size', default=[640], type=int)
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    
    # parser.add_argument('--dataset', nargs='?', type=str, default='DOTA')
    # parser.add_argument('--test_path', nargs='?', type=str, default='/home/lab/libr/obb-RetinaNet/wave_dataset/test/Sentinel-2-0902')

    # parser.add_argument('--dataset', nargs='?', type=str, default='IC13')
    # parser.add_argument('--test_path', type=str, default='ICDAR13/test') 

    parser.add_argument('--dataset', nargs='?', type=str, default='HRSC2016')
    parser.add_argument('--test_path', type=str, default='datasets/HRSC2016/test.txt')
   
    # parser.add_argument('--dataset', nargs='?', type=str, default='NWPU_VHR')
    # parser.add_argument('--test_path', type=str, default='NWPU_VHR/test.txt')

    arg = parser.parse_args()
    hyps = hyp_parse(arg.hyp)
    evaluate(arg.target_size,
             arg.test_path,
             arg.dataset,
             arg.backbone,
             arg.weight,
             hyps = hyps)