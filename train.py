from __future__ import print_function

import os
import argparse
import numpy as np
import time
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import matplotlib
matplotlib.use('Agg')  # 放在最前面
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.distributed as dist

from models.model import RetinaNet
from eval import evaluate
from datasets import *
from utils.utils import *
from torch_warmup_lr import WarmupLR
from thop import profile
import thop
mixed_precision = True
try:  
    from apex import amp
except:
    print('fail to speed up training via apex \n')
    mixed_precision = False  # not installed

DATASETS = {'VOC' : VOCDataset ,
            'IC15': IC15Dataset,
            'IC13': IC13Dataset,
            'HRSC2016': HRSCDataset,
            'DOTA':DOTADataset,
            'UCAS_AOD':UCAS_AODDataset,
            'NWPU_VHR':NWPUDataset
            }

def print_flops_params(model, img_size=640, device="cuda"):
    """
    thop: 返回的是 MACs 和 Params
    通常论文口径：GFLOPs ≈ 2 * GMACs
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # ⚠️ 统计时 thop 不太喜欢 DataParallel 包装，优先用 .module
    net = model.module if isinstance(model, torch.nn.DataParallel) else model
    net = net.to(dev)

    # 你的数据是灰度图 → 1 通道；RGB 改成 3
    x = torch.randn(4, 3, img_size, img_size).to(dev)

    with torch.no_grad():
        macs, params = profile(net, inputs=(x,), verbose=False)

    print(f"[Model Complexity @ {img_size}x{img_size}] "
          f"GMACs={macs/1e9:.3f}, GFLOPs≈{macs:.3f}, Params={params/1e6:.3f}M")

    model.train()


def train_model(args, hyps):
    #  parse configs
    epochs = int(hyps['epochs'])
    batch_size = int(hyps['batch_size'])
    results_file = 'result.txt'
    weight =  'weights' + os.sep + 'last.pth' if args.resume or args.load else args.weight
    last = 'weights' + os.sep + 'last.pth'
    best = 'weights' + os.sep + 'best.pth'
    start_epoch = 0
    best_fitness = 0 #   max f1
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # creat folder
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    for f in glob.glob(results_file):
        os.remove(f)

    # multi-scale
    if args.multi_scale:
        scales = args.training_size + 32 * np.array([x for x in range(-1, 5)])
        # set manually
        # scales = np.array([384, 480, 544, 608, 704, 800, 896, 960])
        print('Using multi-scale %g - %g' % (scales[0], scales[-1]))   
    else :
        scales = args.training_size 
############

    # dataloader
    assert args.dataset in DATASETS.keys(), 'Not supported dataset!'
    ds = DATASETS[args.dataset](dataset=args.train_path, augment=args.augment)
    collater = Collater(scales=scales, keep_ratio=True, multiple=32)
    loader = data.DataLoader(
        dataset=ds,
        batch_size=batch_size,
        num_workers=8,
        collate_fn=collater,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # Initialize model
    init_seeds()
    model = RetinaNet(backbone=args.backbone, hyps=hyps)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=hyps['lr0'])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[round(epochs * x) for x in [0.7, 0.9]], gamma=0.1)
    scheduler = WarmupLR(scheduler, init_lr=hyps['warmup_lr'], num_warmup=hyps['warm_epoch'], warmup_strategy='cos')
    scheduler.last_epoch = start_epoch - 1

    if torch.cuda.is_available():
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    # load chkpt
    if weight.endswith('.pth'):
        chkpt = torch.load(weight)
        # load model
        if 'model' in chkpt.keys() :
            model.load_state_dict(chkpt['model'])
        else:
            model.load_state_dict(chkpt)
        # load optimizer
        if 'optimizer' in chkpt.keys() and chkpt['optimizer'] is not None and args.resume :
            optimizer.load_state_dict(chkpt['optimizer'])
            best_fitness = chkpt['best_fitness']
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        # load results
        if 'training_results' in chkpt.keys() and  chkpt.get('training_results') is not None and args.resume:
            with open(results_file, 'w') as file:
                file.write(chkpt['training_results'])  # write results.txt
        if args.resume and 'epoch' in chkpt.keys():
            start_epoch = chkpt['epoch'] + 1   

        del chkpt
 

    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    model_info(model, report='summary')  # 'full' or 'summary'

    # ====== FLOPs/Params（只统计一次）======
    try:
        print_flops_params(model, img_size=args.training_size, device="cuda")
    except Exception as e:
        print(f"[FLOPs] Failed to profile model: {e}")

    results = (0, 0, 0, 0, 0, 0)

    # ===== best 判定：score = 0.6*mAP50 + 0.4*(mIoU+Dice)/2 =====
    best_score = -1.0
    best_epoch = -1
    best_results = None
    best_path = best  # weights/best.pth

    # 仅用于展示（可选）
    best_map50 = -1.0
    best_miou = -1.0
    best_dice = -1.0

    # === 新增：用来画曲线的历史记录 ===
    epoch_list = []
    map_list = []
    miou_list = []
    dice_list = []

    # === 新增：loss 曲线记录（每个 epoch 的 mean loss）===
    loss_epoch_list = []
    loss_cls_list = []
    loss_reg_list = []
    loss_seg_list = []
    loss_total_list = []

    for epoch in range(start_epoch,epochs):
        print(('\n' + '%10s' * 8) % ('Epoch', 'gpu_mem',  'cls', 'reg', 'seg', 'total', 'targets', 'img_size'))
        pbar = tqdm(enumerate(loader), total=len(loader))  # progress bar
        mloss = torch.zeros(3).cuda() # >>> seg: 统计 [cls, reg, seg]

        for i, (ni, batch) in enumerate(pbar):
            
            model.train()

            if args.freeze_bn:
                if torch.cuda.device_count() > 1:
                    model.module.freeze_bn()
                else:
                    model.freeze_bn()

            optimizer.zero_grad()

            ims, gt_boxes = batch['image'], batch['boxes']
            seg_masks = batch.get('mask', None)  # >>> seg: 取分割标签（可能不存在）

            if torch.cuda.is_available():
                ims, gt_boxes = ims.cuda(), gt_boxes.cuda()
                if seg_masks is not None:
                    seg_masks = seg_masks.cuda()

            # >>> seg: 把 seg_masks 传进模型；模型内部应返回 loss_seg（没有分割就返回0）
            # 你的 RetinaNet.forward 需要支持 seg_masks=...（如果未改模型，这里也可以删除该关键字参数）

            losses = model(ims, gt_boxes, gt_seg=batch.get('mask', None), process =epoch/epochs )
            loss_cls, loss_reg = losses['loss_cls'].mean(), losses['loss_reg'].mean()
            # loss = loss_cls + loss_reg

            loss_seg = losses.get('loss_seg', torch.zeros_like(loss_cls))  # >>> seg: 若无分割，置0
            loss = losses.get('loss', loss_cls + loss_reg * (hyps['lambda1']) + loss_seg * (hyps['lambda2']))  # >>> seg: 若模型没给 total，则相加

            if not torch.isfinite(loss):
                import ipdb; ipdb.set_trace()
                print('WARNING: non-finite loss, ending training ')
                break
            if bool(loss == 0):
                continue

            # calculate gradient
            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            # Print batch results
            loss_items = torch.stack([loss_cls, loss_reg, loss_seg], 0).detach()
            mloss = (mloss * i + loss_items) / (i + 1)  # update mean losses
            mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0  # (GB)

            # s = ('%10s' * 2 + '%10.3g' * 5) % (
            #       '%g/%g' % (epoch, epochs - 1), '%.3gG' % mem, *mloss, mloss.sum(), gt_boxes.shape[1], min(ims.shape[2:]))
            # pbar.set_description(s)

            s = ('%10s' * 2 + '%10.3g' * 6) % (
                '%g/%g' % (epoch, epochs - 1),
                '%.3gG' % mem,
                mloss[0], mloss[1]*(hyps['lambda1']), mloss[2]* (hyps['lambda2']), mloss.sum(),  # cls, reg, seg, total(=sum)
                gt_boxes.shape[1], min(ims.shape[2:])
            )
            pbar.set_description(s)

        # === 记录该 epoch 的 mean losses（注意：mloss 是 tensor）=== 1212
        loss_epoch_list.append(epoch)
        loss_cls_list.append(float(mloss[0].detach().cpu()))
        loss_reg_list.append(float((mloss[1] * hyps['lambda1']).detach().cpu()))
        loss_seg_list.append(float((mloss[2] * hyps['lambda2']).detach().cpu()))
        loss_total_list.append(float(
            (mloss[0] + mloss[1] * hyps['lambda1'] + mloss[2] * hyps['lambda2']).detach().cpu()
        ))

        # Update scheduler
        scheduler.step()

        final_epoch = epoch + 1 == epochs
        
        # eval
        if hyps['test_interval']!= -1 and epoch % hyps['test_interval'] == 0 and epoch > 5 :
            if torch.cuda.device_count() > 1:
                results = evaluate(target_size=args.target_size,
                                   test_path=args.test_path,
                                   dataset=args.dataset,
                                   model=model.module, 
                                   hyps=hyps,
                                   conf = 0.01 if final_epoch else 0.1)    
            else:
                results = evaluate(target_size=args.target_size,
                                   test_path=args.test_path,
                                   dataset=args.dataset,
                                   model=model,
                                   hyps=hyps,
                                   conf = 0.01 if final_epoch else 0.1) #  p, r, map, f1

        
        # Write result log
        with open(results_file, 'a') as f:
            f.write(s + '%10.3g' * 6 % results + '\n')  # P, R, mAP, F1, test_losses=(GIoU, obj, cls)

        P, R, mAP, F1 = results[:4]
        miou, dice = results[4], results[5]

        # 记录当前 epoch 指标（只有在做了 evaluate 时结果才不是全 0）
        if hyps['test_interval'] != -1 and epoch % hyps['test_interval'] == 0 and epoch > 5:
            epoch_list.append(epoch)
            map_list.append(mAP)
            miou_list.append(miou)
            dice_list.append(dice)

        # 判定当前 epoch 是否是新的 best（仅在 evaluate 点才判）
        is_best = False
        if hyps['test_interval'] != -1 and epoch % hyps['test_interval'] == 0 and epoch > 5:
            # score = 0.6*mAP50 + 0.4*avg(mIoU, Dice)  1217.pt是只看分割结果来选best
            seg_avg = 1 * (miou + dice)
            # score = 0.2 * mAP + seg_avg(miou, dice)
            score = seg_avg

            # 主判据：score 更大者为 best
            # 次级判据：score 相等时，优先 mIoU，再 Dice，再 mAP（避免抖动）
            if score > best_score:
                is_best = True
            elif score == best_score:
                if miou > best_miou:
                    is_best = True
                elif miou == best_miou and dice > best_dice:
                    is_best = True
                elif miou == best_miou and dice == best_dice and mAP > best_map50:
                    is_best = True

            if is_best:
                best_score = score
                best_epoch = epoch
                best_results = results[:]  # 保存完整 6 个指标

                # 仅用于展示（可选）
                best_map50 = mAP
                best_miou = miou
                best_dice = dice

                # 你原来 best_fitness 用于 ckpt 里记录，这里直接存 score
                best_fitness = best_score
        else:
            # 非评测点不判优
            is_best = False


        with open(results_file, 'r') as f:
            # Create checkpoint
            # chkpt = {'epoch': epoch,
            #          'best_fitness': best_fitness,
            #          'training_results': f.read(),
            #          'model': model.module.state_dict() if type(
            #             model) is nn.parallel.DistributedDataParallel else model.state_dict(),
            #          'optimizer': None if final_epoch else optimizer.state_dict()}
            chkpt = {
                'epoch': epoch,
                'best_fitness': best_fitness,
                'best_epoch': best_epoch,  # 新增
                'best_results': best_results,  # 新增
                'training_results': f.read(),
                'model': model.module.state_dict() if type(
                    model) is nn.parallel.DistributedDataParallel else model.state_dict(),
                'optimizer': None if final_epoch else optimizer.state_dict()
            }
        

        # Save last checkpoint
        torch.save(chkpt, last)

        # 2) 只有当这一轮是新的 best 时，才覆盖 best.pth
        if is_best:
            torch.save(chkpt, best_path)  # ⭐ 关键补这一句

        # # Save best checkpoint
        # if best_fitness == fitness:
        #     torch.save(chkpt, best)  # 1124

        if (epoch % hyps['save_interval'] == 0  and epoch > 100) or final_epoch:
            if torch.cuda.device_count() > 1:
                torch.save(chkpt, './weights/deploy%g.pth'% epoch)
            else:
                torch.save(chkpt, './weights/deploy%g.pth'% epoch)

    # === 新增：训练结束后分别画三个指标曲线 ===
    try:
        if len(epoch_list) > 0:
            import matplotlib
            matplotlib.use('Agg')  # 使用非GUI后端
            os.makedirs('weights', exist_ok=True)

            # ===== Plot 1: mAP 曲线 =====
            try:
                plt.figure(figsize=(6, 4))
                plt.plot(epoch_list, map_list, marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('mAP@0.5')
                plt.title('mAP@0.5 over Epochs')
                plt.grid(True)
                out_fig = os.path.join('weights', 'map_curve.png')
                plt.savefig(out_fig, dpi=300, bbox_inches='tight')
                plt.close()
                print(f'[Plot] mAP curve saved to {out_fig}')
            except Exception as e:
                print(f'[Plot] Failed to plot mAP curve: {e}')

            # ===== Plot 2: mIoU 曲线 =====
            try:
                plt.figure(figsize=(6, 4))
                plt.plot(epoch_list, miou_list, color='green', marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('mIoU')
                plt.title('mIoU over Epochs')
                plt.grid(True)
                out_fig = os.path.join('weights', 'miou_curve.png')
                plt.savefig(out_fig, dpi=300, bbox_inches='tight')
                plt.close()
                print(f'[Plot] mIoU curve saved to {out_fig}')
            except Exception as e:
                print(f'[Plot] Failed to plot mIoU curve: {e}')

            # ===== Plot 3: Dice 曲线 =====
            try:
                plt.figure(figsize=(6, 4))
                plt.plot(epoch_list, dice_list, color='red', marker='o')
                plt.xlabel('Epoch')
                plt.ylabel('Dice')
                plt.title('Dice over Epochs')
                plt.grid(True)
                out_fig = os.path.join('weights', 'dice_curve.png')
                plt.savefig(out_fig, dpi=300, bbox_inches='tight')
                plt.close()
                print(f'[Plot] Dice curve saved to {out_fig}')
            except Exception as e:
                print(f'[Plot] Failed to plot Dice curve: {e}')

            # ===== Plot 4: Loss curves (cls/reg/seg/total) =====
            try:
                if len(loss_epoch_list) > 0:
                    plt.figure(figsize=(7, 4))
                    plt.plot(loss_epoch_list, loss_cls_list, marker='o', label='cls')
                    plt.plot(loss_epoch_list, loss_reg_list, marker='o', label='reg')
                    plt.plot(loss_epoch_list, loss_seg_list, marker='o', label='seg')
                    plt.plot(loss_epoch_list, loss_total_list, marker='o', label='total')
                    plt.xlabel('Epoch')
                    plt.ylabel('Loss')
                    plt.title('Loss over Epochs')
                    plt.grid(True)
                    plt.legend()
                    out_fig = os.path.join('weights', 'loss_curve.png')
                    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f'[Plot] Loss curve saved to {out_fig}')
                else:
                    print('[Plot] No loss points recorded, skip loss plotting.')
            except Exception as e:
                print(f'[Plot] Failed to plot loss curve: {e}')

        else:
            print('[Plot] No evaluation points recorded, skip plotting.')

    except Exception as e:
        print(f'[Plot] Failed to plot metric curves: {e}')

    # ===== 训练结束后用 best.pth 跑一次 evaluate 并汇报 =====
    try:
        print('\n[Final] Loading best checkpoint and running evaluation ...')

        # 用 best.pth + 你现有的 evaluate 统一评测
        results_best = evaluate(
            target_size=args.target_size,
            test_path=args.test_path,
            dataset=args.dataset,
            backbone=args.backbone,
            weight=best_path,      # weights/best.pth
            hyps=hyps,
            conf=0.01              # 终盘评测阈值
        )

        # 兼容不同返回长度（你现在 data_evaluate 返回 6 个：P,R,mAP,Hmean,mIoU,Dice）
        P = R = mAP = F1 = miou = dice = None
        if isinstance(results_best, (list, tuple)):
            if len(results_best) >= 4:
                P, R, mAP, F1 = results_best[:4]
            if len(results_best) >= 6:
                miou, dice = results_best[4], results_best[5]

        # 汇总字符串
        lines = []
        lines.append('================ BEST RESULT ================')
        lines.append(f' Best epoch: {best_epoch}')
        if P is not None:
            lines.append(f' P: {P:.4f} | R: {R:.4f} | mAP@0.5: {mAP:.4f} | F1: {F1:.4f}')
        if miou is not None:
            lines.append(f' mIoU: {miou:.4f} | Dice: {dice:.4f}')

        # 与训练过程中选优的一致：明确说明是 “mIoU 优先，Dice 次之”
        lines.append(
            f' Best metric (score = 0.6*mAP50 + 0.4*avg(mIoU,Dice)): '
            f'score={best_score:.4f} | mAP50={best_map50:.4f} | mIoU={best_miou:.4f} | Dice={best_dice:.4f}'
        )
        # 如果想顺便看组合分数：
        # lines.append(f' Combined score 0.5*(mIoU+Dice): {best_fitness:.4f}')

        lines.append(f' Best checkpoint: {best_path}')
        lines.append('============================================')

        summary = '\n'.join(lines)
        print('\n' + summary + '\n')

        # 同步写到文件
        with open(os.path.join('weights', 'best_summary.txt'), 'w', encoding='utf-8') as fw:
            fw.write(summary + '\n')

    except Exception as e:
        print(f'[Final] evaluate(best.pth) failed: {e}')


    # end training
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train a detector')
    # config
    parser.add_argument('--hyp', type=str, default='hyp.py', help='hyper-parameter path')
    # network
    parser.add_argument('--backbone', type=str, default='fca101')  # res50
    parser.add_argument('--freeze_bn', type=bool, default=False)
    parser.add_argument('--weight', type=str, default='')   # 
    parser.add_argument('--multi-scale', action='store_true', help='adjust (67% - 150%) img_size every 10 batches')

     # HRSC
    # parser.add_argument('--dataset', type=str, default='HRSC2016')
    # parser.add_argument('--train_path', type=str, default='HRSC2016/train.txt')
    # parser.add_argument('--test_path', type=str, default='HRSC2016/test.txt')

    # HRSC
    parser.add_argument('--dataset', type=str, default='HRSC2016')
    parser.add_argument('--train_path', type=str, default='datasets/HRSC2016/train.txt')
    parser.add_argument('--test_path', type=str, default='datasets/HRSC2016/test.txt')

    # DOTA
    # parser.add_argument('--dataset', type=str, default='DOTA')    
    # parser.add_argument('--train_path', type=str, default='DOTA/trainval.txt')

    parser.add_argument('--training_size', type=int, default=640)
    parser.add_argument('--resume', action='store_true', help='resume training from last.pth')
    parser.add_argument('--load', action='store_true', help='load training from last.pth')
    parser.add_argument('--augment', action='store_true', help='data augment')
    parser.add_argument('--target_size', type=int, default=[640])


    arg = parser.parse_args()
    hyps = hyp_parse(arg.hyp)
    print(arg)
    print(hyps)

    train_model(arg, hyps)