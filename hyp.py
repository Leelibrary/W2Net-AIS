# lr
lr0: 0.0001
warmup_lr: 0.00001
warm_epoch:8


# setting
num_classes: 1

# training
epochs: 120
batch_size: 4
save_interval: 10
test_interval: 2

# loss
lambda1: 1
lambda2: 1

# ---- segmentation ----
seg_pos_weight: 12.0    # BCE 的正样本权重；前景越稀疏可适当加大(如 8~20)
seg_bce_w: 0.3    # BCE 权重
seg_dice_w: 0.7   # Dice 权重

# loss: 1 4  seg-weight=12  batch = 8   0.829    0.315     0.465
