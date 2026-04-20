# import os
# import shutil
#
# # =========================
# # 1. 路径配置（按你自己的改）
# # =========================
# txt_path = r"/home/lab/libr/obb-RetinaNet/wave_dataset/test.txt"
#
# img_dir  = r"/home/lab/libr/obb-RetinaNet/datasets/HRSC2016/JPEGImages"    # 总图像文件夹
# mask_dir = r"/home/lab/libr/obb-RetinaNet/datasets/HRSC2016/Segmentation_masks"     # 总 mask 文件夹
#
# out_img_dir  = r"/home/lab/libr/obb-RetinaNet/datasets/TEST/test_images"
# out_mask_dir = r"/home/lab/libr/obb-RetinaNet/datasets/TEST/test_masks"
#
# IMG_EXTS  = [".jpg", ".png", ".tif", ".tiff"]
# MASK_EXTS = [".png", ".jpg", ".tif", ".tiff"]
#
# # =========================
# # 2. 创建输出目录
# # =========================
# os.makedirs(out_img_dir, exist_ok=True)
# os.makedirs(out_mask_dir, exist_ok=True)
#
# # =========================
# # 3. 读取 txt
# # =========================
# with open(txt_path, "r") as f:
#     lines = [line.strip() for line in f if line.strip()]
#
# print(f"[INFO] Total lines: {len(lines)}")
#
# # =========================
# # 4. 按 txt 拷贝
# # =========================
# miss_img, miss_mask = 0, 0
#
# for full_path in lines:
#     # 👉 关键：只取文件名
#     filename = os.path.basename(full_path)     # 20113.jpg
#     base = os.path.splitext(filename)[0]        # 20113
#
#     # ---------- 图像 ----------
#     img_found = False
#     for ext in IMG_EXTS:
#         img_path = os.path.join(img_dir, base + ext)
#         if os.path.exists(img_path):
#             shutil.copy(img_path, os.path.join(out_img_dir, base + ext))
#             img_found = True
#             break
#
#     if not img_found:
#         print(f"[MISS IMG ] {base}")
#         miss_img += 1
#
#     # ---------- mask ----------
#     mask_found = False
#     for ext in MASK_EXTS:
#         mask_path = os.path.join(mask_dir, base + ext)
#         if os.path.exists(mask_path):
#             shutil.copy(mask_path, os.path.join(out_mask_dir, base + ext))
#             mask_found = True
#             break
#
#     if not mask_found:
#         print(f"[MISS MASK] {base}")
#         miss_mask += 1
#
# # =========================
# # 5. 汇总
# # =========================
# print("=================================")
# print("Done.")
# print(f"Missing images: {miss_img}")
# print(f"Missing masks : {miss_mask}")
# print("=================================")
import os
import shutil

# =========================
# 1. 路径配置
# =========================
txt_path = r"/home/lab/libr/obb-RetinaNet/wave_dataset/test.txt"

img_dir  = r"/home/lab/libr/obb-RetinaNet/wave_dataset/JPEGImages"
mask_dir = r"/home/lab/libr/obb-RetinaNet/datasets/HRSC2016/Segmentation_masks"

# ---- test 输出 ----
test_img_dir  = r"/home/lab/libr/obb-RetinaNet/datasets/TEST/test_images"
test_mask_dir = r"/home/lab/libr/obb-RetinaNet/datasets/TEST/test_masks"

# ---- 剩余数据输出 ----
rest_img_dir  = r"/home/lab/libr/obb-RetinaNet/datasets/TEST/rest_images"
rest_mask_dir = r"/home/lab/libr/obb-RetinaNet/datasets/TEST/rest_masks"

IMG_EXTS  = [".jpg", ".png", ".tif", ".tiff"]
MASK_EXTS = [".png", ".jpg", ".tif", ".tiff"]

# =========================
# 2. 创建输出目录
# =========================
for d in [test_img_dir, test_mask_dir, rest_img_dir, rest_mask_dir]:
    os.makedirs(d, exist_ok=True)

# =========================
# 3. 读取 test.txt（记录 base 名）
# =========================
with open(txt_path, "r") as f:
    test_lines = [line.strip() for line in f if line.strip()]

test_bases = set()
for full_path in test_lines:
    filename = os.path.basename(full_path)
    base = os.path.splitext(filename)[0]
    test_bases.add(base)

print(f"[INFO] Test samples: {len(test_bases)}")

# =========================
# 4. ① 拷贝 test 集
# =========================
miss_img, miss_mask = 0, 0

for base in sorted(test_bases):

    # ---- image ----
    found = False
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(test_img_dir, base + ext))
            found = True
            break
    if not found:
        print(f"[MISS TEST IMG ] {base}")
        miss_img += 1

    # ---- mask ----
    found = False
    for ext in MASK_EXTS:
        p = os.path.join(mask_dir, base + ext)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(test_mask_dir, base + ext))
            found = True
            break
    if not found:
        print(f"[MISS TEST MASK] {base}")
        miss_mask += 1

print(f"[TEST DONE] Missing images: {miss_img}, masks: {miss_mask}")

# =========================
# 5. ② 拷贝剩余样本（总集 - test）
# =========================
all_img_bases = set()

for fname in os.listdir(img_dir):
    base, ext = os.path.splitext(fname)
    if ext.lower() in IMG_EXTS:
        all_img_bases.add(base)

rest_bases = sorted(all_img_bases - test_bases)
print(f"[INFO] Remaining samples: {len(rest_bases)}")

miss_img, miss_mask = 0, 0

for base in rest_bases:

    # ---- image ----
    found = False
    for ext in IMG_EXTS:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(rest_img_dir, base + ext))
            found = True
            break
    if not found:
        print(f"[MISS REST IMG ] {base}")
        miss_img += 1

    # ---- mask ----
    found = False
    for ext in MASK_EXTS:
        p = os.path.join(mask_dir, base + ext)
        if os.path.exists(p):
            shutil.copy(p, os.path.join(rest_mask_dir, base + ext))
            found = True
            break
    if not found:
        print(f"[MISS REST MASK] {base}")
        miss_mask += 1

# =========================
# 6. 汇总
# =========================
print("=================================")
print("ALL DONE.")
print(f"Test samples : {len(test_bases)}")
print(f"Rest samples : {len(rest_bases)}")
print("=================================")
