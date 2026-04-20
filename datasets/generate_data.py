import os

def generate_iamgets():
    trainset = r'/home/lab/libr/obb-RetinaNet/wave_dataset/ImageSets/Main/train.txt'
    valset = r'/home/lab/libr/obb-RetinaNet/wave_dataset/ImageSets/Main/trainval.txt'
    testset = r'/home/lab/libr/obb-RetinaNet/wave_dataset/ImageSets/Main/test.txt'
    img_dir = r'/home/lab/libr/obb-RetinaNet/datasets/HRSC2016/JPEGImages/'
    label_dir = r'/home/lab/libr/obb-RetinaNet/datasets/HRSC2016/Annotations/'
    root_dir = r'/home/lab/libr/obb-RetinaNet/datasets/HRSC2016/'

    # trainset = r'/home/lab/libr/SWIM_Dataset_1.0.0/ImageSets/Main/train.txt'
    # valset = r'/home/lab/libr/SWIM_Dataset_1.0.0/ImageSets/Main/val.txt'
    # testset = r'/home/lab/libr/SWIM_Dataset_1.0.0/ImageSets/Main/test.txt'
    # img_dir = r'/home/lab/libr/SWIM_Dataset_1.0.0/JPEGImages/'
    # label_dir = r'/home/lab/libr/SWIM_Dataset_1.0.0/Annotations/'
    # root_dir = r'/home/lab/libr/SWIM_Dataset_1.0.0/'

    for dataset in [trainset, valset, testset]:
        with open(dataset, 'r') as f:
            names = f.readlines()
            paths = [os.path.join(img_dir, x.strip() + '.jpg \n') for x in names]
            with open(os.path.join(root_dir, os.path.split(dataset)[1]), 'w') as fw:
                fw.write(''.join(paths))


if __name__ == '__main__':
    generate_iamgets()
