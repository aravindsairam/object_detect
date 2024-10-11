import os
from pathlib import Path
import shutil
import pandas as pd
import numpy as np

def visdrone2yolo(dir):
    from PIL import Image
    from tqdm import tqdm

    def convert_box(size, box):
        # Convert VisDrone box to YOLO xywh box
        dw = 1. / size[0]
        dh = 1. / size[1]
        return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh

    Path(os.path.join(dir, 'labels')).mkdir(parents=True, exist_ok=True)  # make labels directory
    pbar = tqdm(Path(os.path.join(dir, 'annotations')).glob('*.txt'), desc=f'Converting {dir}')
    for f in pbar:
        img_size = Image.open(Path(os.path.join(dir, 'images', f.name)).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:  # read annotation.txt
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
                with open(str(f).replace(f'{os.sep}annotations{os.sep}', f'{os.sep}labels{os.sep}'), 'w') as fl:
                    fl.writelines(lines)  # write label.txt


def video2images(IpAnnPath, IpImgPath, OpAnnPath, OpImgPath):

    def arrageImgs(IpImgPath, OpImgPath):
        Path(OpImgPath).mkdir(parents=True, exist_ok=True)
        image_sequences = os.listdir(IpImgPath)
        for seq in image_sequences:
            for img in os.listdir(os.path.join(IpImgPath, seq)):
                shutil.move(os.path.join(IpImgPath, seq, img), os.path.join(OpImgPath, seq+"_"+img))

    def arrageAnns(IpAnnPath, OpAnnPath):
        Path(OpAnnPath).mkdir(parents=True, exist_ok=True)
        image_annotations = os.listdir(IpAnnPath)
        for ann in image_annotations:
            df = pd.read_csv(os.path.join(IpAnnPath,ann), header=None)
            unique_df = df[0].unique()
            for i in range(len(unique_df)):
                # naming the txt file
                add_zeros = "0000" if i >= 99 else "00000" # this condition does not work, change it
                txt_name = add_zeros + str(unique_df[i]) + ".txt"
                ann_name = ann.split(".")[0]+"_"+txt_name

                # get annotations for each image
                filtered_df = df[df[0] == 1]
                with open(os.path.join(OpAnnPath, ann_name), 'w') as f:
                    for _, row in filtered_df.iterrows():
                        ann_lines = ",".join([str(i) for i in row[2:]])
                        f.write(ann_lines + '\n')

    arrageImgs(IpImgPath, OpImgPath)
    arrageAnns(IpAnnPath, OpAnnPath)
    

def processVisDroneDET():
    data_path = "/home/sai/drone_ws/src/object_detect/data/VisDrone"

    for d in ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-VID-DET-dev']:
        visdrone2yolo(os.path.join(data_path, d))  # convert VisDrone annotations to YOLO labels

def processVisDroneVID():
    data_path = "/home/sai/drone_ws/src/object_detect/data/VisDrone/VisVideo"
    for d in ['VisDrone2019-VID-train', 'VisDrone2019-VID-val']: #, 'VisDrone2019-VID-test-dev']:
        video2images(os.path.join(data_path, d, 'old_annotations'),
                     os.path.join(data_path, d, 'sequences'),
                     os.path.join(data_path, d, 'annotations'),
                     os.path.join(data_path, d, 'images'))


if __name__ == '__main__':
    processVisDroneDET()
    # processVisDroneVID()
    # img_list = os.listdir("/home/sai/drone_ws/src/object_detect/data/VisDrone/VisVideo/VisDrone2019-VID-test-dev/images")
    # ann_list = os.listdir("/home/sai/drone_ws/src/object_detect/data/VisDrone/VisVideo/VisDrone2019-VID-test-dev/annotations")
    # not_include = []
    # for img in img_list:
    #     img = img.split(".")[0] + ".txt"
    #     if img not in ann_list:
    #         not_include.append(img)

    # print(len(not_include))
    # print(len(img_list))
    # print(len(ann_list))
    # print(not_include[:10])
