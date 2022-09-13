import os
import pprint
import json
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

from .boxes import xywh_to_xyxy
from .draw import draw_boxes_with_label

pp = pprint.PrettyPrinter()


def read_box_ann(ann_file):
    if not isinstance(ann_file, list):
        ann_file = open(ann_file)
        ann_file = json.load(ann_file)

    xyxys, labels = [], []
    for data in ann_file:
        x = data['x']
        y = data['y']
        w = data['w']
        h = data['h']
        label = data['label']
        xyxy = xywh_to_xyxy((x, y, w, h))

        xyxys.append(xyxy)
        labels.append(label)

    return xyxys, labels


def read_ocr_ann(ann_file):
    if not isinstance(ann_file, list):
        ann_file = open(ann_file)
        ann_file = json.load(ann_file)

    xyxys, labels = [], []
    for _dict in ann_file:
        if _dict["label"] == "drugname":
            xyxy = _dict["box"]
            pill_id = _dict["mapping"]

            xyxys.append(xyxy)
            labels.append(pill_id)

    return xyxys, labels


def inspect_images(dir, image_dir, ann_dir, img_name=None, ext='.jpg'):
    image_folder = os.path.join(dir, image_dir)
    ann_folder = os.path.join(dir, ann_dir)

    for file in tqdm(os.listdir(ann_folder)):
        name = file.split('.')[0]
        if img_name is not None:
            name = img_name

        ann_file = os.path.join(ann_folder, f"{name}.json")
        image_file = f"{image_folder}/{name}{ext}"

        xyxys, labels = read_box_ann(ann_file)
        image = cv2.imread(image_file)

        img = draw_boxes_with_label(image, xyxys, labels)
        plt.imshow(img[:, :, ::-1])
        plt.show()

        if img_name is not None:
            break


def inspect_images_pres(dir, image_dir, ann_dir, pres_dir, pre_im_dir, pres_label_dir, pill_preds_map, name=None):
    """
    :param dir:
    :param image_dir:
    :param ann_dir:
    :param pres_dir:
    :param pre_im_dir:
    :param pres_label_dir:
    :param pill_preds_map:
    :param ext:
    :return:
    """
    image_folder = os.path.join(dir, image_dir)
    ann_folder = os.path.join(dir, ann_dir)

    pres_img_folder = os.path.join(pres_dir, pre_im_dir)
    pres_label_folder = os.path.join(pres_dir, pres_label_dir)

    if not isinstance(pill_preds_map, list):
        pill_preds_map = open(pill_preds_map)
        pill_preds_map = json.load(pill_preds_map)

    for data_dict in tqdm(pill_preds_map):
        pres_name = data_dict["pres"].split('.')[0]
        pill_list = data_dict["pill"]

        pres_img_file = f"{pres_img_folder}/{pres_name}.png"
        pres_ann_file = f"{pres_label_folder}/{pres_name}.json"

        pres_img = cv2.imread(pres_img_file)
        pres_ann = read_ocr_ann(pres_ann_file)

        pres_image = draw_boxes_with_label(pres_img, pres_ann[0], pres_ann[1])

        for pill in pill_list:
            pill_name = pill.split('.')[0]

            pill_img_file = f"{image_folder}/{pill_name}.jpg"
            pill_ann_file = f"{ann_folder}/{pill_name}.json"

            pill_img = cv2.imread(pill_img_file)
            pill_ann = read_box_ann(pill_ann_file)

            pill_image = draw_boxes_with_label(pill_img, pill_ann[0], pill_ann[1])

            plt.subplot(1, 2, 1)
            plt.imshow(pres_image[:, :, ::-1])
            plt.subplot(1, 2, 2)
            plt.imshow(pill_image[:, :, ::-1])
            plt.show()
