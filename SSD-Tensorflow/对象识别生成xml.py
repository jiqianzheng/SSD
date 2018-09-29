import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import xml.dom.minidom

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods, ssd_vgg_512
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

net_shape = (512, 512)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))
# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
# ssd_net = ssd_vgg_300.SSDNet()
ssd_net = ssd_vgg_512.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Restore SSD model.
# ckpt_filename = '../checkpoints/ssd_300_vgg.ckpt'
ckpt_filename = './checkpoints/model.ckpt-91516'
print(ckpt_filename)
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


def process_image(img, select_threshold=0.5, nms_threshold=.45, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=21, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes


VOC_LABELS = {
    'no'
    'ne': (0, 'Background'),
    'chengguan': (1, "ChengGuan"),
    'fangwu': (2, "FangWu"),
    'fantou': (3, "FanTou"),
    'qiao': (4, "Qiao"),
    'shanpo': (5, "ShanPo"),
    'shantou': (6, "ShanTou-h"),

    'shu': (7, "Shu-cy"),
    'stzh': (8, "STZH-c"),
    'tikuan': (9, "TiKuan-qc"),
    'yinzhang': (10, "YinZhang-qc"),
}

labels=list(VOC_LABELS.keys())






load_path="/Users/qianzheng/Downloads/fd1/"
save_path_name="/Users/qianzheng/Downloads/fd4/"
pathDir=os.listdir(load_path)
j=1
for pating in pathDir:
    if pating !=".DS_Store":
        pp = pating[:-4]
        print(pp)
        save_path = save_path_name + pp
        isExits = os.path.exists(save_path)
        if not isExits:
            os.makedirs(save_path)

        p = "" + load_path + pating
        print(p)
        image = Image.open(p)

        img = mpimg.imread(p)

        rclasses, rscores, rbboxes = process_image(img)
        height = img.shape[0]
        width = img.shape[1]
        print(rclasses.shape[0])

        doc = xml.dom.minidom.Document()
        root = doc.createElement('annotation')
        doc.appendChild(root)
        # 添加长宽
        fileName = doc.createElement('filename')
        fileName.appendChild(doc.createTextNode(pating))
        root.appendChild(fileName)
        # 添加size标签以及子标签
        size = doc.createElement('size')
        nodeName = doc.createElement('width')
        nodeName.appendChild(doc.createTextNode(str(width)))
        size.appendChild(nodeName)

        nodeName = doc.createElement('height')
        nodeName.appendChild(doc.createTextNode(str(height)))
        size.appendChild(nodeName)

        nodeName = doc.createElement('depth')
        nodeName.appendChild(doc.createTextNode('3'))
        size.appendChild(nodeName)
        root.appendChild(size)

        #看有多少行
        for i in range(rclasses.shape[0]):
            cls_id = int(rclasses[i])
            if cls_id >= 0:
                score = rscores[i]
                ymin = int(rbboxes[i, 0] * height)
                xmin = int(rbboxes[i, 1] * width)
                ymax = int(rbboxes[i, 2] * height)
                xmax = int(rbboxes[i, 3] * width)
                # print(xmin,ymin,xmax-xmin,ymax-ymin)
                class_name = labels[cls_id]
                region = (int(xmin), int(ymin), int(xmax), int(ymax))
                cropImg = image.crop(region)
                screeshot_path = save_path +"/" +class_name + str(j) + ".jpg"

                # 添加一个object
                object = doc.createElement('object')

                nodeName = doc.createElement('name')
                nodeName.appendChild(doc.createTextNode(class_name+str(j)))
                object.appendChild(nodeName)

                nodeName = doc.createElement('pose')
                nodeName.appendChild(doc.createTextNode('Unspecified'))
                object.appendChild(nodeName)

                nodeName = doc.createElement('truncated')
                nodeName.appendChild(doc.createTextNode('0'))
                object.appendChild(nodeName)

                nodeName = doc.createElement('difficult')
                nodeName.appendChild(doc.createTextNode('0'))
                object.appendChild(nodeName)

                bndBox = doc.createElement('bndbox')

                nodeName = doc.createElement('xmin')
                nodeName.appendChild(doc.createTextNode(str(xmin)))
                bndBox.appendChild(nodeName)

                nodeName = doc.createElement('ymin')
                nodeName.appendChild(doc.createTextNode(str(ymin)))
                bndBox.appendChild(nodeName)
                #
                nodeName = doc.createElement('xmax')
                nodeName.appendChild(doc.createTextNode(str(xmax)))
                bndBox.appendChild(nodeName)

                nodeName = doc.createElement('ymax')
                nodeName.appendChild(doc.createTextNode(str(ymax)))
                bndBox.appendChild(nodeName)
                object.appendChild(bndBox)
                root.appendChild(object)

                isExits = os.path.exists(screeshot_path)
                if not isExits:
                    cropImg.save(screeshot_path)
                    j = j + 1
        image.save(str(save_path+ '/'+pating[:-4]+ '.jpg'))
        fp = open(save_path+ '/'+pating[:-4] + '.xml', 'w', encoding="utf-8")
        doc.writexml(fp, indent='\t', addindent='\t', newl='\n', encoding="utf-8")

        print("********")
print("ok!")