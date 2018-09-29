#!/usr/bin/env python
#coding=utf-8

import pymysql

#创建数据库连接，注意这里我加入了charset和cursorclass参数
conn = pymysql.connect(
    host = "127.0.0.1",
    user = "root",
    password = "root",
    database = "shangbo_similar",
    charset = 'utf8',
    cursorclass = pymysql.cursors.DictCursor)

#获取游标
cursor = conn.cursor()
cursor.execute("SELECT * FROM Annotation_painting;")
#fetchall：获取所有的数据，默认以元祖的方式返回，如果你指定了cursorclass = pymysql.cursors.DictCursor，则以dict的方式返回
#result = cursor.fetchall()
#for row in result:
#    print("id是%d,name是%s，age是%d"%(row["id"],row["name"],row["age"]))
#fetchone：获取剩余数据的第一条数据
result = cursor.fetchone()
print(result)
#fetchmany:获取剩余数据的前2条数据
result = cursor.fetchmany(4)
for re in result:
    print(re)
cursor.close()
conn.close()

import os
import math
import random

import numpy as np
import tensorflow as tf
import cv2

slim = tf.contrib.slim
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods, ssd_vgg_512
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization
from PIL import Image


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
#原始模型
# ckpt_filename = './checkpoints/model.ckpt-91516'
#翻转90+270
# ckpt_filename = './checkpoints/model.ckpt-96098'
#翻转90
# ckpt_filename = './checkpoints/model.ckpt-94705'
#翻转+转置
# ckpt_filename = './checkpoints/model.ckpt-98901'
#不翻转+转置
# ckpt_filename = './checkpoints/model.ckpt-78658'
#自己训练数据
ckpt_filename = './checkpoints/model.ckpt-100433'
print(ckpt_filename)
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)

# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)


def process_image(img, select_threshold=0.5, nms_threshold=.2, net_shape=(300, 300)):
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



def endWith(s,*endstring):
    array=map(s.endswith,endstring)
    if True in array:
        return True
    else:
        return False

load_path="/Users/qianzheng/Downloads/fd/"
save_path_name="/Users/qianzheng/Downloads/fd3/"
pathDir=os.listdir(load_path)



VOC_LABELS = {

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


j = 1
for p in pathDir :
    if p!=".DS_Store":
        pp=load_path+p
        print(pp)
        img = mpimg.imread(pp)
        image = Image.open(pp)
        rclasses, rscores, rbboxes = process_image(img)
        # print(rbboxes)
        # visualization.plt_bboxes(img, rclasses, rscores, rbboxes)
        # plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]

        colors = dict()
        print(rclasses.shape[0])
        #看有多少行

        save_path = save_path_name + p[:-4]
        # isExits = os.path.exists(save_path)
        # if not isExits:
        #     os.makedirs(save_path)
        dpi = 150
        fig = plt.figure(figsize=(img.shape[1] / dpi, img.shape[0] / dpi), dpi=dpi)  # canvas
        plt.imshow(img)

        for i in range(rclasses.shape[0]):
            cls_id = int(rclasses[i])
            if cls_id >= 0:
                score = rscores[i]
                colors[cls_id] = (random.random(), random.random(), random.random())
                ymin = int(rbboxes[i, 0] * height)
                xmin = int(rbboxes[i, 1] * width)
                ymax = int(rbboxes[i, 2] * height)
                xmax = int(rbboxes[i, 3] * width)
                print(xmin,ymin,xmax-xmin,ymax-ymin)

                class_name = cls_id
                #裁剪部分
                # region = (int(xmin), int(ymin), int(xmax), int(ymax))
                # cropImg = image.crop(region)
                # screeshot_path = save_path+'/'  + labels[class_name] + str(j) + ".jpg"
                # isExits = os.path.exists(screeshot_path)
                # if not isExits:
                #     cropImg.save(screeshot_path)
                #     j = j + 1


                rect = plt.Rectangle((int(xmin), int(ymin)), (int(xmax) - int(xmin)),
                                     (int(ymax) - int(ymin)), fill=False,
                                     edgecolor=colors[cls_id],
                                     linewidth=3
                                     )
                plt.gca().add_patch(rect)

                plt.gca().text(xmin, ymin - 2,
                           '{:s} '.format(labels[class_name]),

                           fontsize=20, color='white')
        plt.axis('off')  # off axis
        # plt.show()
        # save_picture_name=save_path+'/'+p
        save_picture_name = save_path_name + p
        plt.savefig(save_picture_name)
        plt.clf()
        # plt.close()