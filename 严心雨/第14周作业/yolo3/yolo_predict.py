import tensorflow as tf
import numpy as np
import os
import colorsys
import random
from yolo3_model import yolo
import config

class yolo_predictor:
    def __init__(self,obj_threshold,nms_threshold,classes_file,anchors_file):
        self.obj_threshold=obj_threshold
        self.nms_threshold=nms_threshold
        # 预读取
        self.classes_path=classes_file
        self.anchors_path=anchors_file
        # 读取种类名称
        self.class_names=self._get_class()
        # 读取先验框
        self.anchors=self._get_anchors()

        #画框框用
        hsv_tuples = [(x/len(self.class_names),1.,1.) for x in range(len(self.class_names))]
        # 为不同的类别配置不同的颜色
        # hsv转rgb
        self.colors = list(map(lambda x:colorsys.hsv_to_rgb(*x),hsv_tuples))
        # 将 RGB 分量从 [0, 1] 转换为 [0, 255]
        self.colors = list(map(lambda x:(int(x[0]*255),int(x[1]*255),int(x[2]*255)),self.colors))

        """
        在目标检测中，对颜色列表 self.colors 进行随机打乱是一个常见的做法，主要是为了增加颜色分配的随机性和多样性。
        这样可以确保生成的颜色在视觉上更加分散和易于区分，从而提高可视化效果。
        """
        random.seed(10101)
        random.shuffle(self.colors)
        random.seed(None)

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as pf:
            class_names = pf.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as pf:
            anchors = pf.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1,2)
        return anchors


    #---------------------------------------#
    #   对三个特征层解码
    #   进行排序并进行非极大抑制
    #---------------------------------------#


    # 其实是解码的过程
    # 解析模型的输出特征图，并将其转换为可用于后续处理的格式
    # 将特征图张量 feats 转换为包含边界框预测、置信度和类别概率的张量
    def _get_feats(self,feats,anchors,num_classes,input_shape):
        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors,dtype=tf.float32),[1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        # 重塑模型的输出张量
        predictions = tf.reshape(feats,[-1,grid_size[0],grid_size[1],num_anchors,num_classes+5])
        # 构建13*13*1*2的矩阵
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]),[-1,1,1,1]),[1,grid_size[1],1,1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x,grid_y],axis=-1)
        grid = tf.cast(grid,tf.float32)
        # box_xy 归一化
        box_xy = (tf.sigmoid(predictions[...,:2])+grid)/tf.cast(grid_size[::-1],tf.float32)
        # box_wh 归一化
        box_wh = (tf.exp(predictions[...,2:4])*anchors_tensor)/tf.cast(input_shape[::-1],tf.float32)
        # 预测的边界框的置信度
        box_confidence = tf.sigmoid(predictions[...,4:5])
        # 预测的边界框的类别概率
        box_class_probs = tf.sigmoid(predictions[...,5:])
        return box_xy,box_wh,box_confidence,box_class_probs

    # 获得在原图上框的位置
    def correct_boxes(self,box_xy,box_wh,input_shape,image_shape):
        box_yx = box_xy[...,::-1]
        box_hw = box_wh[...,::-1]
        input_shape = tf.cast(input_shape,dtype =tf.float32)
        image_shape = tf.cast(image_shape,dtype =tf.float32)
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape/image_shape))
        offset = (input_shape-new_shape)/2./input_shape
        scale = input_shape / new_shape
        box_yx = (box_yx-offset) * scale
        box_hw *= scale
        box_mins = box_yx - (box_hw / 2.)
        box_maxs = box_yx + (box_hw / 2.)
        boxes = tf.concat([box_mins[...,0:1],box_mins[...,1:2],box_maxs[...,0:1],box_maxs[...,1:2]],axis=-1)
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        return boxes

    # 将预测出的box坐标转换为对应原图的坐标，然后计算每个box的分数
    def boxes_and_scores(self,feats,anchors,classes_num,input_shape,image_shape):
        box_xy, box_wh, box_confidence, box_class_probs = self._get_feats(feats,anchors,classes_num,input_shape)
        boxes = self.correct_boxes(box_xy,box_wh,input_shape,image_shape)
        boxes = tf.reshape(boxes,[-1,4])
        box_scores = box_confidence * box_class_probs
        box_scores = tf.reshape(box_scores,[-1,classes_num])
        return boxes,box_scores

    # 根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
    def eval(self,yolo_inputs,image_shape,max_boxes = 20):
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        boxes = []
        box_scores = []
        input_shape = tf.shape(yolo_inputs[0])[1:3] * 32
        # 对三个特征层的输出获取每个预测box坐标和box的分数，score = 置信度x类别概率
        # ---------------------------------------#
        #   对三个特征层解码
        #   获得分数和框的位置
        # ---------------------------------------#
        for i in range(len(yolo_inputs)):
            _boxes, _box_scores = self.boxes_and_scores(yolo_inputs[i],self.anchors[anchor_mask[i]],len(self.class_names),input_shape,image_shape)
            boxes.append(_boxes)
            box_scores.append(_box_scores)
        boxes = tf.concat(boxes,axis=0)
        box_scores = tf.concat(box_scores,axis=0)

        mask = box_scores >= self.obj_threshold
        max_boxes_tensor = tf.constant(max_boxes,dtype=tf.int32)
        boxes_ = []
        scores_ = []
        classes_ = []

        # ---------------------------------------#
        #   1、取出每一类得分大于self.obj_threshold
        #   的框和得分
        #   2、对得分进行非极大抑制
        # ---------------------------------------#
        # 对每一个类进行判断
        for c in range(len(self.class_names)):
            class_boxes = tf.boolean_mask(boxes,mask[:,c])
            class_box_scores = tf.boolean_mask(box_scores[:,c],mask[:,c])
            nms_idx = tf.image.non_max_suppression(class_boxes,class_box_scores,max_boxes_tensor,iou_threshold=self.nms_threshold)
            class_boxes = tf.gather(class_boxes,nms_idx)
            class_box_scores = tf.gather(class_box_scores,nms_idx)
            classes = tf.ones_like(class_box_scores,'int32')*c
            boxes_.append(class_boxes)
            scores_.append(class_box_scores)
            classes_.append(classes)

        boxes_ = tf.concat(boxes_,axis=0)
        scores_ = tf.concat(scores_,axis=0)
        classes_ = tf.concat(classes_,axis=0)

        return boxes_,scores_,classes_

    def predict(self,inputs,image_shape):
        model = yolo(config.norm_epsilon,config.norm_decay,self.anchors_path,self.classes_path,pre_train=False)
        output = model.yolo_inference(inputs,config.num_anchors // 3,config.num_classes,training=False)
        boxes,scores,classes = self.eval(output,image_shape,max_boxes = 20)
        return boxes,scores,classes






