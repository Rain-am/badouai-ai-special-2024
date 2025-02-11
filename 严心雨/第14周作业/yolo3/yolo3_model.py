import numpy as np
import tensorflow as tf
import os

class yolo:
    def __init__(self,norm_epsilon,norm_decay,anchors_path,classes_path,pre_train):
        self.norm_epsilon = norm_epsilon
        self.norm_decay = norm_decay
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.pre_train = pre_train
        self.anchors = self._get_anchors()
        self.classes = self._get_classes()

    #---------------------------------------#
    #   获取种类和先验框
    #---------------------------------------#
    def _get_classes(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as pf:
            classes_names = pf.readlines()
        classes_names = [c.strip() for c in classes_names]
        return classes_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as pf:
            anchors = pf.readline()
        anchors = [float(c) for c in anchors.split(',')]
        # 一行表示一个候选框，每个候选框需要2个值，高度和宽度
        return np.array(anchors).reshape(-1,2)

    # ---------------------------------------#
    #   用于生成层
    # ---------------------------------------#
    # l2 归一化
    # 加速训练过程、提高模型的稳定性和性能。
    def _batch_normalization_layer(self,input_layer,name = None,training = True,norm_decay = 0.99, norm_epsilon = 1e-3):
        bn_layer = tf.layers.batch_normalization(inputs=input_layer,
                                                momentum=norm_decay,
                                                epsilon=norm_epsilon,
                                                center=True,
                                                scale=True,
                                                training=training,
                                                name=name
                                                )
        """
        tf.nn.leaky_relu 是 TensorFlow 中实现的 Leaky ReLU 激活函数。
        Leaky ReLU 是一种改进的 ReLU（Rectified Linear Unit）激活函数，它允许一部分负输入通过，从而避免了传统 ReLU 在负输入区域完全为零的问题。
        """
        return tf.nn.leaky_relu(bn_layer, alpha = 0.1)

    def _conv2d_layer(self,inputs,filters_num,kernel_size,name,use_bias=False,strides=1):
        conv = tf.layers.conv2d(inputs=inputs,
                                filters=filters_num,
                                kernel_size=kernel_size,
                                strides=[strides, strides],
                                kernel_initializer=tf.glorot_uniform_initializer(),
                                padding=('SAME' if strides==1 else 'VALID'),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=5e-4),
                                use_bias = use_bias,
                                name = name
                                )
        return conv

    # 残差卷积
    # 进行一次3X3的卷积，然后保存该卷积layer
    # 再进行一次1X1的卷积(通道数为输入通道数的一半)和一次3X3的卷积(通道数与输入通道数相同)，并把这个结果加上layer作为最后的结果
    # 用于Darknet-53残差块
    def _Residual_block(self,inputs,filters_num, blocks_num, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        inputs = tf.pad(inputs,paddings=[[0, 0], [1, 0], [1, 0], [0, 0]],mode='CONSTANT')
        layer = self._conv2d_layer(inputs,filters_num,kernel_size=3,strides=2,name='conv2d_'+str(conv_index))
        layer = self._batch_normalization_layer(layer,name='batch_normalization_'+str(conv_index),training = training,norm_decay = norm_decay, norm_epsilon = norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer,filters_num//2,kernel_size=1,strides=1,name='conv2d_'+str(conv_index))
            layer = self._batch_normalization_layer(layer,name='batch_normalization_'+str(conv_index),training = training,norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv_index += 1

            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1,name='conv2d_' + str(conv_index))
            layer = self._batch_normalization_layer(layer,name='batch_normalization_' + str(conv_index), training=training, norm_decay=norm_decay,norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer,conv_index

    # ---------------------------------------#
    #   生成_darknet53和逆卷积层
    # ---------------------------------------#
    def _darknet53(self, inputs, conv_index, training=True, norm_decay=0.99, norm_epsilon=1e-3):
        with tf.variable_scope('darknet53'):
            # 416,416,3 -> 416,416,32
            conv = self._conv2d_layer(inputs, filters_num=32, kernel_size=3, strides=1,
                                      name='conv2d_' + str(conv_index))
            conv = self._batch_normalization_layer(conv,name='batch_normalization_' + str(conv_index), training=training, norm_decay=norm_decay,
                                                  norm_epsilon=norm_epsilon
                                                  )
            conv_index += 1
            # 416,416,32 -> 208,208,64
            conv, conv_index = self._Residual_block(conv,conv_index=conv_index, filters_num=64, blocks_num=1,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # 208,208,64 -> 104,104,128
            conv, conv_index = self._Residual_block(conv,conv_index=conv_index, filters_num=128, blocks_num=2,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # 104,104,128 -> 52,52,256
            conv, conv_index = self._Residual_block(conv,conv_index=conv_index, filters_num=256, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # route1 = 52,52,256
            route1 = conv
            # 52,52,256 -> 26,26,512
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index,filters_num=512, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # route2 = 26,26,512
            route2 = conv
            # 26,26,512 -> 13,13,1024
            conv, conv_index = self._Residual_block(conv,conv_index=conv_index, filters_num=1024, blocks_num=4,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            # route3 = 13,13,1024
        return route1, route2, conv, conv_index

    # 输出两个网络结果
    # 第一个是进行5次卷积后，用于下一次逆卷积的，卷积过程是1X1，3X3，1X1，3X3，1X1
    # 第二个是进行5+2次卷积，作为一个特征层的，卷积过程是1X1，3X3，1X1，3X3，1X1，3X3，1X1
    def _yolo_block(self,inputs,filters_num, out_filters, conv_index, training = True, norm_decay = 0.99, norm_epsilon = 1e-3):
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1,name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1,name="conv2d_" + str(conv_index), use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    # 构建yolo模型结构
    # 返回三个特征层的内容
    def yolo_inference(self,inputs,num_anchors,num_classes,training=True):
        conv_index = 1
        # route1 = 52,52,256、route2 = 26,26,512、route3 = 13,13,1024
        conv2d_26, conv2d_43, conv, conv_index = self._darknet53(inputs, conv_index, training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
        # print("conv2d_43 shape:", conv2d_43.shape)
        with tf.variable_scope('yolo'):
            # --------------------------------------#
            #   获得第一个特征层
            # --------------------------------------#
            # conv2d_57 = 13,13,512，conv2d_59 = 13,13,255(3x(80+5))
            #  使用 512 个卷积核进行特征提取，而不是 1024（论文）。
            #  这是因为检测模块的目的是进一步处理特征图，而不是像骨干网络那样提取原始特征
            #  在计算复杂度和特征提取能力之间取得了良好的平衡
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, 512, num_anchors*(num_classes+5), conv_index=conv_index, training=training, norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            # print("conv2d_57 shape:", conv2d_57.shape)

            # --------------------------------------#
            #   获得第二个特征层
            # --------------------------------------#
            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num =256, kernel_size=1, strides=1,name='conv2d_'+str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_" + str(conv_index),training=training, norm_decay=self.norm_decay, norm_epsilon=self.norm_epsilon)
            # print("conv2d_60 shape:", conv2d_60.shape)
            conv_index += 1
            # 上采样→放大
            unSample_0 = tf.image.resize_nearest_neighbor(conv2d_60,[2*tf.shape(conv2d_60)[1],2*tf.shape(conv2d_60)[1]],name='unSample_0')
            # print("unSample_0 shape:", unSample_0.shape)
            route0 = tf.concat([unSample_0,conv2d_43],axis=-1,name='route_0')
            # conv2d_65 = 52,52,256，conv2d_67 = 26,26,255
            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, 256, num_anchors * (num_classes + 5),conv_index=conv_index, training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            # --------------------------------------#
            #   获得第三个特征层
            # --------------------------------------#
            conv2d_68 = self._conv2d_layer(conv2d_65, 128, kernel_size=1,strides=1, name='conv2d_' + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_" + str(conv_index),training=training, norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)
            conv_index += 1
            # 上采样→放大
            # unSample_1 = 52,52,128
            unSample_1 = tf.image.resize_nearest_neighbor(conv2d_68,[2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[1]],name='unSample_1')
            # route1= 52,52,384
            route1 = tf.concat([unSample_1, conv2d_26], axis=-1, name='route_1')
            # conv2d_75 = 52,52,255
            _, conv2d_75, _ = self._yolo_block(route1, 128, num_anchors * (num_classes + 5),conv_index=conv_index, training=training,norm_decay=self.norm_decay,norm_epsilon=self.norm_epsilon)

            return [conv2d_59,conv2d_67,conv2d_75]












