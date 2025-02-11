from keras.layers import Input,Conv2D,MaxPool2D,Dense,Flatten, Permute
from keras.layers.advanced_activations import PReLU
from keras.models import Model
import utils
import cv2
import numpy as np

#-----------------------------#
#   粗略获取人脸框
#   输出bbox位置和是否有人脸
#-----------------------------#
def create_Pnet(weight_path):
    input = Input(shape=[None,None,3])
    # 1 Conv
    # 12*12*3→10*10*10
    x = Conv2D(10,(3,3),strides=(1, 1),padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='PReLU1')(x)
    # 10*10*10→5*5*10
    x = MaxPool2D(pool_size=(2, 2))(x)

    # 2 Conv
    # 5*5*10→3*3*16
    x = Conv2D(16,(3,3),strides=(1, 1),padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='PReLU2')(x)

    # 3 Conv
    # 3*3*16→1*1*32
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='valid', name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='PReLU3')(x)

    # 是否有人脸的二分类
    # 1*1*32→1*1*2
    classifier = Conv2D(2, (1, 1), strides=(1, 1),activation='softmax', name='conv4-1')(x)
    # 边界框位置
    # 1*1*32→1*1*4
    bbox_regress = Conv2D(4, (1, 1), strides=(1, 1),name='conv4-2')(x)

    model = Model([input],[classifier,bbox_regress])
    model.load_weights(weight_path,by_name=True)

    return model
#-----------------------------#
#   mtcnn的第二段
#   精修框
#-----------------------------#
def create_Rnet(weight_path):
    input = Input(shape=[24,24,3])
    # 1 Conv
    # 24*24*3→22*22*28
    x = Conv2D(28,(3,3),strides=(1,1),padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    # 22*22*28→11*11*28
    x = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

    # 2 Conv
    # 11*11*28→9*9*48
    x = Conv2D(48,(3,3),strides=(1,1),padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1,2],name='prelu2')(x)
    # 9*9*48→4*4*48
    x = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='valid')(x)

    # 3 Conv
    # 4*4*48→3*3*64
    x = Conv2D(64,(2,2),strides=(1,1),padding='valid',name='conv3')(x)
    x = PReLU(shared_axes=[1,2],name='prelu3')(x)

    # FC
    # 3*3*64→(576,)
    """
    如果模型权重是基于特定的输入形状训练的，那么在加载权重时，输入形状必须与训练时一致。如果训练时使用了 Permute((3, 2, 1))(x)，那么在加载权重时也必须使用相同的操作。
    如果自己训练的话，可以不要这个操作
    """
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    x = Dense(128,name='conv4')(x)
    x = PReLU(name='prelu4')(x)

    # 是否有人脸的二分类
    classifier = Dense(2,activation='softmax',name='conv5-1')(x)
    # 边界框位置
    bbox_regress = Dense(4,name='conv5-2')(x)

    model = Model([input],[classifier,bbox_regress])
    model.load_weights(weight_path,by_name=True)
    return model

#-----------------------------#
#   mtcnn的第三段
#   精修框并获得五个点
#-----------------------------#
def create_Onet(weight_path):
    input = Input(shape=[48,48,3])

    # 1 Conv1
    # 48*48*3→46*46*32
    x = Conv2D(32,(3,3),strides=(1,1),padding='valid',name='conv1')(input)
    x = PReLU(shared_axes=[1,2],name='prelu1')(x)
    # 46*46*32→23*23*32
    x = MaxPool2D(pool_size=(3,3),strides=(2,2),padding='same')(x)

    # 2 Conv2
    # 23*23*32→21*21*64
    x = Conv2D(64,(3,3),strides=(1,1),padding='valid',name='conv2')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu2')(x)
    # 21*21*64→10*10*64
    x = MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(x)

    # 3 Conv3
    # 10*10*64→8*8*64
    x = Conv2D(64,(3,3),strides=(1,1),padding='valid',name='conv3')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu3')(x)
    # 8*8*64→4*4*64
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

    # 4 Conv4
    # 4*4*64→3*3*128
    x = Conv2D(128, (2, 2), strides=(1, 1), padding='valid', name='conv4')(x)
    x = PReLU(shared_axes=[1, 2], name='prelu4')(x)

    # FC
    x = Permute((3, 2, 1))(x)
    x = Flatten()(x)
    x = Dense(256,name='conv5')(x)
    x = PReLU(name='prelu5')(x)

    # 是否有人脸的二分类
    classifier = Dense(2,activation='softmax',name='conv6-1')(x)
    # 边界框位置
    bbox_regress = Dense(4,name='conv6-2')(x)
    # 人脸特征点定位：5个人脸特征点的xy坐标，10个输出
    landmark_regress = Dense(10,name='conv6-3')(x)

    model = Model([input],[classifier,bbox_regress,landmark_regress])
    model.load_weights(weight_path,by_name=True)
    return model

class mtcnn():
    def __init__(self):
        self.Pnet = create_Pnet('model_data/pnet.h5')
        self.Rnet = create_Rnet('model_data/rnet.h5')
        self.Onet = create_Onet('model_data/onet.h5')
    def detectFace(self,img,threshold):
        #-----------------------------#
        #   归一化，加快收敛速度
        #   把[0,255]映射到(-1,1)
        #-----------------------------#
        copy_img = (img.copy()-127.5)/127.5
        origin_h,origin_w,_ = copy_img.shape
        #-----------------------------#
        #   计算图像金字塔的缩放比例
        #-----------------------------#
        scales = utils.calculateScales(img)
        #-----------------------------#
        #    PNet 阶段：粗略计算人脸框
        #-----------------------------#
        # 对每个缩放后的图像进行 PNet 检测，生成初步的候选框
        out = []
        for scale in scales:
            hs = int(origin_h * scale)
            ws = int(origin_w * scale)
            # 在OpenCV中，cv2.resize的尺寸参数通常是（宽度，高度），也就是（width, height）。而图像的shape属性返回的是（高度，宽度，通道数）
            inputs = cv2.resize(copy_img,(ws, hs))
            inputs = inputs.reshape(1,*inputs.shape)
            outputs = self.Pnet.predict(inputs)
            out.append(outputs)

        # 提取候选框并进行第一次 NMS
        rectangles = []
        for i in range(len(scales)):
            cls_prob = out[i][0][0][:,:,1]
            roi = out[i][1][0]
            # PNet 是全卷积网络，输出特征图的尺寸与输入图像的缩放比例严格对应。
            out_h,out_w = cls_prob.shape
            outside = max(out_h,out_w)
            rectangle = utils.detect_face_12net(cls_prob, roi, outside, 1/scales[i], origin_w, origin_h, threshold[0])
            rectangles.extend(rectangle)

        rectangles = utils.NMS(rectangles,0.7)
        if len(rectangles) == 0:
            return rectangles
        # -----------------------------#
        #   稍微精确计算人脸框
        #   Rnet部分
        # -----------------------------#
        """
        1. 裁剪并缩放候选框：从原始图像中裁剪出每个候选框，并将其缩放到 24x24。
        2. 将裁剪后的图像转换为 NumPy 数组：将裁剪后的图像列表转换为一个 NumPy 数组。
        3. 使用 RNet 模型进行预测：将裁剪后的图像输入到 RNet 模型中，获取预测结果。
        4. 提取分类概率和边界框回归结果：从 RNet 的输出中提取分类概率和边界框回归结果。
        5. 筛选候选框：使用 RNet 的输出进一步筛选候选框，调整边界框的位置，并去除低概率的候选框。
        6. 检查是否有候选框：如果筛选后的候选框为空，直接返回空列表。
        """
        predict_24_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2])]
            inputs = cv2.resize(crop_img,(24,24))
            predict_24_batch.append(inputs)
        predict_24_batch = np.array(predict_24_batch)
        out = self.Rnet.predict(predict_24_batch)
        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi = out[1]
        roi = np.array(roi)
        rectangles = utils.filter_face_24net(cls_prob,roi,rectangles,origin_w,origin_h,threshold[1])
        if len(rectangles) == 0:
            return rectangles
        #-----------------------------#
        #   计算人脸框
        #   onet部分
        #-----------------------------#
        predict_batch = []
        for rectangle in rectangles:
            crop_img = copy_img[int(rectangle[1]):int(rectangle[3]),int(rectangle[0]):int(rectangle[2])]
            inputs = cv2.resize(crop_img,(48,48))
            predict_batch.append(inputs)
        predict_batch = np.array(predict_batch)
        out = self.Onet.predict(predict_batch)
        cls_prob = out[0]
        cls_prob = np.array(cls_prob)
        roi = out[1]
        roi = np.array(roi)
        pts = out[2]
        pts = np.array(pts)
        rectangles = utils.filter_face_48net(cls_prob,roi,pts,rectangles,origin_w,origin_h,threshold[2])

        return rectangles





