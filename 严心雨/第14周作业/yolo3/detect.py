import os
import config
import tensorflow as tf
import numpy as np
from yolo_predict import yolo_predictor
from utils import letterbox_image,load_weights
from PIL import Image,ImageFont,ImageDraw

# 指定使用GPU的Index
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_index

# 加载模型，进行预测
def detect(image_path,model_path,yolo_weights=None):
    # ---------------------------------------#
    #   图片预处理
    # ---------------------------------------#
    image = Image.open(image_path)
    resized_image = letterbox_image(image,(416,416))
    image_data = np.array(resized_image,dtype=np.float32)
    image_data /= 255.
    image_data = np.expand_dims(image_data,axis=0)

    # ---------------------------------------#
    #   图片输入
    # ---------------------------------------#
    input_image_shape = tf.placeholder(dtype=tf.int32,shape=(2,))
    input_image = tf.placeholder(shape=[None,416,416,3],dtype=tf.float32)

    # ---------------------------------------#
    #   创建预测器
    # ---------------------------------------#
    predictor = yolo_predictor(config.obj_threshold,config.nms_threshold,config.classes_path,config.anchors_path)

    # ---------------------------------------#
    #   加载模型和权重
    # ---------------------------------------#
    with tf.Session() as sess:
        if yolo_weights is not None:
            with tf.variable_scope('predict'):
                boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            # 载入模型
            load_op = load_weights(tf.global_variables(scope='predict'),weights_file =yolo_weights)
            sess.run(load_op)
            # 进行预测
            out_boxes, out_scores, out_classes= sess.run(
                 [boxes, scores, classes],
                 feed_dict={
                        input_image:image_data,
                        input_image_shape:[image.size[1],image.size[0]]
                    })
        else:
            boxes, scores, classes = predictor.predict(input_image, input_image_shape)
            saver = tf.train.Saver()
            saver.restore(sess,model_path)
            # 进行预测
            out_boxes, out_scores, out_classes = sess.run(
                [boxes, scores, classes],
                feed_dict={
                    input_image: image_data,
                    input_image_shape: [image.size[1], image.size[0]]
                })
        #---------------------------------------#
        #   画框
        #---------------------------------------#
        print('Found {} boxes for {}'.format(len(out_boxes),'img'))
        # 字体及字体大小
        font = ImageFont.truetype(font='E:/YAN/HelloWorld/homeworks/yolo3-tensorflow-master/font/FiraMono-Medium.otf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        # 厚度
        thickness = (image.size[0]+image.size[1])//300

        for i,c in reversed(list(enumerate(out_classes))):
            predicted_class = predictor.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{}{:.2f}'.format(predicted_class,score)
            # 用于画框框和文字
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label,font)
            # 获得左上角和右下角y,x
            top,left,bottom,right= box

            top = max(0,np.floor(top+0.5).astype('int32'))
            left = max(0,np.floor(left+0.5).astype('int32'))
            bottom = min(image.size[1]-1,np.floor(bottom+0.5).astype('int32'))
            right = min(image.size[0]-1,np.floor(right+0.5).astype('int32'))
            print(label,(left,top),(right,bottom))
            print(label_size)

            # 确保文本标签不会超出图像的上边界
            if top - label_size[1] >= 0:
                text_origin = np.array([left,top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 通过循环 for i in range(thickness)，绘制多层边界框，以实现线条厚度的效果
            for i in range(thickness):
                draw.rectangle([left+i,top+i,right-i,bottom-i],outline=predictor.colors[c])

            # 绘制文本框
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill = predictor.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        image.show()
        image.save('E:/YAN/HelloWorld/homeworks/yolo3-tensorflow-master/img/result1.jpg')

if __name__ == '__main__':
    # 当使用yolo3自带的weights的时候
    if config.pre_train_yolo3 == True:
        detect(config.image_file, config.model_dir, config.yolo3_weights_path)
    # 当使用自训练模型的时候
    else:
        detect(config.image_file, config.model_dir)




