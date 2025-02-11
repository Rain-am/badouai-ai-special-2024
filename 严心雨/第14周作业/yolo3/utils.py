import tensorflow as tf
import numpy as np
from PIL import Image

def load_weights(var_list,weights_file):
    with open(weights_file,'rb') as fp:
        _ = np.fromfile(fp,dtype=np.int32,count=5)
        weights = np.fromfile(fp,dtype=np.float32)

    # 从权重数组中加载权重并分配给模型中的变量
    ptr = 0
    i = 0
    assign_op = []

    while i < len(var_list)-1:
        var1 = var_list[i]
        var2 = var_list[i+1]
        if 'conv2d' in var1.name.split('/')[-2]:
            if 'batch_normalization' in var2.name.split('/')[-2]:
                gamma,beta,mean,var = var_list[i+1:i+5]
                batch_norm_vars = [beta, gamma, mean, var] # 注意顺序
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr+num_params].reshape(shape)
                    ptr += num_params
                    assign_op.append(tf.assign(var,var_weights,validate_shape=True))
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                # 卷积层的偏置参数。
                bias = var2
                bias_shape = bias.shape.as_list()
                bias_params = np.prod(bias_shape)
                bias_weights = weights[ptr:ptr+bias_params].reshape(bias_shape)
                ptr += bias_params
                assign_op.append(tf.assign(bias,bias_weights,validate_shape=True))
                i += 1

        shape = var1.shape.as_list()
        num_params = np.prod(shape)
        var_weights = weights[ptr:ptr + num_params].reshape((shape[3],shape[2],shape[0], shape[1]))
        var_weights = np.transpose(var_weights,(2,3,1,0))
        ptr += num_params
        assign_op.append(tf.assign(var1, var_weights, validate_shape=True))
        i += 1
    return assign_op

# 对预测输入图像进行缩放，按照长宽比进行缩放，不足的地方进行填充
def letterbox_image(image,size):
    image_w,image_h = image.size
    w,h = size
    new_w = int(image_w * min(w*1.0/image_w,h*1.0/image_h))
    new_h = int(image_h * min(w*1.0/image_w,h*1.0/image_h))
    resize_image = image.resize((new_w,new_h),Image.BICUBIC)
    boxed_image = Image.new('RGB',size,(128,128,128))
    boxed_image.paste(resize_image,((w-new_w)//2,(h-new_h)//2))
    return boxed_image


