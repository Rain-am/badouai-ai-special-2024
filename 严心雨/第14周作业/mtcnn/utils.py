import numpy as np
from PIL import Image
import cv2

#-----------------------------#
#   构建原图金字塔
#   计算原始输入图像
#   每一次缩放的比例
#-----------------------------#
def calculateScales(img):
    copy_img = img.copy()
    h,w,_ = copy_img.shape
    pr_scale = 1.0
    if min(w,h)>500:
        pr_scale = 500.0/min(h,w)
        h = int(h * pr_scale)
        w = int(w * pr_scale)
    elif max(w,h)<500:
        pr_scale = 500.0 / max(h, w)
        h = int(h * pr_scale)
        w = int(w * pr_scale)

    scales = []
    factor = 0.709
    factor_count = 0
    minl = min (h,w)

    while minl>=12:
        scales.append(pr_scale * pow(factor,factor_count))
        minl *=  factor
        factor_count += 1
    return  scales

#-----------------------------#
#   将长方形调整为正方形
#-----------------------------#
def rect2sqaure(rectangles):
    w = rectangles[:,2]- rectangles[:,0] # x2-x1
    h = rectangles[:,3]- rectangles[:,1] # y2-y1
    l = np.maximum(w,h).T
    # 左上角
    rectangles[:, 0] = rectangles[:, 0] + w * 0.5 - l * 0.5
    rectangles[:, 1] = rectangles[:, 1] + h * 0.5 - l * 0.5
    # 右下角
    rectangles[:, 2:4] = rectangles[:, 0:2] + np.repeat([l],2,axis=0).T
    return rectangles

#-------------------------------------#
#   非极大抑制
#-------------------------------------#
def NMS(rectangles,threshold):
    """
        这段代码实现了一个标准的非极大值抑制（NMS）算法，用于去除冗余的边界框，保留最有可能包含目标的边界框。
        主要步骤包括：
        1. 按分类概率排序。
        2. 逐个处理最高概率的边界框，计算其与其他边界框的重叠度（IOU）。
        3. 保留重叠度小于阈值的边界框。
        4. 返回最终保留的边界框
        """
    if len(rectangles)==0:
        return rectangles
    boxes = np.array(rectangles)
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    s = boxes[:,4]
    area = np.multiply(x2-x1+1,y2-y1+1)

    pick = []
    I = np.array(s.argsort())
    while len(I)>0:
        xx1 = np.maximum(x1[I[-1]],x1[I[0:-1]])
        yy1 = np.maximum(y1[I[-1]],y1[I[0:-1]])
        xx2 = np.minimum(x2[I[-1]], x2[I[0:-1]])
        yy2 = np.minimum(y2[I[-1]], y2[I[0:-1]])
        w = np.maximum(0.0,xx2-xx1+1)
        h = np.maximum(0.0,yy2-yy1+1)
        inter = w * h
        iou = inter / (area[I[-1]]+area[I[0:-1]]-inter)
        pick.append(I[-1])
        I = I[np.where(iou<=threshold)[0]]
    result_rectangles = boxes[pick].tolist()
    return result_rectangles

#-------------------------------------#
#   对pnet处理后的结果进行处理
#-------------------------------------#
def detect_face_12net(cls_prob, roi, out_side, scale, width, height, threshold):
    """
    这个函数的主要步骤包括：
    1.轴交换：调整 cls_prob 和 roi 的形状。
    2.计算步长：确定特征图与原始图像的映射关系。
    3.找到满足阈值的坐标：提取可能的人脸区域。
    4.映射到原始图像坐标：将特征图上的坐标映射回原始图像。
    5.提取边界框回归偏移量：从 roi 中提取偏移量。
    6.调整边界框：根据偏移量调整边界框。
    7.合并边界框和分类概率：形成完整的矩形框数组。
    8.调整为正方形边界框：优化边界框形状。
    9.限制边界框范围：确保边界框在图像范围内。
    10.非极大值抑制：去除重叠的边界框，保留最佳人脸检测结果。
    """
    # 1.轴交换：调整 cls_prob 和 roi 的形状。
    cls_prob = np.swapaxes(cls_prob, 0, 1)
    roi = np.swapaxes(roi, 0, 2)

    # 2.计算步长：确定特征图与原始图像的映射关系。
    stride = 0
    if out_side != 1:
        stride = float(2 * out_side - 1) / (out_side - 1)
    else:
        stride = 2.0
    # 3.找到满足阈值的坐标：提取可能的人脸区域。
    (x, y) = np.where(cls_prob >= threshold)
    boundingbox = np.array([x, y]).T
    # 4.映射到原始图像坐标：将特征图上的坐标映射回原始图像。
    bb1 = np.fix((stride * boundingbox + 0) * scale)
    bb2 = np.fix((stride * boundingbox + 11) * scale)
    boundingbox = np.concatenate((bb1, bb2), axis=1)
    # 5.提取边界框回归偏移量：从 roi 中提取偏移量。
    dx1 = roi[0][x, y]
    dx2 = roi[1][x, y]
    dx3 = roi[2][x, y]
    dx4 = roi[3][x, y]

    score = np.array([cls_prob[x, y]]).T
    offset = np.array([dx1, dx2, dx3, dx4]).T
    # 6.调整边界框：根据偏移量调整边界框。
    boundingbox = boundingbox + offset * 12.0 * scale
    # 7.合并边界框和分类概率：形成完整的矩形框数组。
    rectangles = np.concatenate((boundingbox, score), axis=1)
    # 8.调整为正方形边界框：优化边界框形状。
    rectangles = rect2sqaure(rectangles)
    # 9.限制边界框范围：确保边界框在图像范围内。
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0     ,rectangles[i][0]))
        y1 = int(max(0     ,rectangles[i][1]))
        x2 = int(min(width ,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        score = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,score])
    return NMS(pick,0.3)

#-------------------------------------#
#   对Rnet处理后的结果进行处理
#-------------------------------------#
def filter_face_24net(cls_prob,roi,rectangles,width,height,threshold):
    """
    步骤
    1. 提取人脸概率
    2. 筛选满足阈值的样本
    3. 提取边界框坐标
    4. 提取满足条件的分类概率
    5. 提取边界框回归偏移量
    6. 调整边界框的位置
    7. 合并调整后的边界框和分类概率
    8. 调整为正方形边界框
    9. 限制边界框范围
    10. 非极大值抑制
    """
    # 1. 提取人脸概率
    prob = cls_prob[:,1]
    # 2. 筛选满足阈值的样本
    pick = np.where(prob>=threshold)

    rectangles = np.array(rectangles)
    # 3. 提取边界框坐标
    x1 = rectangles[pick,0]
    y1 = rectangles[pick,1]
    x2 = rectangles[pick,2]
    y2 = rectangles[pick,3]
    w = x2 - x1
    h = y2 - y1
    # 4. 提取满足条件的分类概率，并将其转换为列向量。
    sc = np.array([prob[pick]]).T
    # 5. 提取边界框回归偏移量
    dx1 = roi[pick,0]
    dy1 = roi[pick,1]
    dx2 = roi[pick,2]
    dy2 = roi[pick,3]
    # 6. 调整边界框的位置
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dy1 * h)[0]]).T
    x2 = np.array([(x2 + dx2 * w)[0]]).T
    y2 = np.array([(y2 + dy2 * h)[0]]).T
    # 7. 合并调整后的边界框和分类概率
    rectangles = np.concatenate((x1,y1,x2,y2,sc),axis=1)
    # 8. 调整为正方形边界框
    rectangles = rect2sqaure(rectangles)
    # 9. 限制边界框范围
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0, rectangles[i][0]))
        y1 = int(max(0, rectangles[i][1]))
        x2 = int(min(width, rectangles[i][2]))
        y2 = int(min(height, rectangles[i][3]))
        sc = rectangles[i][4]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc])
    # 10. 非极大值抑制
    return NMS(pick,0.3)

#-------------------------------------#
#   对onet处理后的结果进行处理
#-------------------------------------#
def filter_face_48net(cls_prob,roi,pts,rectangles,width,height,threshold):
    """
    1. 提取人脸概率
    2. 筛选满足阈值的样本
    3. 提取边界框坐标
    4. 提取满足条件的分类概率，并将其转换为列向量。
    5. 提取边界框回归偏移量
    6. 计算边界框的宽度和高度
    7. 提取人脸关键点回归结果
    8. 调整边界框的位置
    9. 合并调整后的边界框、分类概率和关键点
    10. 限制边界框范围
    11. 非极大值抑制
    """
    # 1. 提取人脸概率
    prob = cls_prob[:,1]
    # 2. 筛选满足阈值的样本
    pick = np.where(prob>=threshold)

    rectangles = np.array(rectangles)
    # 3. 提取边界框坐标
    x1 = rectangles[pick,0]
    y1 = rectangles[pick,1]
    x2 = rectangles[pick,2]
    y2 = rectangles[pick,3]
    # 4. 提取满足条件的分类概率，并将其转换为列向量。
    sc = np.array([prob[pick]]).T
    # 5. 提取边界框回归偏移量
    dx1 = roi[pick,0]
    dy1 = roi[pick,1]
    dx2 = roi[pick,2]
    dy2 = roi[pick,3]
    # 6. 计算边界框的宽度和高度
    w = x2 - x1
    h = y2 - y1
    # 7. 提取人脸关键点回归结果
    pts0 = np.array([(w * pts[pick,0] + x1)[0]]).T
    pts1 = np.array([(h * pts[pick,5] + y1)[0]]).T
    pts2 = np.array([(w * pts[pick,1] + x1)[0]]).T
    pts3 = np.array([(h * pts[pick,6] + y1)[0]]).T
    pts4 = np.array([(w * pts[pick,2] + x1)[0]]).T
    pts5 = np.array([(h * pts[pick,7] + y1)[0]]).T
    pts6 = np.array([(w * pts[pick,3] + x1)[0]]).T
    pts7 = np.array([(h * pts[pick,8] + y1)[0]]).T
    pts8 = np.array([(w * pts[pick,4] + x1)[0]]).T
    pts9 = np.array([(h * pts[pick,9] + y1)[0]]).T
    # 8. 调整边界框的位置
    x1 = np.array([(x1 + dx1 * w)[0]]).T
    y1 = np.array([(y1 + dy1 * h)[0]]).T
    x2 = np.array([(x2 + dx2 * w)[0]]).T
    y2 = np.array([(y2 + dy2 * h)[0]]).T

    # 9. 合并调整后的边界框、分类概率和关键点
    rectangles = np.concatenate((x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9),axis=1)
    # 10. 限制边界框范围
    pick = []
    for i in range(len(rectangles)):
        x1 = int(max(0,rectangles[i][0]))
        y1 = int(max(0,rectangles[i][1]))
        x2 = int(min(width,rectangles[i][2]))
        y2 = int(min(height,rectangles[i][3]))
        sc = rectangles[i][4]
        pts0 = rectangles[i][5]
        pts1 = rectangles[i][6]
        pts2 = rectangles[i][7]
        pts3 = rectangles[i][8]
        pts4 = rectangles[i][9]
        pts5 = rectangles[i][10]
        pts6 = rectangles[i][11]
        pts7 = rectangles[i][12]
        pts8 = rectangles[i][13]
        pts9 = rectangles[i][14]
        if x2>x1 and y2>y1:
            pick.append([x1,y1,x2,y2,sc,pts0,pts1,pts2,pts3,pts4,pts5,pts6,pts7,pts8,pts9])
    return NMS(pick,0.3)
