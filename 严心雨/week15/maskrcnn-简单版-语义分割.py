import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import numpy as np
import cv2
from PIL import Image

model = maskrcnn_resnet50_fpn(pretrained = True)
model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# 图片预处理
def preprocess_image(image):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor()]
    )
    return transform(image).unsqueeze(0)

# 推理
def infer(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_tensor = preprocess_image(image)
    image = image_tensor.to(device)
    with torch.no_grad():
        prediction = model(image)
    return prediction

# 画图
def show_result(image_path,prediction):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    color_mapping = {
        1: (255, 0, 0),  # 人用蓝色表示
        2: (0, 255, 0),  # 自行车用绿色表示
        3: (0, 0, 255)  # 汽车用红色表示
    }
    for pred in prediction:
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for i,(mask,label,score) in enumerate(zip(masks,labels,scores)):
            if score > 0.5:
                mask = mask[0]
                # 将掩码二值化
                """
                mask > 0.5：逐元素比较 mask 中的值是否大于 0.5。结果是一个布尔数组，形状与 mask 相同。
                .astype(np.uint8)：将布尔数组转换为无符号8位整数数组。布尔值 True 转换为 1，False 转换为 0
                """
                mask = (mask > 0.5).astype(np.uint8)
                contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                # 获取对应的颜色
                color = color_mapping.get(label.item(),(255, 255, 255))
                # 绘制所有轮廓
                image_result = cv2.drawContours(image,contours,-1,color,2)
    cv2.imshow('result',image_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    image_path = 'E:/YAN/HelloWorld/homeworks/maskrcnn/street.jpg'
    prediction = infer(image_path)
    image_result = show_result(image_path,prediction)
