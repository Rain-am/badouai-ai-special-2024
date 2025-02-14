import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import cv2
import numpy as np
from PIL import Image

model = maskrcnn_resnet50_fpn(pretrained = True)
model.eval()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# 图片预处理
def preprocess_img(image):
    transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),]
    )
    return transform(image).unsqueeze(0)

# 进行推理
def infer(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image_tensor = preprocess_img(image)
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        prediction = model(image_tensor)
    return prediction

# 画图
def show_result(image_path,prediction):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    instance_colors = {}
    for pred in prediction:
        masks = pred['masks'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        for i,(mask,label,score) in enumerate(zip(masks,labels,scores)):
            if score > 0.5:
                mask = mask[0]
                mask = (mask>0.5).astype(np.uint8)
                if i not in instance_colors:
                    instance_colors[i] = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
                color = instance_colors[i]
                contours , _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)
    cv2.imshow('result',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ =='__main__':
    image_path = 'E:/YAN/HelloWorld/homeworks/maskrcnn/street.jpg'
    prediction = infer(image_path)
    image_result = show_result(image_path,prediction)
