# torch 是 PyTorch 的核心库，提供了深度学习的基础功能，适用于各种任务（包括计算机视觉、自然语言处理等）。
# torchvision 是 PyTorch 的扩展库，专注于计算机视觉任务，提供了预训练模型、数据集和图像处理工具，是对 torch 的补充。
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image,ImageDraw

# 加载预训练模型
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 设定要运行的硬件
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model.to(device) 是 PyTorch 中用于将模型移动到指定设备的常用操作。它确保模型的参数和计算可以在 CPU 或 GPU 上正确执行。
# 同时，输入数据也必须移动到相同的设备，否则会报错。
# tensor_or_model.to(device)。
# tensor_or_model：可以是一个张量（Tensor）或一个模型（torch.nn.Module）。
# device：目标设备，可以是一个字符串（如 "cpu" 或 "cuda"）或 torch.device 对象。
# 返回值是一个新的张量或模型，其数据被移动到了指定的设备。
model = model.to(device)

# 图像预处理
def preprocess_image(image):
    # torchvision.transforms.Compose 是 torchvision 中的一个非常实用的工具，用于将多个图像变换操作组合成一个完整的预处理管道。
    # 它允许你将一系列的图像处理步骤（如裁剪、缩放、归一化等）按顺序应用到图像上，从而简化代码并提高可读性。
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    # unsqueeze(0):在预处理后的张量前添加一个批次维度（batch dimension）.大多数深度学习模型（如卷积神经网络）期望输入是一个批次（batch）数据，即使处理单张图像，也需要将图像包装成一个批次。
    # unsqueeze(0) 的作用是在张量的第 0 维（最前面）插入一个大小为 1 的维度，从而将形状从 [C, H, W] 转换为 [1, C, H, W]。
    return transform(image).unsqueeze(0)

# 推理
def infer(image_path):
    # 因为Image.open()读取数据的操作，都是把数据读到CPU的单元上。但是如果我们的device是GPU,需要把数据再读到GPU上。
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess_image(image)
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        # prediction = model(image_tensor) 是深度学习中一个非常核心的操作，表示将输入张量 image_tensor 传递给模型 model，并获取模型的输出（预测结果）。
        # 这个操作通常被称为前向传播（forward pass），是模型推理（inference）或训练过程中的关键步骤。
        prediction = model(image_tensor)
    return prediction

# 结果展示
def show_result(image,prediction):
    # prediction[0]['boxes'] 提取第一个图像预测结果中的边界框张量。它的形状通常是 (N, 4)，表示有 N 个边界框（N个目标），每个边界框由 4 个坐标值表示。
    # .cpu() 将张量从 GPU（如果在 GPU 上）移动到 CPU。这是一个必要的步骤，因为 NumPy 不支持 GPU 张量。如果张量已经在 CPU 上，.cpu() 不会产生任何影响。
    # .numpy() 将 PyTorch 张量转换为 NumPy 数组，方便后续处理。
    boxes = prediction[0]['boxes'].cpu().numpy()
    labels = prediction[0]['labels'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    draw = ImageDraw.Draw(image)

    for box, label, score in zip(boxes,labels,scores):
        if score > 0.5:
            top_left = (box[0],box[1])
            bottom_right = (box[2],box[3])
            # outline='red'：矩形的边框颜色，设置为红色
            # width=2：矩形边框的宽度，这里设置为 2 像素
            draw.rectangle([top_left,bottom_right],outline='red',width=2)
            # draw.text()在给定位置绘制字符串;fill 用于文本的颜色
            draw.text([box[0],box[1]-10],str(label),fill='red')
    # image.show() 是 Python 中 Pillow（PIL）库的一个方法，用于快速显示图像。它是一个非常方便的工具，尤其适用于调试和快速查看图像处理的结果
    image.show()


# 用测试图片进行推理
if __name__ == '__main__':
    image_path = 'E:/YAN/HelloWorld/cv/【13】目标检测/代码/fasterrcnn简单版/street.jpg'
    prediction = infer(image_path)
    image = Image.open(image_path)
    image = show_result(image,prediction)
