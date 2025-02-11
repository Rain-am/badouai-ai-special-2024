import cv2
from mtcnn import mtcnn

img = cv2.imread('img/timg.jpg')
draw = img.copy()
model = mtcnn()
threshold = [0.5,0.6,0.7]  # 三段网络的置信度阈值不同
rectangles = model.detectFace(img,threshold)


for rectangle in rectangles:
    if rectangle is not None:
        w = int(rectangle[2])-int(rectangle[0])
        h = int(rectangle[3])-int(rectangle[1])
        paddingh = 0.01 * w
        paddingw = 0.02 * h
        x1 = max(0,rectangle[0]-paddingw)
        y1 = max(0,rectangle[1]-paddingh) # 应该是将上边界向上移动
        x2 = min(0,rectangle[2]+paddingw)
        y2 = min(0,rectangle[3]+paddingh)
        if x2>x1 and y2>y1:
            crop_img = img[y1:y2,x1:x2]
            # 画人脸边界框
            cv2.rectangle(draw,(x1,y1),(x2,y2),(255,0,0),1)
        print("Rectangle:", rectangle)

        for i in range(5,15,2):
            # 画5个人脸关键点
            cv2.circle(draw, (int(rectangle[i+0]), int(rectangle[i + 1])), 2, (0, 255, 0))

cv2.imwrite('img/out.jpg',draw)
cv2.imshow('test/jpg',draw)
cv2.waitKey(0)
