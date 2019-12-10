- [车牌识别（一）-车牌定位](https://www.cnblogs.com/polly333/p/7367479.html)
- [车牌识别（一）——车牌定位(附详细代码及注释）](https://blog.csdn.net/lixiaoyu101/article/details/86626626)
- [车牌识别1：License Plate Detection and Recognition in Unconstrained Scenarios阅读笔记](https://www.cnblogs.com/greentomlee/p/10863363.html)
- [Github HyperLPR 高性能开源中文车牌识别框架](https://github.com/zeusees/HyperLPR)
- [Kaggle Car plates detection and recognition](https://www.kaggle.com/c/car-plates-detection-recognition/data)
```py
# cd workspace/datasets/Car_plate/
import cv2
img = cv2.imread('./test.png')
# plt.imshow(img)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
median = cv2.medianBlur(gaussian, 5)
plt.imshow(median, cmap='gray')

sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0,  ksize = 3)
plt.imshow(sobel, cmap='gray')

ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY)
plt.imshow(binary, cmap='gray')

# 膨胀和腐蚀操作的核函数
element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 6))
# 膨胀一次，让轮廓突出
dilation = cv2.dilate(binary, element2, iterations = 1)
# 腐蚀一次，去掉细节
erosion = cv2.erode(dilation, element1, iterations = 1)
# 再次膨胀，让轮廓明显一些
dilation2 = cv2.dilate(erosion, element2,iterations = 12)
plt.imshow(dilation2, cmap='gray')

def findPlateNumberRegion(img):
    region = []
    # 查找轮廓
    contours,hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选面积小的
    boxes = []
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        # if (area < 2000):
        #     continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)


        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boxes.append(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 车牌正常情况下长高比在2.7-5之间
        ratio =float(width) / float(height)
        print("rect = %s, ratio = %f" % (rect, ratio))
        if (ratio > 5 or ratio < 1.5):
            continue
        region.append(box)

    return region, boxes
region, boxes = findPlateNumberRegion(dilation2)
for box in region:
    cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

ys = [box[0, 1], box[1, 1], box[2, 1], box[3, 1]]
xs = [box[0, 0], box[1, 0], box[2, 0], box[3, 0]]
ys_sorted_index = np.argsort(ys)
xs_sorted_index = np.argsort(xs)

x1 = box[xs_sorted_index[0], 0]
x2 = box[xs_sorted_index[3], 0]

y1 = box[ys_sorted_index[0], 1]
y2 = box[ys_sorted_index[3], 1]

img_org2 = img.copy()
img_plate = img_org2[y1:y2, x1:x2]
plt.imshow(img_plate)
```
```py
from skimage.io import imread, imsave
aa = imread('./fff.png')
plt.imshow(aa)

import cv2
img = aa.copy()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
kernel = np.ones((5,5),np.uint8)
dilation = cv2.dilate(gray,kernel,iterations = 2)
kernel = np.ones((4,4),np.uint8)
erosion = cv2.erode(dilation,kernel,iterations = 2)
edged = cv2.Canny(erosion, 30, 200)
plt.imshow(edged, cmap='gray')

contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img,contours,-1,(0,0,255),1)   
plt.imshow(img)

i = 0
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.09 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        cv2.drawContours(img, [screenCnt], -1, (0, 255, 0), 3)
        # cv2.imwrite('cropped\\' + str(i) + '_img.jpg', img)

        i += 1

rects = [cv2.boundingRect(cnt) for cnt in contours]
rects = sorted(rects,key=lambda  x:x[1],reverse=True)
for (x, y, w, h) in rects:
    plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y])

from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.transform import warp, AffineTransform
image = edged
coords = corner_peaks(corner_harris(image), min_distance=5)
coords_subpix = corner_subpix(image, coords, window_size=13)

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)
ax.plot(coords[:, 1], coords[:, 0], color='cyan', marker='o',
        linestyle='None', markersize=6)
ax.plot(coords_subpix[:, 1], coords_subpix[:, 0], '+r', markersize=15)
plt.show()
```
```py
import matplotlib.pyplot as plt
from imutils.perspective import four_point_transform
import cv2
import imutils

# ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}
img = cv2.imread('fff.png')

img = img * float(2)
img[img > 255] = 255
img = np.round(img)
img = img.astype(np.uint8)

blurred = cv2.GaussianBlur(img, (3, 3), 5)
median = cv2.medianBlur(blurred, 3)
kernel = np.ones((3, 3), np.uint8)
edged = cv2.Canny(blurred, 75, 200)
erosion = cv2.erode(edged, kernel)
dilation = cv2.dilate(edged, kernel)
edged = cv2.Canny(dilation, 75, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
docCnt = None
if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            docCnt = approx
            break
print(docCnt)

# docCnt = np.array([[427,289],[428,335],[517,224 ],[518,268]])
paper = four_point_transform(img, docCnt.reshape(4, 2))

cv2.imshow('a',paper)
cv2.waitKey(0)
```
```py
boxes = []
areas = []
tt = []
for cnt in contours:
    # 轮廓近似，作用很小
    epsilon = 0.001 * cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    # 找到最小的矩形，该矩形可能有方向
    rect = cv2.minAreaRect(approx)

    # box是四个点的坐标
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    height = abs(box[0][1] - box[2][1])
    width = abs(box[0][0] - box[2][0])
    if height != 0 and width != 0: # and np.all(box > 0):
        boxes.append(box)
        area = height * width
        areas.append(area)
        tt.append(cnt)

for bb in boxes:
    plt.plot(np.hstack([bb[:, 0], bb[0, 0]]), np.hstack([bb[:, 1], bb[0, 1]]))

bb = boxes[np.argmax(areas)]
plt.plot(np.hstack([bb[:, 0], bb[0, 0]]), np.hstack([bb[:, 1], bb[0, 1]]))
```
```py
from skimage.draw import polygon
aa = '1575270218.2943406_46.5&28.6_48.8&71.4_253.5&71.4_251.2&28.6.jpg'
pp = np.array([[float(jj) for jj in ii.split('&')] for ii in os.path.splitext(aa)[0].split('_')[1:]])
img = np.ones((100, 300), dtype=np.uint8)
rr, cc = polygon(pp[:, 0], pp[:, 1])
img[cc, rr] = 255
plt.imshow(img)
```
