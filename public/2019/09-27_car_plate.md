- [车牌识别（一）-车牌定位](https://www.cnblogs.com/polly333/p/7367479.html)
- [车牌识别（一）——车牌定位(附详细代码及注释）](https://blog.csdn.net/lixiaoyu101/article/details/86626626)
- [车牌识别1：License Plate Detection and Recognition in Unconstrained Scenarios阅读笔记](https://www.cnblogs.com/greentomlee/p/10863363.html)
- [Github HyperLPR 高性能开源中文车牌识别框架](https://github.com/zeusees/HyperLPR)
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
        if (area < 2000):
            continue

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
