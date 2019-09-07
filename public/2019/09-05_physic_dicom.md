- [图割-最大流最小切割的最直白解读](https://www.jianshu.com/p/beca253fdc9f)
- [skimage Module: segmentation](https://scikit-image.org/docs/dev/api/skimage.segmentation.html#module-skimage.segmentation)
- [W3cubDocs scikit_image](https://docs.w3cub.com/scikit_image/api/skimage.segmentation/#skimage.segmentation.chan_vese)
- [Segmentation of objects](https://scikit-image.org/docs/stable/auto_examples/index.html#segmentation-of-objects)
- [pydicom](https://github.com/pydicom/pydicom)
- [SimpleITK Notebooks](http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/)
```py
adjust_sigmoid(image, cutoff=0.5, gain=10, inv=False)
	/cbica/home/IPP/wrappers-bin/libra --inputdir /cbica/home/IPP/IPP-users/626088490417991686/Experiments/512271977657680325/mammograms/ --outputdir /cbica/home/IPP/IPP-users/626088490417991686/Experiments/512271977657680325/Results/ --saveintermed 0
```
import matplotlib.pyplot as plt
import pydicom
aa = pydicom.read_file('./IM287')
img = aa.pixel_array
plt.imshow(img, cmap='gray')
plt.show()
```py
import glob2
from skimage import exposure

imm = glob2.glob('./*.png')
imgs = np.array([imread(ii) for ii in imm])
iee = np.array([exposure.adjust_sigmoid(ii) for ii in imgs])

plt.imshow(np.vstack([np.hstack(iee[5:]) - np.hstack(iee[:5])]))
plt.tight_layout()

plt.imshow(np.vstack([np.hstack(imgs[5:]) - np.hstack(imgs[:5])]))
plt.tight_layout()

itt = glob2.glob('./*.tif')
imt = np.array([gray2rgb(resize(imread(ii), (787, 1263))) * 255 for ii in itt]).astype(uint)
plt.imshow(np.vstack([np.hstack(iee[:5]), np.hstack(iee[5:]), np.hstack(imt)]))
plt.tight_layout()

plt.imshow(np.vstack([np.hstack(iee[:5]), np.hstack(iee[5:]), np.hstack(imt), np.vstack([np.hstack(imgs[5:]) - np.hstack(imgs[:5])])]))
plt.tight_layout()
```
```py
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimageplt.imshow(label)
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *

aa = pydicom.read_file('./IM287')
bb = pydicom.read_file('./IM288')
tt = bb.ImagePositionPatient[2] - aa.ImagePositionPatient[2]
aa.SliceThickness = tt
bb.SliceThickness = tt
image = np.stack([aa.pixel_array, bb.pixel_array])
```
```py
from skimage.color import rgb2gray                                                                                                                                                                        
from skimage.morphology import opening, binary_dilation                                                                                                                                                   
from skimage.morphology.selem import square

def pick_highlight(img, opening_square=50, dilation_square=100, dilation_thresh=0.05):
    aa = rgb2gray(img)
    bb = opening(aa, square(50))
    cc = binary_dilation(bb > 0.05, square(100))
    return aa * cc

def pick_highlight(img, opening_square=50, dilation_square=180):
    aa = rgb2gray(img)
    bb = opening(aa, square(opening_square))
    cc = binary_dilation(bb == bb.max(), square(dilation_square))
    return aa * cc

ipp = np.hstack([pick_highlight(ii) for ii in iee[5:]])
```
```py
from sklearn.cluster import KMeans
km = KMeans(n_clusters=3)
label = km.fit_predict(imgs[-1].reshape(-1, 3)).reshape([787, 1263])
plt.imshow(np.hstack([rgb2gray(imgs[-1]), label]))
```
```py
def pick_by_blobs(img, cutoff=0.5, gain=120, min_sigma=100, max_sigma=150, num_sigma=10, threshold=.1):
    itt = exposure.adjust_sigmoid(img, cutoff=cutoff, gain=gain)
    image_gray = rgb2gray(itt)
    blobs_log = blob_log(image_gray, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=threshold)
    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)
    return blobs_log

def show_imgs_and_blobs(imgs, blobs):
    rows = np.min([len(imgs), len(blobs)])
    fig, axes = plt.subplots(1, rows, figsize=(3 * rows, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for id, (ii, bbs) in enumerate(zip(imgs, blobs)):
        ax[id].imshow(ii, interpolation='nearest')
        for bb in bbs:
            y, x, r = bb
            c = plt.Circle((x, y), r, color="y", linewidth=2, fill=False)
            ax[id].add_patch(c)
        ax[id].set_axis_off()

blobs = [pick_by_blobs(ii) for ii in imgs[5:]]
show_imgs_and_blobs(imgs, blobs)
plt.tight_layout()
```
# sunny_demmo
  ```py
  # MATLAB
  H = fspecial('Gaussian', [r, c], sigma);

  # opencv-python
  # cv2.getGaussianKernel(r, sigma)返回一个shape为(r, 1)的np.ndarray, fspecial里核的size的参数是先行后列, 因此:
  H = np.multiply(cv2.getGaussianKernel(r, sigma), (cv2.getGaussianKernel(c, sigma)).T)  # H.shape == (r, c)
  ```
  ```py
  图像滤波函数imfilter函数的应用及其扩展

  import cv2
  import numpy as np
  import matplotlib.pyplot as plt

  img = cv2.imread(‘flower.jpg‘,0) #直接读为灰度图像
  img1 = np.float32(img) #转化数值类型
  kernel = np.ones((5,5),np.float32)/25

  dst = cv2.filter2D(img1,-1,kernel)
  #cv2.filter2D(src,dst,kernel,auchor=(-1,-1))函数：
  #输出图像与输入图像大小相同
  #中间的数为-1，输出数值格式的相同plt.figure()
  plt.subplot(1,2,1),plt.imshow(img1,‘gray‘)#默认彩色，另一种彩色bgr
  plt.subplot(1,2,2),plt.imshow(dst,‘gray‘)
  ```
