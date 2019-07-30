# 2019 - 06 - 12 Programming Computer Vision with Python
histeq 直方图均衡化
imresize 图像缩放

# Basic
## 交互式标注
  ```py
  from skimage import data
  imm = data.camera()

  plt.imshow(imm)
  plt.ginput(3)
  # Click on picture three times
  # [(195.93290043290042, 220.31385281385275),
  #  (330.3051948051948, 217.54329004328997),
  #  (272.12337662337666, 290.9632034632034)]

  plt.contour(imm, origin='image', cmap=plt.cm.gray)
  axis('equal')
  axis('off')

  hist(imm.flatten(), 128)

  from skimage.exposure import equalize_hist
  plt.imshow(equalize_hist(imm), cmap=plt.cm.gray)
  ```
  ```py
  def histeq(im, nbr_bins=256):
      """ 对一幅灰度图像进行直方图均衡化"""

      # 计算图像的直方图
      imhist, bins = np.histogram(im.flatten(), nbr_bins)
      cdf = imhist.cumsum() # cumulative distribution function
      cdf = 255 * cdf / cdf[-1] # 归一化

      # 使用累积分布函数的线性插值，计算新的像素值
      im2 = np.interp(im.flatten(), bins[:-1], cdf)

      return im2.reshape(im.shape), cdf

  from skimage import data

  imm = data.camera()
  iee, cdf = histeq(imm)
  fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=False)
  axes[0].imshow(iee, cmap=plt.cm.gray)
  axes[1].hist(imm.flatten(), 256, facecolor='r', alpha=0.5)
  axes[1].hist(iee.flatten(), 256, facecolor='b', alpha=0.5)
  axes[2].plot(cdf)
  ```
## 图像的主成分分析 PCA
  - **主成分分析 PCA** Principal Component Analysis，，在使用尽可能少维数的前提下，尽量多地保持训练数据的信息，PCA 产生的投影矩阵可以被视为将原始坐标变换到现有的坐标系，坐标系中的各个坐标按照重要性递减排列
  - 为了对图像数据进行 PCA 变换，图像需要转换成一维向量表示，将变平的图像堆积起来，可以得到一个矩阵，矩阵的一行表示一幅图像
  - 在计算主方向之前，所有的行图像按照平均图像进行中心化，然后计算协方差矩阵对应最大特征值的特征向量，通常使用 **SVD Singular Value Decomposition 奇异值分解** 方法来计算主成分
  - 当矩阵的维数很大时，SVD 的计算非常慢，所以此时通常不使用 SVD 分解，而使用另一种紧致技巧
  ```py
  def pca(X):
    """ 主成分分析：
      输入：矩阵X ，其中该矩阵中存储训练数据，每一行为一条训练数据
      返回：投影矩阵（按照维度的重要性排序）、方差和均值"""

    # 获取维数
    num_data,dim = X.shape

    # 数据中心化
    mean_X = X.mean(axis=0)
    X = X - mean_X

  if dim>num_data:
    # PCA- 使用紧致技巧
    M = np.dot(X, X.T) # 协方差矩阵
    e,EV = np.linalg.eigh(M) # 特征值和特征向量
    tmp = np.dot(X.T, EV).T # 这就是紧致技巧
    V = tmp[::-1] # 由于最后的特征向量是我们所需要的，所以需要将其逆转
    S = np.sqrt(e)[::-1] # 由于特征值是按照递增顺序排列的，所以需要将其逆转
    for i in range(V.shape[1]):
      V[:,i] /= S
  else:
    # PCA - 使用 SVD 方法
    U, S, V = np.linalg.svd(X)
    V = V[:num_data] # 仅仅返回前nun_data 维的数据才合理

  # 返回投影矩阵、方差和均值
  return V, S, mean_X
  ```
## 图像导数
  - 在很多应用中图像强度的变化情况是非常重要的信息,强度的变化可以用灰度图像 I 的 x 和 y 方向导数 Ix 和 Iy 进行描述
    ```py
    ∇I = [Ix, Iy]T
    ```
  - 梯度有两个重要的属性
    - **梯度的大小**，描述了图像强度变化的强弱
    - **梯度的角度** ``α=arctan2(Iy, Ix)``，描述了图像中在每个像素上强度变化最大的方向
  - 可以用离散近似的方式来计算图像的导数，图像导数大多数可以通过卷积简单地实现
    ```py
    Ix = I * Dx
    Iy = I * Dy
    ```
    对于 Dx 和 Dy，通常选择 **Prewitt 滤波器**
    ```py
    Dx = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]
    Dy = [[-1, -1, -1], [0, 0, 0], [1, 1, 1]]
    ```
    或者 **Sobel 滤波器**
    ```py
    Dx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    Dy = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
    ```
    ```py
    from skimage import data
    from skimage.filters import sobel, sobel_h, sobel_v

    imm = data.camera()
    imx = sobel_h(imm)  # x 方向导数
    imy = sobel_v(imm)  # y 方向导数

    magnitude = np.sqrt(imx ** 2 + imy ** 2)
    edges = sobel(imm)

    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    axes[0].imshow(imx, cmap=plt.cm.gray)
    axes[1].imshow(imy, cmap=plt.cm.gray)
    axes[2].imshow(magnitude, cmap=plt.cm.gray)
    axes[3].imshow(edges, cmap=plt.cm.gray)
    ```
    - 在导数图像中，正导数显示为亮的像素，负导数显示为暗的像素，灰色区域表示导数的值接近于零
## 图像去噪
  - **ROF (Rudin-Osher-Fatemi) 去噪模型** 图像去噪对于很多应用来说都非常重要，ROF 模型具有很好的性质，使处理后的图像更平滑，同时保持图像边缘和结构信息
  - **TV (Total Variation) 变差**。定义为一幅（灰度）图像 I的梯度范数之和
    - 在 **连续** 表示的情况下，全变差表示为 ``J(I) = ∫|∇I|dx``
    - 在 **离散** 表示的情况下，全变差表示为 ``J(I) = ∑(x)|∇I|``
  - 在 ROF 模型里，**目标函数** 为寻找降噪后的图像 U，使下式最小
    ```py
    min(u)||I - U|| ^ 2 + 2λJ(U)
    ```
    其中范数 ||I-U|| 是去噪后图像 U 和原始图像 I 差异的度量。也就是说，本质上该模型使去噪后的图像像素值“平坦”变化，但是在图像区域的边缘上，允许去噪后的图像像素值“跳跃”变化
  - python 实现
      ```py
      from numpy import *

      def denoise(im,U_init,tolerance=0.1,tau=0.125,tv_weight=100):
        """ 使用A. Chambolle（2005）在公式（11）中的计算步骤实现Rudin-Osher-Fatemi（ROF）去噪模型

          输入：含有噪声的输入图像（灰度图像）、U 的初始值、TV 正则项权值、步长、停业条件

          输出：去噪和去除纹理后的图像、纹理残留"""

        m,n = im.shape # 噪声图像的大小

        # 初始化
        U = U_init
        Px = im # 对偶域的x 分量
        Py = im # 对偶域的y 分量
        error = 1

        while (error > tolerance):
          Uold = U

          # 原始变量的梯度
          GradUx = roll(U,-1,axis=1)-U # 变量U 梯度的x 分量
          GradUy = roll(U,-1,axis=0)-U # 变量U 梯度的y 分量

          # 更新对偶变量
          PxNew = Px + (tau/tv_weight)*GradUx
          PyNew = Py + (tau/tv_weight)*GradUy
          NormNew = maximum(1,sqrt(PxNew**2+PyNew**2))

          Px = PxNew/NormNew # 更新x 分量（对偶）
          Py = PyNew/NormNew # 更新y 分量（对偶）

          # 更新原始变量
          RxPx = roll(Px,1,axis=1) # 对x 分量进行向右x 轴平移
          RyPy = roll(Py,1,axis=0) # 对y 分量进行向右y 轴平移

          DivP = (Px-RxPx)+(Py-RyPy) # 对偶域的散度
          U = im + tv_weight*DivP # 更新原始变量

          # 更新误差
          error = linalg.norm(U-Uold)/sqrt(n*m);

        return U,im-U # 去噪后的图像和纹理残余
      ```
      - roll() 在一个坐标轴上，循环“滚动”数组中的元素值。该函数可以非常方便地计算邻域元素的差异，比如这里的导数
      - linalg.norm() 可以衡量两个数组间（这个例子中是指图像矩阵 U和 Uold）的差异
  - 合成的噪声图像
    ```py
    from numpy import *
    from numpy import random
    from scipy.ndimage import filters
    import rof

    # 使用噪声创建合成图像
    im = zeros((500,500))
    im[100:400,100:400] = 128
    im[200:300,200:300] = 255
    im = im + 30*random.standard_normal((500,500))

    U,T = rof.denoise(im,im)
    G = filters.gaussian_filter(im,10)

    # 保存生成结果
    from scipy.misc import imsave
    imsave('synth_rof.pdf',U)
    imsave('synth_gaussian.pdf',G)
    ```
  - 实际图像中使用 ROF 模型去噪的效果
    ```py
    from PIL import Image
    from pylab import *
    import rof

    im = array(Image.open('empire.jpg').convert('L'))
    U,T = rof.denoise(im,im)

    figure()
    gray()
    imshow(U)
    axis('equal')
    axis('off')
    show()
    ```
***

X̅ （贴到 word 或者记事本或者其他编辑器里，横线可以正常显示在 x 上面）
刚也在找，word 提供的大概是 HTML 实现的？
这个网站 https://unicode-table.com 可以找到一些定义好的特殊字符，比如 lattin capital letter a with Macron：Ā
其他需要的带上面横线的可以通过字符 **组合用上横线 Combining Overline** 贴到字符后面显示，比如 M̅，先打 M， 再把 **组合用上横线** 贴到 M 后面
需要通过网页复制该字符 组合用上横线：https://unicode-table.com/cn/0305/，点击 复制
A̅ M̅ C̅ D̅ E̅ F̅ G̅ H̅ I̅ J̅ K̅ L̅ M̅ N̅ O̅ P̅ Q̅ R̅ S̅ T̅ U̅ V̅ W̅ X̅ Y̅ Z̅
a̅ b̅ c̅ d̅ e̅ f̅ g̅ h̅ i̅ j̅ k̅ l̅ m̅ n̅ o̅ p̅ q̅ r̅ s̅ t̅ u̅ v̅ w̅ x̅ y̅ z̅

# 局部图像描述子
## Harris 角点检测器
  - Harris 角点检测算法（也称 Harris & Stephens 角点检测器）是一个极为简单的角点检测算法，主要思想是，如果像素周围显示存在多于一个方向的边，我们认为该点为兴趣点，该点就称为 **角点**

我们把图像域中点 x 上的对称半正定矩阵 MI=MI（x）定义为：
MI = ∇I ∇IT = [[Ix], [Iy]] [Ix, Iy] = [[Ix ^ 2, Ix Iy], [Ix Iy, Iy ^ 2]]

其中 ∇I 为包含导数 Ix 和 Iy 的图像梯度（我们已经在第 1 章定义了图像的导数和梯度）。由于该定义，MI 的秩为 1，特征值为 λ1 = |∇I| ^ 2 和 λ2 = 0。现在对于图像的每一个像素，我们可以计算出该矩阵。

选择权重矩阵 W̄（通常为高斯滤波器 Gσ），我们可以得到卷积：
M̅I = W * MI

该卷积的目的是得到 MI 在周围像素上的局部平均。计算出的矩阵  有称为 Harris 矩阵。W 的宽度决定了在像素 x 周围的感兴趣区域。像这样在区域附近对矩阵  取平均的原因是，特征值会依赖于局部图像特性而变化。如果图像的梯度在该区域变化，那么  的第二个特征值将不再为 0。如果图像的梯度没有变化， 的特征值也不会变化。

取决于该区域∇I 的值，Harris 矩阵  的特征值有三种情况：

如果 λ1 和 λ2 都是很大的正数，则该 x 点为角点；

如果 λ1 很大，λ2 ≈ 0，则该区域内存在一个边，该区域内的平均 MI 的特征值不会变化太大；

如果 λ1≈ λ2 ≈ 0，该区域内为空。

在不需要实际计算特征值的情况下，为了把重要的情况和其他情况分开，Harris 和 Stephens 在文献 [12] 中引入了指示函数：
det(M̅I) - k trace(M̅I) ^ 2


为了去除加权常数 κ，我们通常使用商数：
det(M̅I) / trace(M̅I) ^ 2


作为指示器。

下面我们写出 Harris 角点检测程序。像 1.4.2 节介绍的一样，对于这个函数，我们需要使用 scipy.ndimage.filters 模块中的高斯导数滤波器来计算导数。使用高斯滤波器的道理同样是，我们需要在角点检测过程中抑制噪声强度。

首先，将角点响应函数添加到 harris.py 文件中，该函数使用高斯导数实现。同样地，参数 σ 定义了使用的高斯滤波器的尺度大小。你也可以修改这个函数，对 x 和 y 方向上不同的尺度参数，以及尝试平均操作中的不同尺度，来计算 Harris 矩阵。

from scipy.ndimage import filters
def compute_harris_response(im,sigma=3):
  """ 在一幅灰度图像中，对每个像素计算Harris 角点检测器响应函数"""

  # 计算导数
  imx = zeros(im.shape)
  filters.gaussian_filter(im, (sigma,sigma), (0,1), imx)
  imy = zeros(im.shape)
  filters.gaussian_filter(im, (sigma,sigma), (1,0), imy)

  # 计算Harris 矩阵的分量
  Wxx = filters.gaussian_filter(imx*imx,sigma)
  Wxy = filters.gaussian_filter(imx*imy,sigma)
  Wyy = filters.gaussian_filter(imy*imy,sigma)

  # 计算特征值和迹
  Wdet = Wxx*Wyy - Wxy**2
  Wtr = Wxx + Wyy

  return Wdet / Wtr

上面的函数会返回像素值为 Harris 响应函数值的一幅图像。现在，我们需要从这幅图像中挑选出需要的信息。然后，选取像素值高于阈值的所有图像点；再加上额外的限制，即角点之间的间隔必须大于设定的最小距离。这种方法会产生很好的角点检测结果。为了实现该算法，我们获取所有的候选像素点，以角点响应值递减的顺序排序，然后将距离已标记为角点位置过近的区域从候选像素点中删除。将下面的函数添加到 harris.py 文件中：

def get_harris_points(harrisim,min_dist=10,threshold=0.1):
  """ 从一幅Harris 响应图像中返回角点。min_dist 为分割角点和图像边界的最少像素数目"""

  # 寻找高于阈值的候选角点
  corner_threshold = harrisim.max() * threshold
  harrisim_t = (harrisim > corner_threshold) * 1

  # 得到候选点的坐标
  coords = array(harrisim_t.nonzero()).T

  # 以及它们的Harris 响应值
  candidate_values = [harrisim[c[0],c[1]] for c in coords]

  # 对候选点按照Harris 响应值进行排序
  index = argsort(candidate_values)

  # 将可行点的位置保存到数组中
  allowed_locations = zeros(harrisim.shape)
  allowed_locations[min_dist:-min_dist,min_dist:-min_dist] = 1

  # 按照min_distance 原则，选择最佳Harris 点
  filtered_coords = []
  for i in index:
    if allowed_locations[coords[i,0],coords[i,1]] == 1:
      filtered_coords.append(coords[i])
      allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),
            (coords[i,1]-min_dist):(coords[i,1]+min_dist)] = 0

  return filtered_coords

现在你有了检测图像中角点所需要的所有函数。为了显示图像中的角点，你可以使用 Matplotlib 模块绘制函数，将其添加到 harris.py 文件中，如下：

def plot_harris_points(image,filtered_coords):
  """ 绘制图像中检测到的角点"""

  figure()
  gray()
  imshow(image)
  plot([p[1] for p in filtered_coords],[p[0] for p in filtered_coords],'*')
  axis('off')
  show()

试着运行下面的命令：

im = array(Image.open('empire.jpg').convert('L'))
harrisim = harris.compute_harris_response(im)
filtered_coords = harris.get_harris_points(harrisim,6)
harris.plot_harris_points(im, filtered_coords)

首先，打开该图像，转换成灰度图像。然后，计算响应函数，基于响应值选择角点。最后，在原始图像中覆盖绘制检测出的角点。绘制出的结果图像如图 2-1 所示。



图 2-1：使用 Harris 角点检测器检测角点：（a）为 Harris 响应函数；（b-d）分别为使用阈值 0.01、0.05 和 0.1 检测出的角点

如果你想概要了解角点检测的不同方法，包括 Harris 角点检测器的改进和进一步的开发应用，可以查找资源，如网站 http://en.wikipedia.org/wiki/Corner_detection。

在图像间寻找对应点
Harris 角点检测器仅仅能够检测出图像中的兴趣点，但是没有给出通过比较图像间的兴趣点来寻找匹配角点的方法。我们需要在每个点上加入描述子信息，并给出一个比较这些描述子的方法。

兴趣点描述子是分配给兴趣点的一个向量，描述该点附近的图像的表观信息。描述子越好，寻找到的对应点越好。我们用对应点或者点的对应来描述相同物体和场景点在不同图像上形成的像素点。

Harris 角点的描述子通常是由周围图像像素块的灰度值，以及用于比较的归一化互相关矩阵构成的。图像的像素块由以该像素点为中心的周围矩形部分图像构成。

通常，两个（相同大小）像素块 I1(x) 和 I2(x) 的相关矩阵定义为：

c(I1, I2) = ∑(x)f(I1(x), I2(x))


其中，函数 f 随着相关方法的变化而变化。上式取像素块中所有像素位置 x 的和。对于互相关矩阵，函数 f(I1，I2)=I1I2,因此，c(I1，I2)=I1 · I2,其中 · 表示向量乘法（按照行或者列堆积的像素）。c(I1， I2) 的值越大，像素块 I1 和 I2 的相似度越高。1

1另一个常用的函数是 f(I1,I22)=(I1-I2)2 ，该函数表示平方差的和（Sum of Squared Difference，SSD）。

归一化的互相关矩阵是互相关矩阵的一种变形，可以定义为：
ncc(I1, I2) = 1 / (n - 1) * ∑(x) {(I1(x) - μ1) / σ1 * (I2(x) - μ2) / σ2}

其中，n 为像素块中像素的数目，μ1 和 μ2 表示每个像素块中的平均像素值强度，σ1 和 σ2 分别表示每个像素块中的标准差。通过减去均值和除以标准差，该方法对图像亮度变化具有稳健性。

为获取图像像素块，并使用归一化的互相关矩阵来比较它们，你需要另外两个函数。将它们添加到 harris.py 文件中：

def get_descriptors(image,filtered_coords,wid=5):
  """ 对于每个返回的点，返回点周围2*wid+1 个像素的值（假设选取点的min_distance > wid）"""

  desc = []
  for coords in filtered_coords:
    patch = image[coords[0]-wid:coords[0]+wid+1,
              coords[1]-wid:coords[1]+wid+1].flatten()
    desc.append(patch)

  return desc

def match(desc1,desc2,threshold=0.5):
  """ 对于第一幅图像中的每个角点描述子，使用归一化互相关，选取它在第二幅图像中的匹配角点"""

    n = len(desc1[0])

  # 点对的距离
  d = -ones((len(desc1),len(desc2)))
  for i in range(len(desc1)):
    for j in range(len(desc2)):
      d1 = (desc1[i] - mean(desc1[i])) / std(desc1[i])
      d2 = (desc2[j] - mean(desc2[j])) / std(desc2[j])
      ncc_value = sum(d1 * d2) / (n-1)
      if ncc_value > threshold:
        d[i,j] = ncc_value

ndx = argsort(-d)
matchscores = ndx[:,0]

return matchscores

第一个函数的参数为奇数大小长度的方形灰度图像块，该图像块的中心为处理的像素点。该函数将图像块像素值压平成一个向量，然后添加到描述子列表中。第二个函数使用归一化的互相关矩阵，将每个描述子匹配到另一个图像中的最优的候选点。由于数值较高的距离代表两个点能够更好地匹配，所以在排序之前，我们对距离取相反数。为了获得更稳定的匹配，我们从第二幅图像向第一幅图像匹配，然后过滤掉在两种方法中不都是最好的匹配。下面的函数可以实现该操作：

def match_twosided(desc1,desc2,threshold=0.5):
  """ 两边对称版本的match()"""

  matches_12 = match(desc1,desc2,threshold)
  matches_21 = match(desc2,desc1,threshold)

  ndx_12 = where(matches_12 >= 0)[0]

  # 去除非对称的匹配
  for n in ndx_12:
    if matches_21[matches_12[n]] != n:
      matches_12[n] = -1
return matches_12

这些匹配可以通过在两边分别绘制出图像，使用线段连接匹配的像素点来直观地可视化。下面的代码可以实现匹配点的可视化。将这两个函数添加到 harris.py 文件中：

def appendimages(im1,im2):
  """ 返回将两幅图像并排拼接成的一幅新图像"""

  # 选取具有最少行数的图像，然后填充足够的空行
  rows1 = im1.shape[0]
  rows2 = im2.shape[0]

  if rows1 < rows2:
    im1 = concatenate((im1,zeros((rows2-rows1,im1.shape[1]))),axis=0)
  elif rows1 > rows2:
    im2 = concatenate((im2,zeros((rows1-rows2,im2.shape[1]))),axis=0)
  # 如果这些情况都没有，那么它们的行数相同，不需要进行填充

  return concatenate((im1,im2), axis=1)

def plot_matches(im1,im2,locs1,locs2,matchscores,show_below=True):
  """ 显示一幅带有连接匹配之间连线的图片
    输入：im1，im2（数组图像），locs1，locs2（特征位置），matchscores（match() 的输出），
    show_below（如果图像应该显示在匹配的下方）"""

  im3 = appendimages(im1,im2)
  if show_below:
    im3 = vstack((im3,im3))

  imshow(im3)

  cols1 = im1.shape[1]
  for i,m in enumerate(matchscores):
    if m>0:
      plot([locs1[i][1],locs2[m][1]+cols1],[locs1[i][0],locs2[m][0]],'c')
  axis('off')

图 2-2 为使用归一化的互相关矩阵（在这个例子中，每个像素块的大小为 11×11）来寻找对应点的例子。该图像可以通过下面的命令实现：

wid = 5
harrisim = harris.compute_harris_response(im1,5)
filtered_coords1 = harris.get_harris_points(harrisim,wid+1)
d1 = harris.get_descriptors(im1,filtered_coords1,wid)

harrisim = harris.compute_harris_response(im2,5)
filtered_coords2 = harris.get_harris_points(harrisim,wid+1)
d2 = harris.get_descriptors(im2,filtered_coords2,wid)

print 'starting matching'
matches = harris.match_twosided(d1,d2)

figure()
gray()
harris.plot_matches(im1,im2,filtered_coords1,filtered_coords2,matches)
show()



图 2-2：将归一化的互相关矩阵应用于 Harris 角点周围图像块，来寻找匹配对应点

为了看得更清楚，你可以画出匹配的子集。在上面的代码中，可以通过将数组 matches 替换成 matches[:100] 或者任意子集来实现。

如图 2-2 所示，该算法的结果存在一些不正确匹配。这是因为，与现代的一些方法相比，图像像素块的互相关矩阵具有较弱的描述性。实际运用中，我们通常使用更稳健的方法来处理这些对应匹配。这些描述符还有一个问题，它们不具有尺度不变性和旋转不变性，而算法中像素块的大小也会影响对应匹配的结果。

近年来诞生了很多用来提高特征点检测和描述性能的方法。在下一节中，我们来学习其中最好的一种算法。
