# ___2019 - 08 - 20 Face Anti Spoofing___
***
- [图像特征提取三大法宝：HOG特征，LBP特征，Haar特征](https://www.cnblogs.com/zhehan54/p/6723956.html)
- [活体检测Face Anti-spoofing综述](https://zhuanlan.zhihu.com/p/43480539)
- [纹理特征提取方法：LBP, 灰度共生矩阵](https://blog.csdn.net/ajianyingxiaoqinghan/article/details/71552744)
- [JinghuiZhou/awesome_face_antispoofing](https://github.com/JinghuiZhou/awesome_face_antispoofing)
- [基于LBP纹理特征计算GLCM的纹理特征统计量+SVM/RF识别纹理图片](https://blog.csdn.net/lovebyz/article/details/84032927)
- [使用深度图像的单目可见光静默活体 Binary or Auxiliary Supervision(1)](https://zhuanlan.zhihu.com/p/60155768)
- [Code for 3rd Place Solution in Face Anti-spoofing Attack Detection Challenge](https://github.com/SoftwareGift/FeatherNets_Face-Anti-spoofing-Attack-Detection-Challenge-CVPR2019)
- [Code for 2nd Place Solution in Face Anti-spoofing Attack Detection Challenge](https://github.com/SeuTao/CVPR19-Face-Anti-spoofing)
- [Aurora Guard 预测深度图 + 光验证码](https://zhuanlan.zhihu.com/p/61100492)

mplayer -tv driver=v4l2:width=352:height=288:device=/dev/video0 tv://
mplayer -tv device=/dev/video0 tv://
GAP - Global Average Pooling

# 活体检测方法
  - **纹理分析** Texture analysis，包括计算面部区域上的局部二进制模式 **LBP** 并使用 **SVM** 将面部分类为真脸或假脸
  - **频率分析** Frequency analysis，例如检查面部的 **傅里叶域**
  - **可变聚焦分析** ariable focusing analysis，例如检查两个连续帧之间的像素值的变化
  - **基于启发式的算法** Heuristic-based algorithms，包括眼球运动 / 嘴唇运动 / 眨眼检测
  - **光流算法** Optical Flow algorithms，即检查从 3D 对象和 2D 平面生成的光流的差异和属性
  - **3D脸部形状**，类似于 Apple 的 iPhone 脸部识别系统所使用的脸部形状，使脸部识别系统能够区分真人脸部和其他人的打印输出的照片图像
***

# LBP
  - **LBP 局部二进制模式** local binary pattern，是一种用来描述图像 **局部纹理特征** 的算子，原始的 LBP 于 1994 年提出，它反映内容是每个像素与周围像素的关系，后被不断的改进和优化，分别提出了 **LBP 旋转不变模式** / **LBP 均匀模式** 等
二、旋转不变的LBP模式<LBPROT>

原始的LBP不具有旋转不变性，这样我们就提出了旋转不变的LBP模式。旋转不变的LBP计算公式如下：

三：均匀LBP模式

旋转LBP模式同样存在缺陷，大量的实验证明LBP模式的36种情况在一幅图像中分布出现的频率差异较大，得到的效果并不是很好。因此人们提出了均匀LBP模式即uniform LBP。

均匀模式就是一个二进制序列从0到1或是从1到0的变过不超过2次（这个二进制序列首尾相连）。比如：10100000的变化次数为3次所以不是一个uniform pattern。所有的8位二进制数中共有58(变化次数为0的有2种，变化次数为1的有0种，变化次数为2的有56种)个uniform pattern.为什么要提出这么个uniform LBP呢，因为研究者发现他们计算出来的大部分值都在这58种之中，可达到90%以上，所以他们把值分为59类，58个uniform pattern为一类，其它的所有值为第59类。59=(2+0+56)+1,这样直方图从原来的256维变成59维。起到了降维的作用。

局部特征检测方法
斑点Blob检测，LoG检测 ， DoG，DoH检测，SIFT算法，SUFT算法
边缘检测： 梯度边缘检测算子，拉普拉斯算子，LoG检测 ，Canny边缘检测算子，Roberts,Sobel,Prewitt,
角点检测： Kitchen-Rosenfeld，Harris角点，多尺度Harris角点，KLT,SUSAN检测算子，Shi-Tomasi
将基于主分量分析和Fisher线性鉴别分析所获得的特征抽取方法，统称为线性投影分析。


静态方法，即不考虑图像之间的时序关联关系。首先，可以利用纹理信息，如傅里叶频谱分析[8]，利用人脸照片在频域中的高频分量比真人脸要少来区分。还有的利用图像多尺度和多区域的LBP特征进行二元SVM分类来区分真伪。而动态方法，便可以利用人脸的动态信息来帮助区分，例如眨眼睛的动作、嘴部的动作或者人脸多个部件的时序运动信息，或者使用背景和前景的光流信息来区分

2014年，Yang et al. [10]使用AlexNet用作特征提取器，最后加上SVM做分类器。这是强大的CNN首次被应用在人脸反欺诈中。他们在CASIA和IDIAP Replay-Attack两个数据集上的HTER均小于5%。2017年，Lucena et al.[11]使用迁移学习的思想将CNN应用在人脸反欺诈上，如下为论文中提到的FASNet[12]网络架构：

首先，他们选择VGG-16预训练模型作为基础结构，然后在人脸欺诈数据集上进行微调网络权重。除了移除最后的全连接层和将另两个全连接层的尺寸修改为256和1将其转换为一个二分类器外，FASNet和VGG-16完全一致。FASNet在3DMAD和REPLAY-ATTACK数据集上分别能达到0.0%和1.2%HTER，几乎能达到最好的水平。FASNet的思路简单，可以很容易的扩展至其他网络结构或结合其他动态特征进行检测等。

让FASNet落败的胜者，Multi-cues intergration NN方法[14]，还是在2016年发表的。它在3DMAD和REPLAY-ATTACK数据集上的HTER为0.0%和0.0%，只用了神经网络，还没用到CNN和LSTM等结构。由此可见，结合多种互补的方法，确实是个不错的思路。Mutli-cues主要包含三个方面的活体特征：shearlet图像质量特征（SBIQF），脸部运动光流特征以及场景运动光流特征，然后使用神经网络做二分类问题，模型如下图所示：


## 一种基于近红外与可见光双目摄像头的活体人脸检测方法与流程
  NIR Near Infrared
  GNIR 近红外摄像头下拍摄的真实人脸图片
  NNIR 对应 GNIR 的近红外伪造人脸图片
  VNIR 可见光伪造人脸图片
  GVIS 可见光摄像头下拍摄的真实人脸图片
  NVIS 对应 GVIS 的近红外伪造人脸图片

  (1)活体人脸分类模型的训练步骤：
  采集训练样本，包括近红外摄像头下拍摄的真实人脸图片GNIR，对应GNIR的近红外伪造人脸图片NNIR、可见光伪造人脸图片VNIR；可见光摄像头下拍摄的真实人脸图片GVIS、对应GVIS的近红外伪造人脸图片NVIS；
  样本清洗：计算训练样本的人脸的侧脸角度，剔除侧脸角度大于阈值的训练样本；
  对清洗后的训练样本进行图像预处理：计算训练样本的人脸平面旋转角度，对图片做旋转变换，使眼睛保持在图片中的水平位置；再截取只包含人脸区域的人脸图片，并进行尺寸归一化处理；
  训练近红外摄像头下区分真实人脸和可见光伪造人脸的第一活体人脸分类模型：
  对预处理后的训练样本进行第一正负样本划分：将真实人脸图像GNIR作为第一正样本；伪造人脸图片NNIR和VNIR作为第一负样本；

  提取第一正负样本的纹理特征向量：
  提取第一正负样本的8位和16位二值编码模式下的Uniform LBP特征，并分别对两种编码模式下的Uniform LBP特征进行直方图统计，得到第一正负样本的两类初始纹理特征向量；
  分别按井字形将第一正负样本均分为9个图像子块，并提取各图像子块的8位二值编码模式下的Uniform LBP特征并进行直方图统计，得到图像子块的纹理特征向量；
  拼接各正负样本的两类初始纹理特征向量和图像子块的纹理特征向量，得到样本的纹理特征向量；其中拼接方式不限，可先拼接8位二值编码模式下的初始纹理特征向量和各图像子块的纹理特征向量，再拼接16位二值编码模式下的初始纹理特征向量，当然也可以是其他方式拼接，只要满足拼接后的纹理特征向量能用于SVM(支持向量机)即可。
  基于第一正负样本的纹理特征向量，进行SVM分类模型训练，得到能够区别真实人脸和可见光伪造人脸图像的第一活体人脸分类模型；

  训练可见光摄像头下区分真实人脸和近红外伪造人脸的第二活体人脸分类模型：
  对预处理后的训练样本进行第二正负样本划分：将真实人脸图像GVIS作为第二正样本；将伪造人脸图像NVIS作为第二负样本；
  提取第二正负样本的颜色特征向量：将第二正负样本图片转换到Lab颜色空间，并对Lab颜色空间的a通道和b通道进行直方图统计，得到统计结果Sa、Sb，并将Sa和Sb拼接成一个向量，作为样本的颜色特征向量；
  基于第二正负样本的颜色特征向量，进行SVM分类模型训练，得到能够区别真实人脸和近红外伪造人脸图像的第二活体人脸分类模型；

  (2)活体人脸检测步骤：
  分别采集待检测对象在近红外摄像头和可见光摄像头下的一段满足检测时长的图像视频，对应近红外摄像头的记为第一图像视频，对应可见光摄像头的记为第二图像视频；
  判断第一和第二图像视频是否同时存在人脸，若否，则判定待检测对象为非活体人脸；若是，则分别从第一和第二图像视频中提取一帧匹配的人脸帧图像，得到第一、二人脸帧图像；其中匹配的人脸帧图像为：两个图像视频中帧时间相同且人脸侧脸角度在预设范围内(以确保所提取的图像尽量为正脸的人脸图像)的一帧图像；

  基于第一、二人脸帧图像进行活体人脸检测：
  采用与训练样本相同的图像预处理方式，对第一、二人脸帧图像进行图像预处理后；再采用提取训练样本的纹理特征向量、颜色特征向量的特征提取方式，提取第一、二人脸帧图像的纹理特征向量和颜色特征向量；
  基于第一活体人脸分类模型和第一人脸帧的纹理特征向量，获取待检测对象的第一分类结果；基于第二活体人脸分类模型和第一人脸帧的纹颜色征向量，获取待检测对象的第二分类结果；
  若第一、二分类结果均为活体人脸，则当前待检测对象为活体人脸；否则为非活体人脸。
  综上所述，由于采用了上述技术方案，本发明的有益效果是：本发明利用近红外摄像头下、视频和大部分纸张不能呈现图像的特性有效的防止了视频中伪造人脸的攻击，利用近红外摄像头和可见光摄像头下真实人脸与照片人脸纹理差异和颜色差异，训练的分类模型可以有效地区分人脸是来自真实人脸还是照片中的伪造人脸，且检测率高，从而有效的防止了视频、照片常见手段中的伪造人脸的攻击。本发明不仅在正确率上相比传统算法做出了很大的提升，保证了安全性，而且不需要用户配合机器做出相应的动作或表情，提升了用户的体验感。

  具体实施方式

  为使本发明的目的、技术方案和优点更加清楚，下面结合实施方式，对本发明作进一步地详细描述。

  在本发明中使用到可见光摄像头、近红外光摄像头，经发现，大部分的纸质材料、全部照片和全部的视频及投影所呈现的图像，在近红外光摄像头下不能正常显现，只有少数的纸张可以在近红外摄像头下呈现出正常的画面。因此使用近红外摄像头可以有效的防止来自视频、投影和大部分纸张的伪造人脸的攻击。并且在近红外摄像头下和可见光摄像头下的人脸呈现明显的差异，由于近红外摄像头拍摄的照片的光源主要来自于摄像头周围的近红外灯，因此呈现出脸部中间亮、脸颊暗、眼睛瞳孔颜色呈灰白色、并且没有颜色信息等特点。因此本发明利用人脸的纹理信息差异，准确地区分近红外摄像头下的真实人脸和可见光伪造人脸照片，利用人脸的颜色信息准确地区分可见光摄像头下的真实人脸和近红外伪造照片，融合近红外摄像头下的纹理分析和可见光摄像头下的颜色分析，两种伪造人脸照片可以检测出来，因而可以抵御照片伪造人脸的攻击、再结合近红外摄像头下无法录制到视频呈现的画面的特点，可以抵御视频伪造人脸的攻击，最终综合分析之后，可以判断出双目摄像头前的是活体还是非活体。

  本发明的具体实现步骤如下：
  (1)活体人脸分类模型的训练步骤：
  步骤1、采集训练样本集。
  采集近红外摄像头和可见光摄像头前的真实人脸和对应的伪造人脸，伪造人脸来自于几种可以近红外摄像头下呈现画面的纸张上打印的可见光人脸照片和近红外人脸照片。

  即采集的初始训练样本集包括：
  在近红外摄像头下拍摄的真实人脸图片(GNIR)、对应GNIR的近红外伪造人脸图片(NNIR)、可见光伪造人脸图片(VNIR)；
  在可见光摄像头下拍摄的真实人脸图片(GVIS)、对应GVIS的近红外伪造人脸图片(NVIS)。

  样本清洗：对采集到的图片检测人脸，根据人脸特征点的定位，计算人脸的侧脸角度，剔除初始训练样本集中，侧脸角度大于阈值的侧脸照片，得到后续步骤使用的训练样本集。

  步骤2、对训练样本集中的各训练样本进行图像预处理。
  步骤2-1：计算训练样本的人脸平面旋转角度，对图片做旋转变换，使眼睛保持在图片中的水平位置。
  步骤2-2：然后截取只包含人脸区域的人脸图片，并进行尺寸归一化处理，例如标准化为65×65大小。

  步骤3、训练近红外摄像头下区分真实人脸和可见光伪造人脸的活体人脸分类模型。
  步骤3-1：对预处理后的训练样本进行第一正负样本划分：
  将近红外摄像头采集的经过预处理后的真实人脸(GNIR)作为第一正样本；
  将伪造人脸，包括近红外伪造人脸图片(NNIR)、和可见光伪造人脸图片(VNIR)作为第一负样本。
  将近红外伪造人脸图片(NNIR)加入第一负样本的原因是，虽然近红外伪造人脸呈现出和真实人脸有很多相似性的纹理信息，但是由于图片会损失一些纹理信息，虽然不能完全抵御但是可以使训练的模型可以一定程度的抵御近红外伪造人脸图片(NNIR)。
  步骤3-2：对第一正负样本作8位和16位二值编码模式下的Uniform LBP(旋转不变LBP(局部二值模式))处理。8位二值编码模式下的Uniform LBP处理后一共得到59种模式，对59种模式进行直方图统计，可以得到一个59维的向量。16位二值编码模式下的Uniform LBP处理后一共有243种模式，进行直方图统计后可以得到一个243维的向量。
  步骤3-3：分别按井字形将第一正负样本均分为9个图像子块，这样划分可以经过Uniform LBP处理后得到眼睛、额头、脸颊、嘴唇等更多局部的特征。对每个子块同样提取8位二值编码模式下的Uniform LBP特征，则可以得到9个59维的向量。
  步骤3-4：拼接10个59维的向量以及一个243维的向量，得到各训练样本的纹理特征向量。
  步骤3-5：将第一正负样本提取出来的纹理特征向量，采用SVM(支持向量机)训练分类模型得到能够区别真实人脸和和可见光伪造人脸图片的第一活体人脸分类模型。

  步骤4、训练可见光摄像头下区分真实人脸和近红外伪造人脸的活体人脸分类模型。
  步骤4-1；对预处理后的训练样本进行第二正负样本划分：
  将经过预处理化后的可见光摄像头采集到的真实人脸图片(GVIS)，作为第二正样本；
  将可见光摄像头采集到近红外伪造人脸图片(NVIS)，作为第二负样本。
  步骤4-2：将第二正负样本图片转换到Lab颜色空间(通常正负样本图片的原颜色空间为RGB颜色空间)，并对Lab颜色空间的a通道和b通道进行直方图统计，得到统计结果Sa、Sb。
  步骤4-3：然后将统计结果Sa、Sb拼接成一个向量，作为颜色特征向量。
  步骤4-4：将第二正负样本提取出来的颜色特征向量，采用SVM训练分类模型。得到能够区别真实人脸和和近红外伪造人脸图片的第二活体人脸分类模型。

  (2)活体人脸检测步骤：
  步骤1：分别采集待检测对象在近红外摄像头和可见光摄像头下的一段满足检测时长(例如10秒)的图像视频，对应近红外摄像头的记为第一图像视频，对应可见光摄像头的记为第二图像视频。
  步骤2；检测两个图像视频是否同时存在人脸。如果两者都检测到人脸，则转向步骤3，若只在可见光摄像头下检测到人脸，在近红外摄像头下检测不到人脸，则可以推断出人脸来自于视频、投影等其他伪造人脸，转向6。若只在近红外摄像头下检测到人脸，在可见光摄像头检测不到人脸，则可以推断出人脸来自于近红外伪造人脸照片或者其他伪造情况，转向6。
  步骤3：分别从第一和第二图像视频中提取一帧匹配的人脸帧图像，得到第一、二人脸帧图像；其中匹配的人脸帧图像为：两个图像视频中帧时间相同且人脸侧脸角度在预设范围内的一帧图像。
  步骤4：采用与训练样本相同的图像预处理方式，对第一、二人脸帧图像进行图像预处理后；再采用提取训练样本的纹理特征向量、颜色特征向量的特征提取方式，提取第一、二人脸帧图像的纹理特征向量和颜色特征向量。
  步骤5：将图像预处理后的第一人脸帧图像(近红外摄像头人脸图像)在第一活体人脸分类模型上用SVM预测分类结果，将图像预处理后第二人脸帧图像(可见光摄像头人脸图像)在第二活体人脸分类模型上用SVM预测分类结果，当两种活体人脸分类模型给出的结果均为活体人脸时，则转向7。若一种活体人脸分类模型输出结果不为活体人脸，则转向6。
  步骤6：判断为非活体，输出结果。
  步骤7：判断为活体，输出结果。

  本发明通过近红外和可见光双目摄像头设计的活体检测方法，利用近红外摄像头下、视频和大部分纸张不能呈现图像的特性有效的防止了视频中伪造人脸的攻击，利用近红外摄像头和可见光摄像头下真实人脸与照片人脸纹理差异和颜色差异，训练的分类模型可以有效地区分人脸是来自真实人脸还是照片中的伪造人脸。经测试，本发明的活体检测的正确率可以达到99.9％，有效的防止了视频、照片常见手段中的伪造人脸的攻击。本发明不仅在正确率上相比传统算法做出了很大的提升，保证了安全性，而且不需要用户配合机器做出相应的动作或表情，提升了用户的体验感。

  以上所述，仅为本发明的具体实施方式，本说明书中所公开的任一特征，除非特别叙述，均可被其他等效或具有类似目的的替代特征加以替换；所公开的所有特征、或所有方法或过程中的步骤，除了互相排斥的特征和/或步骤以外，均可以任何方式组合。


## 红外
这个是demo中用到的双目摄像头,一个是红外的,一个是正常的rgb摄像头
两个usb接口,在电脑上呈现两路摄像头通道
程序检测RGB输出图像,当检测到有人脸时,用RGB人脸的位置到红外画面的位置去检测人脸
如果没有检测到,说明当前目标为非活体
当在红外画面检测到人脸时,说明当前目标为活体目标
再继续使用RGB图像提取特征值
下面为demo效果图

  近红外人脸活体检测算法主要是基于光流法而实现，无需指令配合，检测成功率较高。根据光流法，利用图像序列中的像素强度数据的时域变化和相关性来确定各自像素位置的“运动”，从图像序列中得到各个像素点的运行信息，采用高斯差分滤波器、LBP特征和支持向量机进行数据统计分析。同时，光流场对物体运动比较敏感，利用光流场可以统一检测眼球移动和眨眼。这种活体检测方式可以在用户无配合的情况下实现盲测。

近红外NIR
由于NIR的光谱波段与可见光VIS不同，故真实人脸及非活体载体对于近红外波段的吸收和反射强度也不同，即也可通过近红外相机出来的图像来活体检测。从出来的图像来说，近红外图像对屏幕攻击的区分度较大，对高清彩色纸张打印的区分度较小。

从特征工程角度来说，方法无非也是提取NIR图中的光照纹理特征[15]或者远程人脸心率特征[16]来进行。下图可见，上面两行是真实人脸图中人脸区域与背景区域的直方图分布，明显与下面两行的非活体图的分布不一致；而通过与文章[5]中一样的rPPG提取方法，在文章[]中说明其在NIR图像中出来的特征更加鲁棒~
## LBP
  ```py
  os.chdir("/home/leondgarse/workspace/samba/insightface-master")
  from face_model import FaceModel
  from mtcnn_tf.mtcnn import MTCNN
  fm = FaceModel(model_path=None, min_face_size=40, mtcnn_confidence=0.9)

  os.chdir("/home/leondgarse/workspace/datasets/NUAA")
  import skimage
  from skimage.io import imread, imsave
  from shutil import copy
  from sklearn.svm import SVC
  import os
  import numpy as np
  import skimage.feature
  from sklearn import metrics

  def extract_face_location(ff):
      img = skimage.io.imread(ff)
      bb, pp = fm.get_face_location(img)
      if bb.shape[0] == 0:
          print(">>>> No face in this picture! ff = %s, shape = %s<<<<" % (ff, bb.shape))
          copy(ff, "Detect_flaw/")
      elif bb.shape[0] > 1:
          print(">>>> NOT a single face in this picture! ff = %s, shape = %s<<<<" % (ff, bb.shape))

      nn = fm.face_align_landmarks(img, pp, image_size=[112,112])
      return nn[0] if nn.shape[0] != 0 else np.zeros([112, 112, 3], dtype=uint8)


  def image_collection_by_file(file_name, file_path, limit=None, to_array=True, save_local=True, save_base_path="./Cropped"):
      with open(file_name, 'r') as ff:
          aa = ff.readlines()
      if limit:
          aa = aa[:limit]
      image_list = [os.path.join(file_path, ii.strip().replace('\\', '/')) for ii in aa]
      image_collection = skimage.io.ImageCollection(image_list, load_func=extract_face_location)
      file_names = np.array(image_collection.files)

      if to_array:
          image_collection = image_collection.concatenate()
          pick = np.any(image_collection != np.zeros([112, 112, 3], dtype=uint8), axis=(1, 2, 3))
          image_collection = image_collection[pick]
          file_names = file_names[pick]

      if save_local:
          for tt, ff in zip(image_collection, file_names):
              save_path = os.path.dirname(os.path.join(save_base_path, ff))
              save_file = os.path.basename(ff)
              if not os.path.exists(save_path):
                  os.makedirs(save_path)
              imsave(os.path.join(save_path, save_file), tt)

      return image_collection, file_names


  def load_train_test_data(raw_path="./", limit=None):
      cur_dir = os.getcwd()
      os.chdir(raw_path.replace('~', os.environ['HOME']))
      if not os.path.exists("./train_test_dataset.npz"):
          imposter_train, imposter_train_f = image_collection_by_file("imposter_train_raw.txt", "ImposterRaw", limit=limit, save_base_path="./Cropped/imposter_train")
          client_train, client_train_f = image_collection_by_file("client_train_raw.txt", "ClientRaw", limit=limit, save_base_path="./Cropped/client_train")
          imposter_test, imposter_test_f = image_collection_by_file("imposter_test_raw.txt", "ImposterRaw", limit=limit, save_base_path="./Cropped/imposter_test")
          client_test, client_test_f = image_collection_by_file("client_test_raw.txt", "ClientRaw", limit=limit, save_base_path="./Cropped/client_test")

          train_x = np.concatenate([imposter_train, client_train])
          train_y = np.array([0] * imposter_train.shape[0] + [1] * client_train.shape[0])
          test_x = np.concatenate([imposter_test, client_test])
          test_y = np.array([0] * imposter_test.shape[0] + [1] * client_test.shape[0])

          np.savez("train_test_dataset", train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
      else:
          tt = np.load("train_test_dataset.npz")
          train_x, train_y, test_x, test_y = tt["train_x"], tt["train_y"], tt["test_x"], tt["test_y"]

      os.chdir(cur_dir)
      return train_x, train_y, test_x, test_y

  train_x, train_y, test_x, test_y = load_train_test_data('~/workspace/datasets/NUAA')
  print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)
  # (3486, 112, 112, 3) (3486,) (9119, 112, 112, 3) (9119,)
  ```
  ```py
  def plot_rgb_and_hsv(first_row, second_row, func=skimage.color.rgb2hsv):
      cc = np.min([first_row.shape[0], second_row.shape[0]])
      fig, axes = plt.subplots(4, cc)
      for id, ii in enumerate(first_row):
          pp = func(ii)
          axes[0, id].imshow(ii)
          axes[0, id].set_axis_off()
          axes[1, id].imshow(pp)
          axes[1, id].set_axis_off()
      for id, ii in enumerate(second_row):
          pp = func(ii)
          axes[2, id].imshow(ii)
          axes[2, id].set_axis_off()
          axes[3, id].imshow(pp)
          axes[3, id].set_axis_off()
      fig.tight_layout()
  ```
  ```py
  import skimage
  from skimage.data import coffee
  import skimage.feature

  img = coffee()

  for cc in [0, 1, 2]:
      img[:, :, cc] = skimage.feature.local_binary_pattern(img[:, :, cc], P=8, R=1.0, method='var')

  plt.imshow(img)
  ```
  ```py
  aa = train_x[0]
  bb = np.array([skimage.feature.local_binary_pattern(ii, 8, 1.0, method='uniform') for ii in aa.transpose(2, 0, 1)])
  cc = [skimage.exposure.histogram(ii, nbins=9)[0] for ii in bb]
  dd = np.array([skimage.feature.local_binary_pattern(ii, 16, 1.0, method='uniform') for ii in aa.transpose(2, 0, 1)])
  ee = [skimage.exposure.histogram(ii, nbins=17)[0] for ii in dd]
  ```
  ```py
  from keras.preprocessing.image import ImageDataGenerator
  # construct the training image generator for data augmentation
  aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,
      width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
      horizontal_flip=True, fill_mode="nearest")
  ```
## 灰度共生矩阵(GLCM)
  1. 算法简介
  灰度共生矩阵法(GLCM， Gray-level co-occurrence matrix)，就是通过计算灰度图像得到它的共生矩阵，然后透过计算该共生矩阵得到矩阵的部分特征值，来分别代表图像的某些纹理特征（纹理的定义仍是难点）。灰度共生矩阵能反映图像灰度关于方向、相邻间隔、变化幅度等综合信息，它是分析图像的局部模式和它们排列规则的基础。

  对于灰度共生矩阵的理解，需要明确几个概念：方向，偏移量和灰度共生矩阵的阶数。
  • 方向：一般计算过程会分别选在几个不同的方向来进行，常规的是水平方向0°，垂直90°，以及45°和135°；
  • 步距d：中心像元（在下面的例程中进行说明）；
  • 灰度共生矩阵的阶数：与灰度图像灰度值的阶数相同，即当灰度图像灰度值阶数为N时，灰度共生矩阵为N × N的矩阵；

  GLCM将拍摄的图像（作为矩阵），定义角度（[“0”，“45”，“90”，“135”]__角度在我这影响不大） 和整数距离d（[1, 2, 8, 16]__‘1’最优）。GLCM的轴由图像中存在的灰度级定义。扫描图像的每个像素并将其存储为“参考像素”。然后将参考像素与距离d的像素进行比较，该距离为角度θ（其中“0”度是右边的像素，“90”是上面的像素）远离参考像素，称为相邻像素。每次找到参考值和邻居值对时，GLCM的相应行和列递增1。

  ```py
  greycomatrix(image, distances, angles, levels=None, symmetric=False, normed=False)
      Calculate the grey-level co-occurrence matrix.

      A grey level co-occurrence matrix is a histogram of co-occurring
      greyscale values at a given offset over an image.
  ```
  图像特征值_峰度与偏度
  Kurtosis(峰度）： 表征概率密度分布曲线在平均值处峰值高低的特征数，是对Sample构成的分布的峰值是否突兀或是平坦的描述。直观看来，峰度反映了峰部的尖度。样本的峰度是和正态分布相比较而言的统计量，计算时间序列x的峰度，峰度用于度量x偏离某分布的情况，正态分布的峰度为3。如果峰度大于3，峰的形状比较尖，比正态分布峰要陡峭。反之亦然; 在统计学中，峰度衡量实数随机变量概率分布的峰态，峰度高就意味着方差增大是由低频度的大于或小于平均值的极端差值引起的。
  Skewness(偏度)： 是对Sample构成的分布的对称性状况的描述。计算时间序列x的偏度，偏度用于衡量x的对称性。若偏度为负，则x均值左侧的离散度比右侧强；若偏度为正，则x均值左侧的离散度比右侧弱。对于正态分布(或严格对称分布)偏度等于O。
  YY:（这两个值比较熟悉，以前有个图计算的项目用Spark做，SparkSQL里就有该函数）

  Python的2种参考（计算数据均值、标准差、偏度、峰度）：
  ```py
  import numpy as np
  R = np.array([1， 2， 3， 4， 5， 6]) #初始化一组数据
  R_mean = np.mean(R) #计算均值
  R_var = np.var(R)  #计算方差
  R_sc = np.mean((R - R_mean) ** 3)  #计算偏斜度
  R_ku = np.mean((R - R_mean) ** 4) / pow(R_var， 2) #计算峰度
  print([R_mean， R_var， R_sc， R_ku])


  import numpy as np
  from scipy import stats
  x = np.random.randn(10000)
  mu = np.mean(x， axis=0)
  sigma = np.std(x， axis=0)
  skew = stats.skew(x)
  kurtosis = stats.kurtosis(x)
  ```
  使用GLCM特征值+SVM对纹理图片分类
  通过提取灰度直方图的均值、标准差、峰度等统计特性和灰度共生矩阵的能量、相关性、对比度、熵值等如上所属纹理特性，作为 SVM 训练特征，得到 SVM 分类器，即可用于纹理图像的处理。
  8个类别，图片质量稍低，但也得到了0.8左右的准确率，说明了统计特征的有效性。
## bob.pad
  ```py
  def comp_block_histogram(data, neighbors=8):
      # calculating the lbp image
      lbpimage = skimage.feature.local_binary_pattern(data, neighbors, 1.0, method='nri_uniform').astype(np.int8)
      hist = skimage.exposure.histogram(lbpimage, normalize=True)[0]
      return hist

  def image_2_block_LBP_hist(image, n_vert=3, n_hor=3):
        data = skimage.color.rgb2gray(image)

        # Make sure the data can be split into equal blocks:
        row_max = int(data.shape[0] / n_vert) * n_vert
        col_max = int(data.shape[1] / n_hor) * n_hor
        data = data[:row_max, :col_max]

        blocks = [sub_block for block in np.hsplit(data, n_hor) for sub_block in np.vsplit(block, n_vert)]
        hists = [comp_block_histogram(block) for block in blocks]
        hist = np.hstack(hists)
        hist = hist / len(blocks)  # histogram normalization

        return hist
  ```
  ```py
  from sklearn.model_selection import GridSearchCV
  param_grid = {'C': [2**P for P in range(-3, 14, 2)],
                'gamma': [2**P for P in range(-15, 0, 2)], }
  clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, n_jobs=-1)

  train_x_hist = np.array([image_block_LBP_hist(ii) for ii in train_x])
  clf.fit(train_x_hist, train_y)
  clf.best_estimator_
  # Out[108]:
  # SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
  #     decision_function_shape='ovr', degree=3, gamma=0.5, kernel='rbf',
  #     max_iter=-1, probability=False, random_state=None, shrinking=True,
  #     tol=0.001, verbose=False)

  test_x_hist = np.array([image_2_block_LBP_hist(ii) for ii in test_x])
  print((clf.predict(test_x_hist) == test_y).sum() / test_y.shape[0])
  # 0.7403224037723435
  print((clf.predict(train_x_hist) == train_y).sum() / train_y.shape[0])
  # 0.9971313826735514
  ```
## bob.pad hsv
  ```py
def split_image(image, n_hor=3, n_vert=3, exclude_row=None):
    rr, cc = image.shape[:2]
    for irr in range(n_hor):
        if irr == exclude_row:
            continue
        rr_start = int(rr * irr / n_hor)
        rr_end = int(rr * (irr + 1) / n_hor)

        for icc in range(n_vert):
            cc_start = int(cc * icc / n_vert)
            cc_end = int(cc * (icc + 1) / n_vert)
            yield image[rr_start: rr_end, cc_start: cc_end]

  def plot_image_parts(image, n_hor=3, n_vert=3, exclude_row=None):
      aa = split_image(image, n_hor, n_vert, exclude_row=exclude_row)
      rows, cols = (n_hor - 1, n_vert) if exclude_row else (n_hor, n_vert)
      for id, ii in enumerate(aa):
          plt.subplot(rows, cols, id + 1)
          plt.imshow(ii)
          plt.axis('off')
      plt.tight_layout()

def stack_block_lbp_histogram(data, neighbors=8, method='uniform'):
    # calculating the lbp image
    hists = []
    bins = (neighbors * neighbors - neighbors + 3) if method == "nri_uniform" else (neighbors + 2)
    for data_channel in data.transpose(2, 0, 1):
        lbpimage = skimage.feature.local_binary_pattern(data_channel, neighbors, 1.0, method=method).astype(np.int8)
        hist = np.histogram(lbpimage, bins=bins)[0]
        hists.append(hist / hist.sum())

    return np.hstack(hists)

def image_2_block_LBP_hist(data, n_vert=3, n_hor=3, exclude_row=None, mode="YCbCr", neighbors=8, lbp_method='nri_uniform'):
      if mode.lower() == "hsv":
          data = skimage.color.rgb2hsv(data)
      elif mode.lower() == "ycbcr":
          data = skimage.color.rgb2ycbcr(data)

      # Make sure the data can be split into equal blocks:
      # row_max = int(data.shape[0] / n_vert) * n_vert
      # col_max = int(data.shape[1] / n_hor) * n_hor
      # data = data[:row_max, :col_max]
      # blocks = [sub_block for iid, block in enumerate(np.vsplit(data, n_hor)) if iid != 2 for sub_block in np.hsplit(block, n_vert)]
      blocks = split_image(data, n_hor, n_vert, exclude_row=exclude_row)
      hists = [stack_block_lbp_histogram(block, neighbors=neighbors, method=lbp_method) for block in blocks]
      hists_size = len(hists)
      hist = np.hstack(hists)
      hist = hist / hists_size  # histogram normalization

      return hist
  ```
  ```py
  from sklearn.model_selection import GridSearchCV
  from sklearn import metrics

  param_grid = {'C': [2**P for P in range(-3, 14, 2)],
                'gamma': [2**P for P in range(-15, 0, 2)], }
  param_grid = {'C': [2**P for P in range(-3, 3, 1)],
                'gamma': [2**P for P in range(0, 10, 1)], }
  param_grid = {'C': [2**P for P in range(-5, -1, 1)],
                'gamma': [2**P for P in range(-3, 3, 2)], }
  clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, n_jobs=-1, cv=3)

  train_x_hist = np.array([image_2_block_LBP_hist(ii) for ii in train_x])
  clf = clf.fit(train_x_hist, train_y)
  clf.best_estimator_
  # SVC(C=0.5, cache_size=200, class_weight='balanced', coef0=0.0,
  #     decision_function_shape='ovr', degree=3, gamma=128, kernel='rbf',
  #     max_iter=-1, probability=False, random_state=None, shrinking=True,
  #     tol=0.001, verbose=False)


  test_x_hist = np.array([image_2_block_LBP_hist(ii) for ii in test_x])
  pred = clf.predict(test_x_hist)

  def print_metrics(pred, logic):
      accuracy = (pred == logic).sum() / logic.shape[0]
      precision = np.logical_and(pred, logic).sum() / pred.sum()
      recall = np.logical_and(pred, logic).sum() / logic.sum()
      print("accuracy = %f, precision = %f, recall = %f" % (accuracy, precision, recall))
      print("Classification Report:")
      print(metrics.classification_report(logic, pred))
      print("Confusion Matrix:")
      print(metrics.confusion_matrix(logic, pred))

  print_metrics(pred, test_y)
  # accuracy = 0.831780, precision = 0.694753, recall = 0.969345
  # Classification Report:
  #               precision    recall  f1-score   support
  #
  #            0       0.98      0.75      0.85      5759
  #            1       0.69      0.97      0.81      3360
  #
  #     accuracy                           0.83      9119
  #    macro avg       0.84      0.86      0.83      9119
  # weighted avg       0.87      0.83      0.83      9119
  #
  # Confusion Matrix:
  # [[4328 1431]
  #  [ 103 3257]]
  ```
  ```py
  train_x_hist = np.array([image_2_block_LBP_hist(ii) for ii in train_x])
  train_x_hist_16 = np.array([image_2_block_LBP_hist(ii, neighbors=16) for ii in train_x])
  train_x_hist_single_8 = np.array([image_2_block_LBP_hist(ii, 1, 1) for ii in train_x])
  train_x_hist_single_16 = np.array([image_2_block_LBP_hist(ii, 1, 1, 16) for ii in train_x])

  # clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, n_jobs=-1)
  clf = GridSearchCV(SVC(kernel='rbf', class_weight={0: 0.4, 1: 0.6}), param_grid, n_jobs=-1, cv=3)
  clf = clf.fit(np.hstack([train_x_hist, train_x_hist_single_8, train_x_hist_single_16]), train_y)
  clf.best_estimator_
  # SVC(C=4, cache_size=200, class_weight='balanced', coef0=0.0,
  #     decision_function_shape='ovr', degree=3, gamma=16, kernel='rbf',
  #     max_iter=-1, probability=False, random_state=None, shrinking=True,
  #     tol=0.001, verbose=False)
  test_x_hist = np.array([image_2_block_LBP_hist(ii) for ii in test_x])
  test_x_hist_16 = np.array([image_2_block_LBP_hist(ii, neighbors=16) for ii in test_x])
  test_x_hist_single_8 = np.array([image_2_block_LBP_hist(ii, 1, 1) for ii in test_x])
  test_x_hist_single_16 = np.array([image_2_block_LBP_hist(ii, 1, 1, 16) for ii in test_x])

  pred = clf.predict(np.hstack([test_x_hist, test_x_hist_single_8, test_x_hist_single_16]))
  print_metrics(pred, test_y)
  # accuracy = 0.602149, precision = 0.476663, recall = 0.814583
  # Classification Report:
  #               precision    recall  f1-score   support
  #
  #            0       0.82      0.48      0.60      5759
  #            1       0.48      0.81      0.60      3360
  #
  #     accuracy                           0.60      9119
  #    macro avg       0.65      0.65      0.60      9119
  # weighted avg       0.69      0.60      0.60      9119
  #
  # Confusion Matrix:
  # [[2754 3005]
  #  [ 623 2737]]

  cc = SVC(C=1, gamma=16)
  cc = cc.fit(np.hstack([train_x_hist, train_x_hist_single_8, train_x_hist_single_16]), train_y)
  pred = cc.predict(np.hstack([test_x_hist, test_x_hist_single_8, test_x_hist_single_16]))
  print_metrics(pred, test_y)

  cc = SVC(C=1, gamma=64)
  cc = cc.fit(train_x_hist, train_y)
  pred = cc.predict(test_x_hist)
  print_metrics(pred, test_y)

  np.savez("train_test_hist_nri_uniform",
            train_x_hist=train_x_hist,
            train_x_hist_16=train_x_hist_16,
            train_x_hist_single_8=train_x_hist_single_8,
            train_x_hist_single_16=train_x_hist_single_16,
            test_x_hist=test_x_hist,
            test_x_hist_16=test_x_hist_16,
            test_x_hist_single_8=test_x_hist_single_8,
            test_x_hist_single_16=test_x_hist_single_16,
  )
  ```
  ```py
  def mode_selection_test(color_mode="hsv", lbp_method="uniform", save_name="foo.npz"):
      ''' Load train test data '''
      if not os.path.exists(save_name):
          train_x_hist = np.array([image_2_block_LBP_hist(ii, n_vert=5, n_hor=5, exclude_row=2, mode=color_mode, lbp_method=lbp_method) for ii in train_x])
          train_x_hist_16 = np.array([image_2_block_LBP_hist(ii, n_vert=5, n_hor=5, exclude_row=2, neighbors=16, mode=color_mode, lbp_method=lbp_method) for ii in train_x])
          train_x_hist_single_8 = np.array([image_2_block_LBP_hist(ii, n_vert=1, n_hor=1, exclude_row=None, mode=color_mode, lbp_method=lbp_method) for ii in train_x])
          train_x_hist_single_16 = np.array([image_2_block_LBP_hist(ii, n_vert=1, n_hor=1, exclude_row=None, neighbors=16, mode=color_mode, lbp_method=lbp_method) for ii in train_x])

          test_x_hist = np.array([image_2_block_LBP_hist(ii, n_vert=5, n_hor=5, exclude_row=2, mode=color_mode, lbp_method=lbp_method) for ii in test_x])
          test_x_hist_16 = np.array([image_2_block_LBP_hist(ii, n_vert=5, n_hor=5, exclude_row=2, neighbors=16, mode=color_mode, lbp_method=lbp_method) for ii in test_x])
          test_x_hist_single_8 = np.array([image_2_block_LBP_hist(ii, n_vert=1, n_hor=1, exclude_row=None, mode=color_mode, lbp_method=lbp_method) for ii in test_x])
          test_x_hist_single_16 = np.array([image_2_block_LBP_hist(ii, n_vert=1, n_hor=1, exclude_row=None, neighbors=16, mode=color_mode, lbp_method=lbp_method) for ii in test_x])

          np.savez(save_name[:-4],
                    train_x_hist=train_x_hist,
                    train_x_hist_16=train_x_hist_16,
                    train_x_hist_single_8=train_x_hist_single_8,
                    train_x_hist_single_16=train_x_hist_single_16,
                    test_x_hist=test_x_hist,
                    test_x_hist_16=test_x_hist_16,
                    test_x_hist_single_8=test_x_hist_single_8,
                    test_x_hist_single_16=test_x_hist_single_16,
          )
      else:
          tt = np.load(save_name)
          train_x_hist = tt["train_x_hist"]
          train_x_hist_16 = tt["train_x_hist_16"]
          train_x_hist_single_8 = tt["train_x_hist_single_8"]
          train_x_hist_single_16 = tt["train_x_hist_single_16"]
          test_x_hist = tt["test_x_hist"]
          test_x_hist_16 = tt["test_x_hist_16"]
          test_x_hist_single_8 = tt["test_x_hist_single_8"]
          test_x_hist_single_16 = tt["test_x_hist_single_16"]

      print("Training: %s, %s, %s, %s" % (train_x_hist.shape, train_x_hist_16.shape, train_x_hist_single_8.shape, train_x_hist_single_16.shape))
      print("Testing: %s, %s, %s, %s" % (test_x_hist.shape, test_x_hist_16.shape, test_x_hist_single_8.shape, test_x_hist_single_16.shape))

      ''' Select model '''
      trainings = [
          train_x_hist,
          train_x_hist_16,
          np.hstack([train_x_hist, train_x_hist_single_8]),
          np.hstack([train_x_hist, train_x_hist_single_16]),
          np.hstack([train_x_hist, train_x_hist_single_8, train_x_hist_single_16])
      ]
      testings = [
          test_x_hist,
          test_x_hist_16,
          np.hstack([test_x_hist, test_x_hist_single_8]),
          np.hstack([test_x_hist, test_x_hist_single_16]),
          np.hstack([test_x_hist, test_x_hist_single_8, test_x_hist_single_16])
      ]

      for training, testing in zip(trainings, testings):
          param_grid = {'C': [2**P for P in range(-3, 12, 2)],
                        'gamma': [2**P for P in range(-4, 10, 2)]
                        }
          clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid, n_jobs=-1, cv=3)
          clf = clf.fit(training, train_y)
          print("best_estimator_:")
          print(clf.best_estimator_)

          pred = clf.predict(testing)
          print_metrics(pred, test_y)

  ```
bob.pad.face/bob/pad/face/config/algorithm/video_svm_pad_algorithm.py
bob.pad.face/bob/pad/face/config/lbp_svm.py
bob.pad.face/bob/pad/face/config/qm_lr.py
bob.pad.face/bob/pad/face/extractor/LBPHistogram.py
bob.pad.face/bob/pad/face/extractor/ImageQualityMeasure.py
bob.pad.face/bob/pad/face/test/test.py
## Face_Liveness_Detection - DoG
  Face_Liveness_Detection/liveness.cpp
  ```py
  import skimage.feature
  import numpy as np

  ff = [skimage.feature.blob_dog(ii, min_sigma=0.5, max_sigma=50) for ii in train_x]
  gg = [ii.shape[0] for ii in ff]
  print(np.unique(gg))
  # [0 1 2 3 4]

  ff_test = [skimage.feature.blob_dog(ii, min_sigma=0.5, max_sigma=50) for ii in test_x]
  gg_test = [ii.shape[0] for ii in ff_test]

  def pad_or_trunc_array_constant(aa, max_row=4, max_col=4, constant_values=0):
      if aa.shape[0] > max_row or aa.shape[1] > max_col:
          aa = aa[:max_row, :max_col]
      if aa.shape != (max_row, max_col):
          pad_row = max_row - aa.shape[0]
          pad_col = max_col - aa.shape[1]
          aa = np.pad(aa, pad_width=((0, pad_row), (0, pad_col)), mode='constant', constant_values=constant_values)

      return aa

  tt = [pad_or_trunc_array_constant(ii).flatten() for ii in ff]
  tt_test = [pad_or_trunc_array_constant(ii).flatten() for ii in ff_test]

  dd = np.vstack(tt)
  dd_test = np.vstack(tt_test)

  from sklearn.svm import SVC
  clf = SVC()
  clf = clf.fit(dd, train_y)
  pred = clf.predict(dd_test)
  print_metrics(pred, test_y)

  from sklearn.linear_model import LogisticRegression
  model = LogisticRegression()
  model = model.fit(dd, train_y)
  pred = model.predict(dd_test)
  print_metrics(pred, test_y)
  ```
***

# Report
## train_test_hsv_uniform
  ```py
  In [38]: mode_selection_test(save_name="./train_test_hsv_uniform")
  Training: (3486, 243), (3486, 459), (3486, 27), (3486, 51)
  Testing: (9119, 243), (9119, 459), (9119, 27), (9119, 51)
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.780568, precision = 0.626796, recall = 0.999702
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.65      0.79      5759
             1       0.63      1.00      0.77      3360

      accuracy                           0.78      9119
     macro avg       0.81      0.83      0.78      9119
  weighted avg       0.86      0.78      0.78      9119

  Confusion Matrix:
  [[3759 2000]
   [   1 3359]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=64, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.785832, precision = 0.632711, recall = 0.998214
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.66      0.80      5759
             1       0.63      1.00      0.77      3360

      accuracy                           0.79      9119
     macro avg       0.82      0.83      0.79      9119
  weighted avg       0.86      0.79      0.79      9119

  Confusion Matrix:
  [[3812 1947]
   [   6 3354]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=128, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.725299, precision = 0.572990, recall = 0.998810
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.57      0.72      5759
             1       0.57      1.00      0.73      3360

      accuracy                           0.73      9119
     macro avg       0.79      0.78      0.73      9119
  weighted avg       0.84      0.73      0.72      9119

  Confusion Matrix:
  [[3258 2501]
   [   4 3356]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=32, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=4, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.736594, precision = 0.583420, recall = 0.997024
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.58      0.74      5759
             1       0.58      1.00      0.74      3360

      accuracy                           0.74      9119
     macro avg       0.79      0.79      0.74      9119
  weighted avg       0.84      0.74      0.74      9119

  Confusion Matrix:
  [[3367 2392]
   [  10 3350]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=512, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.25, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.706985, precision = 0.557067, recall = 0.999405
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.54      0.70      5759
             1       0.56      1.00      0.72      3360

      accuracy                           0.71      9119
     macro avg       0.78      0.77      0.71      9119
  weighted avg       0.84      0.71      0.70      9119

  Confusion Matrix:
  [[3089 2670]
   [   2 3358]]
  ```
## train_test_rgb_uniform
  ```py
  In [39]: mode_selection_test(save_name="./train_test_rgb_uniform")
  Training: (3486, 270), (3486, 486), (3486, 30), (3486, 54)
  Testing: (9119, 270), (9119, 486), (9119, 30), (9119, 54)
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.      
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=128, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=16, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

  accuracy = 0.727821, precision = 0.603343, recall = 0.762798
  Classification Report:
                precision    recall  f1-score   support
             0       0.84      0.71      0.77      5759
             1       0.60      0.76      0.67      3360

      accuracy                           0.73      9119
      macro avg       0.72      0.74      0.72      9119
      weighted avg       0.75      0.73      0.73      9119

  Confusion Matrix:
  [[4074 1685]                           
   [ 797 2563]]

  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.      
    warnings.warn(CV_WARNING, FutureWarning)

  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.768505, precision = 0.649010, recall = 0.809524
  Classification Report:
                precision    recall  f1-score   support

             0       0.87      0.74      0.80      5759
             1       0.65      0.81      0.72      3360

      accuracy                           0.77      9119
     macro avg       0.76      0.78      0.76      9119
  weighted avg       0.79      0.77      0.77      9119

  Confusion Matrix:
  [[4288 1471]
   [ 640 2720]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.638009, precision = 0.506563, recall = 0.677679
  Classification Report:
                precision    recall  f1-score   support

             0       0.77      0.61      0.68      5759
             1       0.51      0.68      0.58      3360

      accuracy                           0.64      9119
     macro avg       0.64      0.65      0.63      9119
  weighted avg       0.67      0.64      0.64      9119

  Confusion Matrix:
  [[3541 2218]
   [1083 2277]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.702380, precision = 0.569612, recall = 0.786607
  Classification Report:
                precision    recall  f1-score   support

             0       0.84      0.65      0.73      5759
             1       0.57      0.79      0.66      3360

      accuracy                           0.70      9119
     macro avg       0.70      0.72      0.70      9119
  weighted avg       0.74      0.70      0.71      9119

  Confusion Matrix:
  [[3762 1997]
   [ 717 2643]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.     
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.25, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.666959, precision = 0.534590, recall = 0.742857
  Classification Report:
                precision    recall  f1-score   support

             0       0.81      0.62      0.70      5759
             1       0.53      0.74      0.62      3360

      accuracy                           0.67      9119
     macro avg       0.67      0.68      0.66      9119
  weighted avg       0.71      0.67      0.67      9119

  Confusion Matrix:
  [[3586 2173]
   [ 864 2496]]
  ```
## train_test_ycbcr_uniform
  ```py
  In [40]: mode_selection_test(save_name="./train_test_ycbcr_uniform")
  Training: (3486, 270), (3486, 486), (3486, 30), (3486, 54)
  Testing: (9119, 270), (9119, 486), (9119, 30), (9119, 54)
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=64, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.721461, precision = 0.569515, recall = 0.999702
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.56      0.72      5759
             1       0.57      1.00      0.73      3360

      accuracy                           0.72      9119
     macro avg       0.78      0.78      0.72      9119
  weighted avg       0.84      0.72      0.72      9119

  Confusion Matrix:
  [[3220 2539]
   [   1 3359]]

  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:                                                                                                                                                                                                   
  SVC(C=32, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=64, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.737471, precision = 0.583942, recall = 1.000000
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.58      0.74      5759
             1       0.58      1.00      0.74      3360

      accuracy                           0.74      9119
     macro avg       0.79      0.79      0.74      9119
  weighted avg       0.85      0.74      0.74      9119

  Confusion Matrix:
  [[2711 3048]
   [  49 3311]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.     
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.695800, precision = 0.547782, recall = 0.999702
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.52      0.68      5759
             1       0.55      1.00      0.71      3360

      accuracy                           0.70      9119
     macro avg       0.77      0.76      0.70      9119
  weighted avg       0.83      0.70      0.69      9119

  Confusion Matrix:
  [[2986 2773]
   [   1 3359]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.     
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=512, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.653361, precision = 0.515282, recall = 0.998512
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.45      0.62      5759
             1       0.52      1.00      0.68      3360

      accuracy                           0.65      9119
     macro avg       0.76      0.73      0.65      9119
  weighted avg       0.82      0.65      0.64      9119

  Confusion Matrix:
  [[2603 3156]
   [   5 3355]]
  ```
## train_test_hsv_nri_uniform
  ```py
  In [41]: mode_selection_test(save_name="./train_test_hsv_nri_uniform")                                                                                                                                             
  Training: (3486, 1593), (3486, 6561), (3486, 177), (3486, 768)
  Testing: (9119, 1593), (9119, 6561), (9119, 177), (9119, 768)
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:                                                                                                                                                                                                   
  SVC(C=128, cache_size=200, class_weight='balanced', coef0=0.0,
        decision_function_shape='ovr', degree=3, gamma=4, kernel='rbf', max_iter=-1,
        probability=False, random_state=None, shrinking=True, tol=0.001,
        verbose=False)                                                                                                                                                                                                 
  accuracy = 0.744490, precision = 0.637701, recall = 0.709821
  Classification Report:

                precision    recall  f1-score   support

             0       0.82      0.76      0.79      5759
             1       0.64      0.71      0.67      3360

      accuracy                           0.74      9119
     macro avg       0.73      0.74      0.73      9119
  weighted avg       0.75      0.74      0.75      9119

  Confusion Matrix:
  [[4404 1355]
   [ 975 2385]]

  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.      
    warnings.warn(CV_WARNING, FutureWarning)                                                                                                                                                                         
  best_estimator_:                                                                                                                                                                                                   
  SVC(C=512, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.25, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

  accuracy = 0.831560, precision = 0.706148, recall = 0.929762
  Classification Report:

                precision    recall  f1-score   support

             0       0.95      0.77      0.85      5759
             1       0.71      0.93      0.80      3360

      accuracy                           0.83      9119
     macro avg       0.83      0.85      0.83      9119
  weighted avg       0.86      0.83      0.83      9119

  Confusion Matrix:
  [[4459 1300]
     [ 236 3124]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.25, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.818072, precision = 0.680382, recall = 0.954762
  Classification Report:
                precision    recall  f1-score   support

             0       0.97      0.74      0.84      5759
             1       0.68      0.95      0.79      3360

      accuracy                           0.82      9119
     macro avg       0.82      0.85      0.82      9119
  weighted avg       0.86      0.82      0.82      9119

  Confusion Matrix:
  [[4252 1507]
   [ 152 3208]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.25, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.833644, precision = 0.702305, recall = 0.952083
  Classification Report:
                precision    recall  f1-score   support

             0       0.96      0.76      0.85      5759
             1       0.70      0.95      0.81      3360

      accuracy                           0.83      9119
     macro avg       0.83      0.86      0.83      9119
  weighted avg       0.87      0.83      0.84      9119

  Confusion Matrix:
  [[4403 1356]
   [ 161 3199]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.     
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=32, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.0625, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.654458, precision = 0.516065, recall = 0.999107
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.45      0.62      5759
             1       0.52      1.00      0.68      3360

      accuracy                           0.65      9119
     macro avg       0.76      0.73      0.65      9119
  weighted avg       0.82      0.65      0.64      9119

  Confusion Matrix:
  [[2611 3148]
   [   3 3357]]
  ```
## train_test_rgb_nri_uniform
  ```py
  In [42]: mode_selection_test(save_name="./train_test_rgb_nri_uniform")
  Training: (3486, 1593), (3486, 6561), (3486, 177), (3486, 729)
  Testing: (9119, 1593), (9119, 6561), (9119, 177), (9119, 729)
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.      
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.25, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)

  accuracy = 0.788464, precision = 0.686863, recall = 0.782738
  Classification Report:
                precision    recall  f1-score   support

             0       0.86      0.79      0.83      5759
             1       0.69      0.78      0.73      3360

      accuracy                           0.79      9119
     macro avg       0.77      0.79      0.78      9119
  weighted avg       0.80      0.79      0.79      9119

  Confusion Matrix:
  [[4560 1199]
   [ 730 2630]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=32, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=16, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.791863, precision = 0.691461, recall = 0.785714
  Classification Report:
                precision    recall  f1-score   support

             0       0.86      0.80      0.83      5759
             1       0.69      0.79      0.74      3360

      accuracy                           0.79      9119
     macro avg       0.78      0.79      0.78      9119
  weighted avg       0.80      0.79      0.79      9119

  Confusion Matrix:
  [[4581 1178]
   [ 720 2640]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=128, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.773111, precision = 0.668582, recall = 0.761905
  Classification Report:
                precision    recall  f1-score   support

             0       0.85      0.78      0.81      5759
             1       0.67      0.76      0.71      3360

      accuracy                           0.77      9119
     macro avg       0.76      0.77      0.76      9119
  weighted avg       0.78      0.77      0.78      9119

  Confusion Matrix:
  [[4490 1269]
   [ 800 2560]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.0625, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.764777, precision = 0.671950, recall = 0.706548
  Classification Report:
                precision    recall  f1-score   support

             0       0.82      0.80      0.81      5759
             1       0.67      0.71      0.69      3360

      accuracy                           0.76      9119
     macro avg       0.75      0.75      0.75      9119
  weighted avg       0.77      0.76      0.77      9119

  Confusion Matrix:
  [[4600 1159]
   [ 986 2374]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=2048, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.0625, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.731878, precision = 0.628403, recall = 0.666369
  Classification Report:
                precision    recall  f1-score   support

             0       0.80      0.77      0.78      5759
             1       0.63      0.67      0.65      3360

      accuracy                           0.73      9119
     macro avg       0.71      0.72      0.72      9119
  weighted avg       0.74      0.73      0.73      9119

  Confusion Matrix:
  [[4435 1324]
   [1121 2239]]
  ```
## train_test_ycbcr_nri_uniform
  ```py
  In [43]: mode_selection_test(save_name="./train_test_ycbcr_nri_uniform")
  Training: (3486, 1593), (3486, 6561), (3486, 177), (3486, 729)
  Testing: (9119, 1593), (9119, 6561), (9119, 177), (9119, 729)
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.     
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=512, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=4, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.916438, precision = 0.818851, recall = 0.992857
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.87      0.93      5759
             1       0.82      0.99      0.90      3360

      accuracy                           0.92      9119
     macro avg       0.91      0.93      0.91      9119
  weighted avg       0.93      0.92      0.92      9119

  Confusion Matrix:
  [[5021  738]
   [  24 3336]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.     
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=128, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=16, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.885952, precision = 0.766177, recall = 0.993750
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.82      0.90      5759
             1       0.77      0.99      0.87      3360

      accuracy                           0.89      9119
     macro avg       0.88      0.91      0.88      9119
  weighted avg       0.91      0.89      0.89      9119

  Confusion Matrix:
  [[4740 1019]
   [  21 3339]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.     
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:                                                                                                                                                                                          [15/6362]
  SVC(C=8, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.759074, precision = 0.604643, recall = 1.000000
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.62      0.76      5759
             1       0.60      1.00      0.75      3360

      accuracy                           0.76      9119
     macro avg       0.80      0.81      0.76      9119
  weighted avg       0.85      0.76      0.76      9119

  Confusion Matrix:
  [[3562 2197]
   [   0 3360]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=8, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=1, kernel='rbf', max_iter=-1,
      probability=False, random_state=None, shrinking=True, tol=0.001,
      verbose=False)
  accuracy = 0.765764, precision = 0.611354, recall = 1.000000
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.63      0.77      5759
             1       0.61      1.00      0.76      3360

      accuracy                           0.77      9119
     macro avg       0.81      0.81      0.77      9119
  weighted avg       0.86      0.77      0.77      9119
  Confusion Matrix:
  [[3623 2136]
   [   0 3360]]
  /usr/local/lib/python3.7/dist-packages/sklearn/model_selection/_split.py:1978: FutureWarning: The default value of cv will change from 3 to 5 in version 0.22. Specify it explicitly to silence this warning.     
    warnings.warn(CV_WARNING, FutureWarning)
  best_estimator_:
  SVC(C=32, cache_size=200, class_weight='balanced', coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.0625, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False)
  accuracy = 0.680557, precision = 0.535629, recall = 1.000000
  Classification Report:
                precision    recall  f1-score   support

             0       1.00      0.49      0.66      5759
             1       0.54      1.00      0.70      3360

      accuracy                           0.68      9119
     macro avg       0.77      0.75      0.68      9119
  weighted avg       0.83      0.68      0.67      9119

  Confusion Matrix:
  [[2846 2913]
   [   0 3360]]
  ```
## Testing
```py
C = 8.000000, gamma = 32.000000
accuracy = 0.913697, precision = 0.815551, recall = 0.989583
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.87      0.93      5759
           1       0.82      0.99      0.89      3360

    accuracy                           0.91      9119
   macro avg       0.90      0.93      0.91      9119
weighted avg       0.93      0.91      0.91      9119

Confusion Matrix:
[[5007  752]
 [  35 3325]]

 C = 16.000000, gamma = 32.000000
 accuracy = 0.913477, precision = 0.815151, recall = 0.989583
 Classification Report:
               precision    recall  f1-score   support

            0       0.99      0.87      0.93      5759
            1       0.82      0.99      0.89      3360

     accuracy                           0.91      9119
    macro avg       0.90      0.93      0.91      9119
 weighted avg       0.93      0.91      0.91      9119

 Confusion Matrix:
 [[5005  754]
  [  35 3325]]

 C = 128.000000, gamma = 32.000000
 accuracy = 0.913477, precision = 0.815151, recall = 0.989583
 Classification Report:
               precision    recall  f1-score   support

            0       0.99      0.87      0.93      5759
            1       0.82      0.99      0.89      3360

     accuracy                           0.91      9119
    macro avg       0.90      0.93      0.91      9119
 weighted avg       0.93      0.91      0.91      9119

 Confusion Matrix:
 [[5005  754]
  [  35 3325]]

C = 512.000000, gamma = 32.000000
accuracy = 0.913477, precision = 0.815151, recall = 0.989583
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.87      0.93      5759
           1       0.82      0.99      0.89      3360

    accuracy                           0.91      9119
   macro avg       0.90      0.93      0.91      9119
weighted avg       0.93      0.91      0.91      9119

Confusion Matrix:
[[5005  754]
 [  35 3325]]

C = 8.000000, gamma = 32.000000
accuracy = 0.913697, precision = 0.815551, recall = 0.989583
Classification Report:
              precision    recall  f1-score   support

           0       0.99      0.87      0.93      5759
           1       0.82      0.99      0.89      3360

    accuracy                           0.91      9119
   macro avg       0.90      0.93      0.91      9119
weighted avg       0.93      0.91      0.91      9119

Confusion Matrix:
[[5007  752]
 [  35 3325]]
```
***
```py
tf.enable_eager_execution()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True))
sess = tf.Session(config=config)
keras.backend.set_session(sess)

aa = tf.keras.applications.xception.Xception(include_top=False, weights='imagenet')
aa.trainable = False
from skimage.io import imread, imsave
bb = imread('../test_images/1.jpg')
cc = bb.reshape([1, 976, 1920, 3]).astype(np.float32)
dd = aa(cc).numpy()
print(dd.shape)
# (1, 31, 60, 2048)
```
```py
class Myxception(BasicModule):
    def __init__(self):
        super(Myxception, self).__init__()
        from models.xception import xception
        model = net=xception(pretrained = True)
        self.resnet_lay=nn.Sequential(*list(model.children())[:-1])
        self.conv1_lay = nn.Conv2d(2048, 512, kernel_size = (1,1),stride=(1,1))
        self.relu1_lay = nn.ReLU(inplace = True)
        self.drop_lay = nn.Dropout2d(0.5)
        self.global_average = nn.AdaptiveAvgPool2d((1,1))
        self.fc_Linear_lay2 = nn.Linear(512,2)


    def forward(self, x):
        x= self.resnet_lay(x)
        x = self.conv1_lay(x)
        x = self.relu1_lay(x)
        x = self.drop_lay(x)
        x= self.global_average(x)
        x = x.view(x.size(0),-1)
        x = self.fc_Linear_lay2 (x)

        return x
```
```py
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Conv2D(512, 1, strides=1, activation='relu'),
    layers.Dropout(0.5),
    layers.AveragePooling2D(pool_size=1),
    layers.Flatten(),
    layers.Dense(2)
])

def load_train_test_data(raw_path="./", limit=None):
    cur_dir = os.getcwd()
    os.chdir(raw_path.replace('~', os.environ['HOME']))
    if not os.path.exists("./train_test_dataset.npz"):
        imposter_train, imposter_train_f = image_collection_by_file("imposter_train_raw.txt", "ImposterRaw", limit=limit, save_base_path="./Cropped/imposter_train")
        client_train, client_train_f = image_collection_by_file("client_train_raw.txt", "ClientRaw", limit=limit, save_base_path="./Cropped/client_train")
        imposter_test, imposter_test_f = image_collection_by_file("imposter_test_raw.txt", "ImposterRaw", limit=limit, save_base_path="./Cropped/imposter_test")
        client_test, client_test_f = image_collection_by_file("client_test_raw.txt", "ClientRaw", limit=limit, save_base_path="./Cropped/client_test")

        train_x = np.concatenate([imposter_train, client_train])
        train_y = np.array([0] * imposter_train.shape[0] + [1] * client_train.shape[0])
        test_x = np.concatenate([imposter_test, client_test])
        test_y = np.array([0] * imposter_test.shape[0] + [1] * client_test.shape[0])

        np.savez("train_test_dataset", train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
    else:
        tt = np.load("train_test_dataset.npz")
        train_x, train_y, test_x, test_y = tt["train_x"], tt["train_y"], tt["test_x"], tt["test_y"]

    os.chdir(cur_dir)
    return train_x, train_y, test_x, test_y
train_x, train_y, test_x, test_y = load_train_test_data('~/workspace/datasets/NUAA')

def split_image(image, n_hor=3, n_vert=3, exclude_row=None):
    rr, cc = image.shape[:2]
    for irr in range(n_hor):
        if irr == exclude_row:
            continue
        rr_start = int(rr * irr / n_hor)
        rr_end = int(rr * (irr + 1) / n_hor)

        for icc in range(n_vert):
            cc_start = int(cc * icc / n_vert)
            cc_end = int(cc * (icc + 1) / n_vert)
            yield image[rr_start: rr_end, cc_start: cc_end]

def stack_block_lbp_histogram(data, neighbors=8, method='uniform'):
    # calculating the lbp image
    hists = []
    bins = (neighbors * neighbors - neighbors + 3) if method == "nri_uniform" else (neighbors + 2)
    for data_channel in data.transpose(2, 0, 1):
        lbpimage = skimage.feature.local_binary_pattern(data_channel, neighbors, 1.0, method=method).astype(np.int8)
        hist = np.histogram(lbpimage, bins=bins)[0]
        hists.append(hist / hist.sum())

    return hists

def image_2_block_LBP_hist(data, n_vert=5, n_hor=5, exclude_row=2, mode="YCbCr", neighbors=8, lbp_method='nri_uniform'):
    if mode.lower() == "hsv":
        data = skimage.color.rgb2hsv(data)
    elif mode.lower() == "ycbcr":
        data = skimage.color.rgb2ycbcr(data)

    # Make sure the data can be split into equal blocks:
    # row_max = int(data.shape[0] / n_vert) * n_vert
    # col_max = int(data.shape[1] / n_hor) * n_hor
    # data = data[:row_max, :col_max]
    # blocks = [sub_block for iid, block in enumerate(np.vsplit(data, n_hor)) if iid != 2 for sub_block in np.hsplit(block, n_vert)]
    blocks = split_image(data, n_hor, n_vert, exclude_row=exclude_row)
    hists = [stack_block_lbp_histogram(block, neighbors=neighbors, method=lbp_method) for block in blocks]
    hists_size = len(hists)
    hist = np.array(hists)
    hist = hist / hists_size  # histogram normalization

    return hist

import skimage
import skimage.feature
train_x_hist = np.array([image_2_block_LBP_hist(ii).transpose(0, 2, 1) for ii in train_x])
test_x_hist = np.array([image_2_block_LBP_hist(ii).transpose(0, 2, 1) for ii in test_x])
train_y_oh = np.array([[0, 1] if ii == 0 else [1, 0] for ii in train_y])
test_y_oh = np.array([[0, 1] if ii == 0 else [1, 0] for ii in test_y])

np.savez("train_test_cnn_ycbcr_5_nri_uniform",
          train_x_hist=train_x_hist,
          test_x_hist=test_x_hist,
          train_y_oh=train_y_oh,
          test_y_oh=test_y_oh,
)
tt = np.load("train_test_cnn_ycbcr_5_nri_uniform.npz")
train_x_hist, test_x_hist, train_y_oh, test_y_oh = tt["train_x_hist"], tt["test_x_hist"], tt["train_y_oh"], tt["test_y_oh"]

model(train_x_hist[:10]).numpy()

callbacks = [keras.callbacks.TensorBoard(log_dir='./logs')]
model.fit(train_x_hist, train_y_oh, batch_size=32, epochs=50, callbacks=callbacks, validation_data=(test_x_hist, test_y_oh))

config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True))
sess = tf.Session(config=config)
keras.backend.set_session(sess)

model = keras.Sequential([
    layers.Conv2D(512, 1, strides=1, activation='relu'),
    layers.Dropout(0.5),
    layers.AveragePooling2D(pool_size=1),
    layers.Flatten(),
    layers.Dense(2, activation=tf.nn.softmax)
])

# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=outputs))
# loss = tf.losses.mean_squared_error(tf.argmax(labels, axis=-1), prediction)
BATCH_SIZE = 16
DECAY_STEPS = train_x_hist.shape[0] / BATCH_SIZE
DECAY_RATE = 0.99
LEARNING_RATE_BASE = 0.001
global_step = tf.train.get_global_step()
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, decay_steps=DECAY_STEPS, decay_rate=DECAY_RATE)

global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.001
learning_rate = tf.compat.v1.train.exponential_decay(starter_learning_rate, global_step, DECAY_STEPS, DECAY_RATE, staircase=True)
# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.compat.v1.train.GradientDescentOptimizer(learning_rate)
    .minimize(...my loss..., global_step=global_step)
)

model.compile(optimizer=tf.train.AdamOptimizer(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=tf.train.MomentumOptimizer(0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def scheduler(epoch):
    return 0.001 if epoch < 10 else 0.001 * tf.math.exp(0.1 * (10 - epoch))

callbacks = [keras.callbacks.TensorBoard(log_dir='./logs'), tf.keras.callbacks.LearningRateScheduler(scheduler)]
model.fit(train_x_hist, train_y_oh, batch_size=32, epochs=50, callbacks=callbacks, validation_data=(test_x_hist, test_y_oh))

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
|  model.fit(X_train, Y_train, callbacks=[reduce_lr])
callbacks = [keras.callbacks.TensorBoard(log_dir='./logs'), reduce_lr]

learning_rate = tf.Variable(0.001)
tf_opt = tf.train.AdadeltaOptimizer(learning_rate)
opt = keras.optimizers.TFOptimizer(tf_opt)
opt.lr = learning_rate


aa = np.load('../test_PAD.npy')
tests = np.array([image_2_block_LBP_hist(ii).transpose(0, 2, 1) for ii in aa])
pp = model.predict(tests)
(aa.shape[0] - np.argmax(pp, 1).sum()) / aa.shape[0]

tf.saved_model.simple_save(keras.backend.get_session(), './', inputs={'input_image': model.inputs[0]}, outputs={t.name:t for t in model.outputs})
```

人脸识别模型：
    人脸识别 insightface 根据项目需求部署，调试等；
    Insightface 的人脸检测模型 MTCNN 不适合多线程调用，通过 MMDNN / ONNX 等将模型转化为其他框架模型，测试 mxnet / caffe / pytorch / tensorflow 等框架下的模型效率；
    Tensorflow 框架下的 MTCNN 调优，增加 GPU 下的多线程计算，提高计算效率，单次识别可以在 30ms 以内；
    Insightface 的人脸识别模型转化为 Tensorflow 框架，测试当前运行环境下的执行效率，与高并发下的效率等；
    人脸比对算法优化，使用 dot 计算人脸 Embedded feature 距离，单次识别效率可以在 <10ms；
    模型训练的 Fine tune 调研，通过实际应用场景中的人脸数据，可以提高模型识别精度；
    测试使用 TensorRT 优化当前模型，测试执行效率。
人脸识别服务器部署：
    服务器调试，使用 flask + waitress + gunicorn + nginx 重构当前服务器代码，服务器提供人脸注册 / 人脸识别 / 人脸数据删除的基本功能；
    通过 Flask 封装模型架构，对外提供 HTTP 调用接口；
    通过 waitress 对 Flask 的封装，提高服务器的稳定性；
    通过 Gunicorn 的封装，提供多进程与多线程的调用，提高高并发的支持，并对比 sync / gevent 等多种服务器消息的处理方式，优化参数；
    通过 nginx 封装接口，提供反向代理，请求分发等，提高服务器的稳定性与高并发支持；
    在多进程的运行环境下，关于多进程通信的问题，测试使用信号 / 网络调用 / sql 等多种方式，最终选择 nginx 分发 + 网络调用的方式，重构服务器代码；
    测试各种配置下的高并发下的运行效率，如进程数/ 线程数 / GPU 占用量等，选择最适合当前环境的方式；
    Docker 封装整个模型 + 服务器。    
活体检测算法：
    添加活体检测 anti spoofing 算法，调研当前活体检测的实现方式，各个开源项目的具体实现方式，并测试效果，以及最新 arxiv 论文的研究方向，在针对当前应用环境下选择合适的算法；
    红外镜头与普通光照下的图片对比，检测人脸部分，排除一部分非活体攻击；
    对于静默图片的活体检测，获取开源数据集 NUAA，进一步采用多级 LBP 图片纹理特征的方式检测非活体；
    测试使用 RGB / HSV / YCbCr 等颜色通道下算法效果，每种颜色通道下使用 8位 uniform LBP / 16位 uniform LBP / 8 位 nri-uniform LBP / DoG 角点检测 / 灰度共生矩阵等多种特征与特征组合，检测算法实现效果；
    训练 SVM 分类器，参数调试在测试训练集上达到比较好的结果，并应用于实际场景下测试效果；
    训练 CNN 神经网络模型作为分类器，组合多种模型结构与参数，测试实际场景下的效果；
    训练使用 Alexnet / xception 模型提取特征，组合 CNN / DNN / SVM 作为分类器，测试实际场景下的效果；
