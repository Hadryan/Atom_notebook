什么是fine-tuning？
  在实践中，由于数据集不够大，很少有人从头开始训练网络。常见的做法是使用预训练的网络（例如在ImageNet上训练的分类1000类的网络）来重新fine-tuning（也叫微调），或者当做特征提取器。
  以下是常见的两类迁移学习场景：
1 卷积网络当做特征提取器。使用在ImageNet上预训练的网络，去掉最后的全连接层，剩余部分当做特征提取器（例如AlexNet在最后分类器前，是4096维的特征向量）。这样提取的特征叫做CNN codes。得到这样的特征后，可以使用线性分类器（Liner SVM、Softmax等）来分类图像。
2 Fine-tuning卷积网络。替换掉网络的输入层（数据），使用新的数据继续训练。Fine-tune时可以选择fine-tune全部层或部分层。通常，前面的层提取的是图像的通用特征（generic features）（例如边缘检测，色彩检测），这些特征对许多任务都有用。后面的层提取的是与特定类别有关的特征，因此fine-tune时常常只需要Fine-tuning后面的层。
 
预训练模型
在ImageNet上训练一个网络，即使使用多GPU也要花费很长时间。因此人们通常共享他们预训练好的网络，这样有利于其他人再去使用。例如，Caffe有预训练好的网络地址Model Zoo。

  何时以及如何Fine-tune
决定如何使用迁移学习的因素有很多，这是最重要的只有两个：新数据集的大小、以及新数据和原数据集的相似程度。有一点一定记住：网络前几层学到的是通用特征，后面几层学到的是与类别相关的特征。这里有使用的四个场景：
1、新数据集比较小且和原数据集相似。因为新数据集比较小，如果fine-tune可能会过拟合；又因为新旧数据集类似，我们期望他们高层特征类似，可以使用预训练网络当做特征提取器，用提取的特征训练线性分类器。
2、新数据集大且和原数据集相似。因为新数据集足够大，可以fine-tune整个网络。
3、新数据集小且和原数据集不相似。新数据集小，最好不要fine-tune，和原数据集不类似，最好也不使用高层特征。这时可是使用前面层的特征来训练SVM分类器。
4、新数据集大且和原数据集不相似。因为新数据集足够大，可以重新训练。但是实践中fine-tune预训练模型还是有益的。新数据集足够大，可以fine-tine整个网络。
 
实践建议
预训练模型的限制。使用预训练模型，受限于其网络架构。例如，你不能随意从预训练模型取出卷积层。但是因为参数共享，可以输入任意大小图像；卷积层和池化层对输入数据大小没有要求（只要步长stride fit），其输出大小和属于大小相关；全连接层对输入大小没有要求，输出大小固定。
学习率。与重新训练相比，fine-tune要使用更小的学习率。因为训练好的网络模型权重已经平滑，我们不希望太快扭曲（distort）它们（尤其是当随机初始化线性分类器来分类预训练模型提取的特征时）。

```py
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py
--network r100
--loss-type 4
--margin-m 0.5
--prefix ../model-r100
--pretrained '../model-r100, 287' > ../model-r100.log 2>&1 &

I use ArchLoss
Have to discard the weight of fc7 layer, as the number of output is not the same.
@myfighterforever Use deploy/model_slim.py. As is named, this script can slim model. Just in case you need it still.
```
```py
finetune_lr = {k: 0 for k in arg_params}
for key in aux_params:
finetune_lr[key]=0
opt.set_lr_mult(finetune_lr)
#print(aux_params['fc7_weight'])
#print(arg_params.keys())

model.fit(train_dataiter,
    begin_epoch        = begin_epoch,
    num_epoch          = 1,
    eval_data          = val_dataiter,
    eval_metric        = eval_metrics,
    kvstore            = args.kvstore,
    optimizer          = opt,
    #optimizer_params   = optimizer_params,
    initializer        = initializer,
    arg_params         = arg_params,
    aux_params         = aux_params,
    allow_missing      = True,
    batch_end_callback = _batch_callback,
    epoch_end_callback = epoch_cb )
for key in finetune_lr:
    finetune_lr[key]=args.lr
opt.set_lr_mult(finetune_lr)
print('start train the structure')
model.fit(train_dataiter,
          begin_epoch=1,
          num_epoch=100,
          eval_data=val_dataiter,
          eval_metric=eval_metrics,
          kvstore=args.kvstore,
          optimizer=opt,
          # optimizer_params   = optimizer_params,
          initializer=initializer,
          arg_params=arg_params,
          aux_params=aux_params,
          allow_missing=True,
          batch_end_callback=_batch_callback,
          epoch_end_callback=epoch_cb)
like this
```
```py
Delete fc7 weight by calling deploy/model_slim.py
python model_slim.py --model ../models/model-r34-amf/model,0
```
```py
I use the pretrained model https://github.com/deepinsight/insightface/wiki/Model-Zoo#31-lresnet100e-irarcfacems1m-refine-v2 to finetune on the same ms1m-refine-v2 dataset(https://github.com/deepinsight/insightface/wiki/Dataset-Zoo#ms1m-refine-v2recommended)
The accuracy start from 0, is this normal?
The command line is as follows

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r100 --loss-type 4 --margin-m 0.5 --data-dir ../datasets/faces_emore  --prefix ../model-r100 --per-batch-size 64  --pretrained ../pre_trained_models/model-r100-ii/model,0

Try decreasing learning rate and also it is not recommended to fine-tune the same dataset.
```
```py
Hi
I want to finetune your pretrained model (r50) with embedding size 128. Starting point is your train_softmax.py:
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --emb-size=128 --pretrained '../models/model-r50-am-lfw/model,0' --network r50 --loss-type 0 --margin-m 0.5 --data-dir ../datasets/faces_ms1m_112x112 --prefix ../models/model-r50-am-lfw/

this results in:
mxnet.base.MXNetError: [08:51:39] src/operator/nn/../tensor/../elemwise_op_common.h:123: Check failed: assign(&dattr, (*vec)[i]) Incompatible attr in node at 0-th output: expected [512], got [128]

I have tried to remove the last fc layer (fc1), then add fc7 as done in your code, then freeze all layers except fc7. Still the dimensions don't match.

Any advise on how to do this? Thanks.

Embedding layer(fc1) is not the last FC layer, but fc7(embedding to number of classes) is. You should train from scratch with different embedding size

train fc1 and fc7 from scratch

I had figured out the same as @zhaowwenzhong and managed to finetune the model by replacing the last two fully connected layers fc1 and fc7 and freezing all others.

@nttstar I want to train from scratch with different embedding size.
1、_weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0, wd_mult=args.fc7_wd_mult)
change to ---->>>_weight = mx.symbol.Variable("fc7_weight_finetune", init = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2),shape=(args.num_classes, args.emb_size), lr_mult=1.0, wd_mult=args.fc7_wd_mult)

2、fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7')
change to --->> >fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7_finetune')
```
```py
Appreciate for you goog work! first I train on ms1m dataset until acc is 90% on train dataset. then I wan to finetune on my own data. I have make the *.rec and *.idx. but when I begun to finetune, the program load the pr-trained model, and crash. warning that the number of my dataset classes is not equeal to the ms1m. it seems we should freezen the FC layer.

is your code offer the function of finetuneing?

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py
--network r100
--loss-type 4
--margin-m 0.5
--prefix ../model-r100
--pretrained '../model-r100, 287' > ../model-r100.log 2>&1 &

I use ArchLoss

Have to discard the weight of fc7 layer, as the number of output is not the same.

@myfighterforever Use deploy/model_slim.py. As is named, this script can slim model.
Just in case you need it still.
```
```py
Hi, I also encountered the same problem. And I find solution here #569 .

Check config.ckpt_embedding in your config.py, it's True as default.
If config.ckpt_embedding is set to True, the checkpoint only save the model until embedding layer. The other weights are lost such as fc7_weight. This led to the low accuracy when fine-tuning.

The conclusion is, if I want to train A model from scratch and take A as a pretrained model for training B model, config.ckpt_embedding should be set to False while training A and be set to True while training B. (If B model is final model I need.)
```
```sh
export MXNET_ENABLE_GPU_P2P=0
CUDA_VISIBLE_DEVICES='0,1' python -u train_parall.py --network m1 --loss arcface --dataset emore --per-batch-size 96
CUDA_VISIBLE_DEVICES='0,1' python -u train.py --network m1 --loss arcface --dataset emore --per-batch-size 96 --pretrained ./models/m1-arcface-emore/model --lr 0.0001
CUDA_VISIBLE_DEVICES='' python -u train.py --network m1 --loss triplet --dataset emore --per-batch-size 120 --pretrained ~/workspace/samba/m1-arcface-emore/model --lr 0.0001 --verbose 400 --lr-steps '10000,16000,22000'
CUDA_VISIBLE_DEVICES='1' python -u train.py --network m1 --loss triplet --dataset emore --per-batch-size 150 --pretrained ./models/m1-triplet-emore_97083/model --lr 0.0001 --lr-steps '1000,100000,160000,220000,280000,340000'
CUDA_VISIBLE_DEVICES='0' python -u train.py --network m1 --loss triplet --dataset glint --per-batch-size 150 --pretrained ./models/m1-triplet-emore_290445/model --pretrained-epoch 602 --lr 0.0001 --lr-steps '1000,100000,160000,220000,280000,340000'

CUDA_VISIBLE_DEVICES='0,1' python3 -u train_parall.py --network vargfacenet --loss softmax --dataset emore --per-batch-size 96
CUDA_VISIBLE_DEVICES='1' python3 -u train.py --network vargfacenet --loss arcface --dataset glint --per-batch-size 150 --pretrained ./model
s/vargfacenet-softmax-emore/model --pretrained-epoch 166 --lr 0.0001 --lr-steps '100000,160000,220000,280000,340000'

```
# save and restore Tensorflow models
  - [A quick complete tutorial to save and restore Tensorflow models](https://cv-tricks.com/tensorflow-tutorial/save-restore-tensorflow-models-quick-complete-tutorial/)
  ```py
  oring model and retraining with your own dataPython

  import tensorflow as tf

  sess=tf.Session()    
  #First let's load meta graph and restore weights
  saver = tf.train.import_meta_graph('my_test_model-1000.meta')
  saver.restore(sess,tf.train.latest_checkpoint('./'))


  # Now, let's access and create placeholders variables and
  # create feed-dict to feed new data

  graph = tf.get_default_graph()
  w1 = graph.get_tensor_by_name("w1:0")
  w2 = graph.get_tensor_by_name("w2:0")
  feed_dict ={w1:13.0,w2:17.0}

  #Now, access the op that you want to run.
  op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

  print sess.run(op_to_restore,feed_dict)
  #This will print 60 which is calculated
  #using new values of w1 and w2 and saved value of b1.
  ```
  ```py
  import tensorflow as tf

  sess=tf.Session()    
  #First let's load meta graph and restore weights
  saver = tf.train.import_meta_graph('my_test_model-1000.meta')
  saver.restore(sess,tf.train.latest_checkpoint('./'))


  # Now, let's access and create placeholders variables and
  # create feed-dict to feed new data

  graph = tf.get_default_graph()
  w1 = graph.get_tensor_by_name("w1:0")
  w2 = graph.get_tensor_by_name("w2:0")
  feed_dict ={w1:13.0,w2:17.0}

  #Now, access the op that you want to run.
  op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

  #Add more to the current graph
  add_on_op = tf.multiply(op_to_restore,2)

  print sess.run(add_on_op,feed_dict)
  #This will print 120.
  ```
  ```py
  ......
  ......
  saver = tf.train.import_meta_graph('vgg.meta')
  # Access the graph
  graph = tf.get_default_graph()
  ## Prepare the feed_dict for feeding data for fine-tuning

  #Access the appropriate output for fine-tuning
  fc7= graph.get_tensor_by_name('fc7:0')

  #use this if you only want to change gradients of the last layer
  fc7 = tf.stop_gradient(fc7) # It's an identity function
  fc7_shape= fc7.get_shape().as_list()

  new_outputs=2
  weights = tf.Variable(tf.truncated_normal([fc7_shape[3], num_outputs], stddev=0.05))
  biases = tf.Variable(tf.constant(0.05, shape=[num_outputs]))
  output = tf.matmul(fc7, weights) + biases
  pred = tf.nn.softmax(output)

  # Now, you run this with fine-tuning data in sess.run()
  ```
# 人脸识别损失函数
  - 人脸识别模型训练的损失函数主要分为 **基于分类 softmax 的损失函数** 和 **基于 triplet loss 的损失函数** 两大类
    - **基于分类 softmax 的损失函数** 因为是否对 embedding 或分类权重 W 做归一化以及是否增加额外的间隔 margin 等产生了多种变体
    - **基于 triplet loss 的损失函数** 则分为基于欧氏距离和基于角度距离两种
  - **基于分类 softmax 的损失函数**
    - **基本的 softmax 分类** 通过将 embedding 输入一层全连接层以及 softmax 函数得到分类概率，由于 softmax 的分母对 embedding 在各个类别上的结果进行了求和，因此最小化这一损失一定程度上能够使类间距离变大，类内距离变小
      - N 表示样本数量
      - n 表示类别总数
      - yi 表示样本 xi 的真实类别
    - **Sphereface Loss** 在 softmax 的基础上进一步引入了显式的角度间隔 angular margin，从而训练时能够进一步缩小类内距离，扩大类间距离
    - **CosineFace Loss** 进一步对人脸表示 embedding 进行了归一化，从而使分类结果仅取决于夹角余弦，并进一步引入了余弦间隔 m，用于扩大类间距离，缩小类内距离。由于余弦的取值范围较小，为了使类别间差别更显著，进一步引入一个超参数 s 用于放大余弦值
    - **Arcface Loss** 为了使人脸表示 embedding 的学习更符合超球体流形假设，Arcface 进一步将 Cosineface 中的余弦间隔修改为角度间隔，得到如下损失

    | 损失函数   | 分类边界                      |
    | ---------- | ----------------------------- |
    | Softmax    | (W1 - W2) * x + b1 - b2 = 0   |
    | SphereFace | ∥x∥ * (cosmθ1 - cosθ2) = 0    |
    | CosineFace | s * (cosθ1 - m - cosθ2) = 0   |
    | ArcFace    | s * (cos(θ1 + m) - cosθ2) = 0 |

  - **基于 triplet loss 的损失函数** 与通过 softmax 优化使类内距离缩小，类间距离扩大不同，Triplet Loss 直接对样本间的距离进行优化，使不同类样本间的距离比同类样本间的距离大出一个间隔，因此计算 Triplet Loss 每次需要采样三个样本 anchor / positive / negative，其中，anchor 与 positive 样本属于同一类别，与 negative 样本属于不同类别

    - x代表人脸表示 embedding
    - 上标 a,p,n 分别表示 anchor，positive 和 negative
    - dist(x,y) 表示 x,y 的距离函数
    - m 则表示不同类样本间距离比同类样本间距离大出的间隔，这里的距离函数和间隔既可以是欧氏距离也可以是角度距离等形式
  - **softmax 损失与 Triplet Loss 结合**
    - Triplet Loss 直接对样本表示间的距离进行优化，在训练数据足够多，模型表示能力足够强的情况下，能够学得很好的结果
    - 其缺点是，一方面训练时模型收敛速度较慢，另一方面在构造triplet时需要选择合适的正样本对和负样本对，因此需要设计 triplet 的构造选择机制，这一过程通常比较复杂
    - 较好的训练方式是先用分类损失训练模型，然后再用 Triplet Loss 对模型进行 finetune 以进一步提升模型性能
# 计算点集中的矩形面积
  ```py
  def cal_rectangle_areas(aa):
      ''' 每个 x 坐标对应的所有 y 值 '''
      xxs = {}
      for ixx, iyy in aa:
          tt = xxs.get(ixx, set())
          tt.add(iyy)
          xxs[ixx] = tt
      print(xxs)
      # {1: [1, 3], 4: [1], 2: [2, 3], 3: [2, 3], 5: [2, 3]}

      ''' 遍历 xxs，对于每个 x 值，在所有其他 x 值对应的 y 值集合上，查找交集 '''
      rect = []
      areas = []
      while len(xxs) != 0:
          xa, ya = xxs.popitem()
          if len(ya) < 2:
              continue
          for xb, yb in xxs.items():
              width = abs(xb - xa)
              tt = list(ya.intersection(yb))
              while len(tt) > 1:
                  rect.extend([((xa, xb), (tt[0], ii)) for ii in tt[1:]])
                  areas.extend([width * abs(tt[0] - ii) for ii in tt[1:]])
                  tt = tt[1:]

      print(rect, areas)
      # [((5, 2), (2, 3)), ((5, 3), (2, 3)), ((3, 2), (2, 3))] [3, 2, 1]
      return rect, areas

  def draw_rectangles(dots, coords, labels, alpha=0.3):
      colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] * 8
      fig, axes = plt.subplots(1, 1)
      axes.scatter(dots[:, 0], dots[:, 1])
      for (xx, yy), label, cc in zip(coords, labels, colors):
          axes.plot([xx[0], xx[1], xx[1], xx[0], xx[0]], [yy[0], yy[0], yy[1], yy[1], yy[0]], color=cc, label=label)
          rect = plt.Rectangle((xx[0], yy[0]), xx[1] - xx[0], yy[1] - yy[0], color=cc, alpha=alpha)
          axes.add_patch(rect)
      axes.legend()
      fig.tight_layout()

  aa = np.array([[1, 1], [4, 1], [2, 2], [3, 2], [5, 2], [1, 3], [2, 3], [3, 3], [5, 3]])
  plt.scatter(aa[:, 0], aa[:, 1])
  rect, areas = cal_rectangle_areas(aa)
  draw_rectangles(aa, rect, areas)

  bb = aa[:, ::-1]
  rect, areas = cal_rectangle_areas(bb)
  draw_rectangles(bb, rect, areas)

  xx = np.random.randint(1, 50, 100)
  yy = np.random.randint(1, 50, 100)
  aa = np.array(list(zip(xx, yy)))
  rect, areas = cal_rectangle_areas(aa)
  draw_rectangles(aa, rect, areas)
  ```
  ![](images/calc_rectangle_area.png)
***
后台管理系统 http://47.103.82.71:38838/admin/login
人脸注册 http://47.103.82.71:38838/faceRecognize/registerFace
人脸检测 http://47.103.82.71:38838/faceRecognize/detectFace
人脸删除 http://47.103.82.71:38838/faceRecognize/deleteFace

```py
loaded = tf.saved_model.load('models_resnet')
interf = loaded.signatures['serving_default']
teacher_model_interf = lambda images: interf(images)['output'].numpy()

def image_generator(teacher_model_interf, input_shape=[112, 112, 3], batch_size=64):
    while True:
        batch_x = tf.random.uniform([batch_size] + input_shape, minval=0, maxval=255, dtype=tf.float32)
        batch_y = teacher_model_interf(batch_x)
        yield (batch_x, batch_y)
train_gen = image_generator(teacher_model_interf)
ixx, iyy = next(train_gen)

xx = tf.keras.applications.MobileNetV2(input_shape=[112, 112, 3], include_top=False, weights=None)
# xx = tf.keras.applications.NASNetMobile(input_shape=[112, 112, 3], include_top=False, weights=None)
xx.trainable = True
model = tf.keras.models.Sequential([
    xx,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(512)
])
model.compile(optimizer='adam', loss='mse', metrics=["mae",'accuracy'])
model.summary()

model.fit_generator(train_gen, epochs=20, steps_per_epoch=10, use_multiprocessing=True)
```
```py
import glob2
from skimage.io import imread
def data_gen(path, batch_size=64, shuffle=True, base_path_replace=[]):
    while True:
        image_path_files = glob2.glob(os.path.join(path, '*_img.foo'))
        image_path_files = np.random.permutation(image_path_files)
        emb_files = [ii.replace('_img.foo', '_emb.npy') for ii in image_path_files]
        print("This should be the epoch start, total files = %d" % (image_path_files.shape[0]))
        for ipps, iees in zip(image_path_files, emb_files):
            with open(ipps, 'r') as ff:
                image_paths = np.array([ii.strip() for ii in ff.readlines()])
            image_embs = np.load(iees)
            total = len(image_paths)
            if shuffle:
                indexes = np.random.permutation(total)
            else:
                indexes = np.arange(total)
            print("\nimage_path_files = %s, emb_files = %s, total = %d" % (ipps, iees, total))
            for id in range(0, total, batch_size):
                cc = indexes[id: id + batch_size]
                # print("id start = %d, end = %d, cc = %s" % (id, id + batch_size, cc))
                image_batch_data = image_paths[cc]
                if len(image_batch_data) < batch_size:
                    continue
                if len(base_path_replace) != 0:
                    image_batch_data = [ii.replace(base_path_replace[0], base_path_replace[1]) for ii in image_batch_data]
                images = (np.array([imread(ii) for ii in image_batch_data]) / 255).astype('float32')
                embs = image_embs[cc]

                yield (images, embs)
            print("Processed Id: %d - %d" % (id, id + batch_size))

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

BATCH_SIZE = 100
DATA_PATH = './'
# train_gen = data_gen(DATA_PATH, batch_size=BATCH_SIZE, shuffle=True, base_path_replace=['/media/uftp/images', '/home/leondgarse/workspace/images'])
train_gen = data_gen(DATA_PATH, batch_size=BATCH_SIZE, shuffle=True)
steps_per_epoch = int(np.floor(100000 / BATCH_SIZE) * len(os.listdir(DATA_PATH)) / 2)

hist = model.fit_generator(train_gen, epochs=50, steps_per_epoch=steps_per_epoch, verbose=1)
```
```py
from skimage.io import imread
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

pp = '/media/uftp/images/faces_emore_112x112_folders/'
with open('faces_emore_img.foo', 'w') as ff:
    for dd in os.listdir(pp):
        dd = os.path.join(pp, dd)
        for ii in os.listdir(dd):
            ff.write(os.path.join(dd, ii) + '\n')
# 5822653
with open('faces_emore_img.foo', 'r') as ff:
    tt = [ii.strip() for ii in ff.readlines()]

for ii in range(59):
    print(ii * 100000, (ii+1) * 100000)
    with open('./{}_img.foo'.format(ii), 'w') as ff:
        ff.write('\n'.join(tt[ii * 100000: (ii+1) * 100000]))

loaded = tf.saved_model.load('./model_resnet')
_interp = loaded.signatures["serving_default"]
interp = lambda ii: _interp(tf.convert_to_tensor(ii, dtype="float32"))["output"].numpy()

with open('0_img.foo', 'r') as ff:
    tt = [ii.strip() for ii in ff.readlines()]
print(len(tt))

ees = []
for id, ii in enumerate(tt):
    ii = ii.replace('/media/uftp', '/home/leondgarse/workspace')
    imm = imread(ii)
    ees.append(interp([imm])[0])
    if id % 10 == 0:
        print("Processing %d..." % id)

from skimage.io import ImageCollection
icc = ImageCollection(tt)
for id, (iff, imm) in enumerate(zip(icc.files, icc)):
    if id % 10 == 0:
        print("Processing %d..." % id)
    ees.append(interp([imm])[0])

import glob2
for fn in glob2.glob('./*_img.foo'):
    with open(fn, 'r') as ff:
        tt = [ii.strip() for ii in ff.readlines()]
    target_file = fn.replace('_img.foo', '_emb')
    print(fn, len(tt), target_file)

    ees = []
    for id, ii in enumerate(tt):
        # ii = ii.replace('/media/uftp', '/home/leondgarse/workspace')
        imm = imread(ii)
        ees.append(interp([imm])[0])
        if id % 100 == 0:
            print("Processing %d..." % id)
    ees = np.array(ees)
    print(ees.shape)
    np.save(target_file, ees)
```
```py
from skimage.io import imread
from sklearn.preprocessing import normalize

def model_test(image_paths, model_path, scale=1.0):
    loaded = tf.saved_model.load(model_path)
    interf = loaded.signatures['serving_default']
    images = [imread(ipp) * scale for ipp in image_paths]

    preds = interf(tf.convert_to_tensor(images, dtype='float32'))['output'].numpy()
    return np.dot(normalize(preds), normalize(preds).T), preds

images = ['/home/leondgarse/workspace/samba/1770064353.jpg', '/home/leondgarse/workspace/samba/541812715.jpg']
model_test(images, 'model_mobilefacenet/')
```
```sh
My 2-stage pipeline:

Train softmax with lr=0.1 for 120K iterations.
LRSTEPS='240000,360000,440000'
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 0 --prefix "$PREFIX" --per-batch-size 128 --lr-steps "$LRSTEPS" --margin-s 32.0 --margin-m 0.1 --ckpt 2 --emb-size 128 --fc7-wd-mult 10.0 --wd 0.00004 --max-steps 140002
Switch to ArcFace loss to do normal training with '100K,140K,160K' iterations.
LRSTEPS='100000,140000,160000'
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 4 --prefix "$PREFIX" --per-batch-size 128 --lr-steps "$LRSTEPS" --margin-s 64.0 --margin-m 0.5 --ckpt 1 --emb-size 128 --fc7-wd-mult 10.0 --wd 0.00004 --pretrained '../models2/model-y1-test/model,70'
Pretrained model: baiduyun
training dataset: ms1m
LFW: 99.50, CFP_FP: 88.94, AgeDB30: 95.91
```
```py
import glob2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

path = '/home/tdtest/workspace/insightface-master/faces_emore_img/emb_done'
image_path_files = glob2.glob(os.path.join(path, '*_img.foo'))
emb_files = [ii.replace('_img.foo', '_emb.npy') for ii in image_path_files]
image_names = []
image_embs = []
for ii, ee in zip(image_path_files, emb_files):
    with open(ii, 'r') as ff:
        image_names.extend([ii.strip() for ii in ff.readlines()])
    image_embs.append(np.load(ee))
image_embs = np.concatenate(image_embs)
image_classes = [int(os.path.basename(os.path.dirname(ii))) for ii in image_names]
print(len(image_names), len(image_classes), image_embs.shape)

np.savez('faces_emore_class_emb', image_names=np.array(image_names), image_classes=np.array(image_classes), image_embs=image_embs)
aa = np.load('faces_emore_class_emb.npz')
image_names, image_classes, image_embs = aa['image_names'], aa['image_classes'], aa['image_embs']
classes = np.max(image_classes) + 1
print(image_names.shape, image_classes.shape, image_embs.shape, classes)
# (5822653,) (5822653,) (5822653, 512) 85742

data_df = pd.DataFrame({"image_names": image_names, "image_classes": image_classes, "image_embs": list(image_embs)})
image_gen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
train_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col=["image_classes", "image_embs"], class_mode='multi_output', target_size=(112, 112), batch_size=128, seed=1, subset='training', validate_filenames=False)
# Found 5240388 non-validated image filenames.
val_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col=["image_classes", "image_embs"], class_mode='multi_output', target_size=(112, 112), batch_size=128, seed=1, subset='validation', validate_filenames=False)
# Found 582265 non-validated image filenames.

xx = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
xx.trainable = True
inputs = layers.Input(shape=(112, 112, 3))
nn = xx(inputs)
nn = layers.GlobalAveragePooling2D()(nn)
nn = layers.BatchNormalization()(nn)
nn = layers.Dropout(0.1)(nn)
embedding = layers.Dense(512)(nn)
logits = layers.Dense(classes, activation='softmax')(embedding)

model = keras.models.Model(inputs, [logits, embedding])
model.compile(optimizer='adam', loss=[keras.losses.sparse_categorical_crossentropy, keras.losses.mse])
# model.compile(optimizer='adam', loss=[keras.losses.sparse_categorical_crossentropy, keras.losses.mse], metrics=['accuracy', 'mae'])
model.summary()

reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=5, verbose=1)
model_checkpoint = ModelCheckpoint("./keras_checkpoints", 'val_loss', verbose=1, save_best_only=True)
callbacks = [model_checkpoint, reduce_lr]
hist = model.fit_generator(train_data_gen, validation_data=val_data_gen, epochs=200, verbose=1, callbacks=callbacks)
```
***
# Keras Insightface
## MXnet record to folder
  ```py
  import os
  import numpy as np
  import mxnet as mx
  from tqdm import tqdm

  # read_dir = '/datasets/faces_glint/'
  # save_dir = '/datasets/faces_glint_112x112_folders'
  read_dir = '/datasets/faces_emore'
  save_dir = '/datasets/faces_emore_112x112_folders'
  idx_path = os.path.join(read_dir, 'train.idx')
  bin_path = os.path.join(read_dir, 'train.rec')

  imgrec = mx.recordio.MXIndexedRecordIO(idx_path, bin_path, 'r')
  rec_header, _ = mx.recordio.unpack(imgrec.read_idx(0))

  # for ii in tqdm(range(1, 10)):
  for ii in tqdm(range(1, int(rec_header.label[0]))):
      img_info = imgrec.read_idx(ii)
      header, img = mx.recordio.unpack(img_info)
      img_idx = str(int(np.sum(header.label)))
      img_save_dir = os.path.join(save_dir, img_idx)
      if not os.path.exists(img_save_dir):
          os.makedirs(img_save_dir)
      # print(os.path.join(img_save_dir, str(ii) + '.jpg'))
      with open(os.path.join(img_save_dir, str(ii) + '.jpg'), 'wb') as ff:
          ff.write(img)
  ```
  ```py
  import io
  import pickle
  import tensorflow as tf
  from skimage.io import imread

  test_bin_file = '/datasets/faces_emore/agedb_30.bin'
  test_bin_file = '/datasets/faces_emore/cfp_fp.bin'
  with open(test_bin_file, 'rb') as ff:
      bins, issame_list = pickle.load(ff, encoding='bytes')

  bb = [tf.image.encode_jpeg(imread(io.BytesIO(ii))) for ii in bins]
  with open(test_bin_file, 'wb') as ff:
      pickle.dump([bb, issame_list], ff)
  ```
## Loading data by ImageDataGenerator
  ```py
  ''' flow_from_dataframe '''
  import glob2
  import pickle
  image_names = glob2.glob('/datasets/faces_emore_112x112_folders/*/*.jpg')
  image_names = np.random.permutation(image_names).tolist()
  image_classes = [int(os.path.basename(os.path.dirname(ii))) for ii in image_names]

  with open('faces_emore_img_class_shuffle.pkl', 'wb') as ff:
      pickle.dump({'image_names': image_names, "image_classes": image_classes}, ff)

  import pickle
  from keras.preprocessing.image import ImageDataGenerator
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  with open('faces_emore_img_class_shuffle.pkl', 'rb') as ff:
      aa = pickle.load(ff)
  image_names, image_classes = aa['image_names'], aa['image_classes']
  image_names = np.random.permutation(image_names).tolist()
  image_classes = [int(os.path.basename(os.path.dirname(ii))) for ii in image_names]
  print(len(image_names), len(image_classes))
  # 5822653 5822653

  data_df = pd.DataFrame({"image_names": image_names, "image_classes": image_classes})
  data_df.image_classes = data_df.image_classes.map(str)
  # image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.1)
  image_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, validation_split=0.05)
  train_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col="image_classes", class_mode='categorical', target_size=(112, 112), batch_size=128, subset='training', validate_filenames=False)
  # Found 5240388 non-validated image filenames belonging to 85742 classes.
  val_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col="image_classes", class_mode='categorical', target_size=(112, 112), batch_size=128, subset='validation', validate_filenames=False)
  # Found 582265 non-validated image filenames belonging to 85742 classes.

  classes = data_df.image_classes.unique().shape[0]
  steps_per_epoch = np.ceil(len(train_data_gen.classes) / 128)
  validation_steps = np.ceil(len(val_data_gen.classes) / 128)

  ''' Convert to tf.data.Dataset '''
  train_ds = tf.data.Dataset.from_generator(lambda: train_data_gen, output_types=(tf.float32, tf.int32), output_shapes=([None, 112, 112, 3], [None, classes]))
  # train_ds = train_ds.cache()
  # train_ds = train_ds.shuffle(buffer_size=128 * 1000)
  train_ds = train_ds.repeat()
  train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)

  val_ds = tf.data.Dataset.from_generator(lambda: val_data_gen, output_types=(tf.float32, tf.int32), output_shapes=([None, 112, 112, 3], [None, classes]))

  xx, yy = next(iter(train_ds))
  print(xx.shape, yy.shape)
  # (128, 112, 112, 3) (128, 85742)
  ```
## Loading data by Datasets
  ```py
  import glob2
  import pickle
  image_names = glob2.glob('/datasets/faces_emore_112x112_folders/*/*.jpg')
  image_names = np.random.permutation(image_names).tolist()
  image_classes = [int(os.path.basename(os.path.dirname(ii))) for ii in image_names]

  with open('faces_emore_img_class_shuffle.pkl', 'wb') as ff:
      pickle.dump({'image_names': image_names, "image_classes": image_classes}, ff)

  import pickle
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  with open('faces_emore_img_class_shuffle.pkl', 'rb') as ff:
      aa = pickle.load(ff)
  image_names, image_classes = aa['image_names'], aa['image_classes']
  classes = np.max(image_classes) + 1
  print(len(image_names), len(image_classes), classes)
  # 5822653 5822653 85742

  # list_ds = tf.data.Dataset.list_files('/datasets/faces_emore_112x112_folders/*/*')
  list_ds = tf.data.Dataset.from_tensor_slices(image_names)

  def process_path(file_path, classes, img_shape=(112, 112)):
      parts = tf.strings.split(file_path, os.path.sep)[-2]
      label = tf.cast(tf.strings.to_number(parts), tf.int32)
      label = tf.one_hot(label, depth=85742, dtype=tf.int32)
      img = tf.io.read_file(file_path)
      img = tf.image.decode_jpeg(img, channels=3)
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = tf.image.resize(img, img_shape)
      img = tf.image.random_flip_left_right(img)
      return img, label

  def prepare_for_training(ds, cache=True, shuffle_buffer_size=None, batch_size=128):
      if cache:
          ds = ds.cache(cache) if isinstance(cache, str) else ds.cache()
      if shuffle_buffer_size == None:
          shuffle_buffer_size = batch_size * 100

      ds = ds.shuffle(buffer_size=shuffle_buffer_size)
      ds = ds.repeat()
      ds = ds.map(lambda xx: process_path(xx, classes), num_parallel_calls=AUTOTUNE)
      ds = ds.batch(batch_size)
      ds = ds.prefetch(buffer_size=AUTOTUNE)
      return ds

  # train_ds = prepare_for_training(labeled_ds, cache="/tmp/faces_emore.tfcache")
  batch_size = 128 * len(tf.config.experimental.get_visible_devices('GPU'))
  train_ds = prepare_for_training(list_ds, cache=False, shuffle_buffer_size=len(image_names), batch_size=batch_size)
  steps_per_epoch = np.ceil(len(image_names) / batch_size)
  image_batch, label_batch = next(iter(train_ds))
  print(image_batch.shape, label_batch.shape)
  # (128, 112, 112, 3) (128, 85742)
  ```
## Evaluate
  ```py
  import pickle
  import io
  from tqdm import tqdm
  from skimage.io import imread
  from sklearn.preprocessing import normalize

  class epoch_eval_callback(tf.keras.callbacks.Callback):
      def __init__(self, test_bin_file, batch_size=128, rescale=1./255, save_model=None):
          super(epoch_eval_callback, self).__init__()
          bins, issame_list = np.load(test_bin_file, encoding='bytes', allow_pickle=True)
          ds = tf.data.Dataset.from_tensor_slices(bins)
          _imread = lambda xx: tf.image.convert_image_dtype(tf.image.decode_jpeg(xx), dtype=tf.float32)
          ds = ds.map(_imread)
          self.ds = ds.batch(128)
          self.test_issame = np.array(issame_list)
          self.test_names = os.path.splitext(os.path.basename(test_bin_file))[0]
          self.max_accuracy = 0
          self.batch_size = batch_size
          self.steps = int(np.ceil(len(bins) / batch_size))
          self.save_model = save_model

      # def on_batch_end(self, batch=0, logs=None):
      def on_epoch_end(self, epoch=0, logs=None):
          dists = []
          embs = []
          tf.print("\n")
          for img_batch in tqdm(self.ds, 'Evaluating ' + self.test_names, total=self.steps):
              emb = basic_model.predict(img_batch)
              embs.extend(emb)
          embs = np.array(embs)
          if np.isnan(embs).sum() != 0:
              tf.print("NAN in embs, not a good one")
              return
          embs = normalize(embs)
          embs_a = embs[::2]
          embs_b = embs[1::2]
          dists = (embs_a * embs_b).sum(1)

          self.tt = np.sort(dists[self.test_issame[:dists.shape[0]]])
          self.ff = np.sort(dists[np.logical_not(self.test_issame[:dists.shape[0]])])

          max_accuracy = 0
          thresh = 0
          for vv in reversed(self.ff[-300:]):
              acc_count = (self.tt > vv).sum() + (self.ff <= vv).sum()
              acc = acc_count / dists.shape[0]
              if acc > max_accuracy:
                  max_accuracy = acc
                  thresh = vv
          tf.print("\n")
          if max_accuracy > self.max_accuracy:
              is_improved = True
              self.max_accuracy = max_accuracy
              if self.save_model:
                  save_path = '%s_%d' % (self.save_model, epoch)
                  tf.print("Saving model to: %s" % (save_path))
                  model.save(save_path)
          else:
              is_improved = False
          tf.print(">>>> %s evaluation max accuracy: %f, thresh: %f, overall max accuracy: %f, improved = %s" % (self.test_names, max_accuracy, thresh, self.max_accuracy, is_improved))
  ```
  ```py
  class mi_basic_model:
      def __init__(self):
          self.predict = lambda xx: interf(xx)['embedding'].numpy()
      def save(self, path):
          print('Saved to %s' % (path))
  basic_model = mi_basic_model()

  aa = epoch_eval_callback('/datasets/faces_emore/agedb_30.bin', save_model='./test')
  aa = epoch_eval_callback('/home/leondgarse/workspace/datasets/faces_emore/lfw.bin')
  aa.on_epoch_end()
  ```
  ```py
  # basic_model_centsoft_0_split.h5
  >>>> lfw evaluation max accuracy: 0.992833, thresh: 0.188595, overall max accuracy: 0.992833
  >>>> cfp_fp evaluation max accuracy: 0.909571, thresh: 0.119605, overall max accuracy: 0.909571
  >>>> agedb_30 evaluation max accuracy: 0.887500, thresh: 0.238278, overall max accuracy: 0.887500
  ```
  ```py
  # basic_model_arc_8_split.h5
  >>>> lfw evaluation max accuracy: 0.994167, thresh: 0.141986, overall max accuracy: 0.994167
  >>>> cfp_fp evaluation max accuracy: 0.867429, thresh: 0.106673, overall max accuracy: 0.867429
  >>>> agedb_30 evaluation max accuracy: 0.902167, thresh: 0.128596, overall max accuracy: 0.902167
  ```
  ```py
  # basic_model_arc_split.h5
  >>>> lfw evaluation max accuracy: 0.993000, thresh: 0.125761, overall max accuracy: 0.993000
  >>>> agedb_30 evaluation max accuracy: 0.912667, thresh: 0.084312, overall max accuracy: 0.912667
  >>>> cfp_fp evaluation max accuracy: 0.859143, thresh: 0.068290, overall max accuracy: 0.859143
  ```
## Basic model
  ```py
  from tensorflow.keras import layers

  ''' Basic model '''
  # xx = keras.applications.ResNet101V2(include_top=False, weights='imagenet')
  # xx = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
  # xx = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
  xx = tf.keras.applications.ResNet50V2(input_shape=(112, 112, 3), include_top=False, weights='imagenet')
  xx.trainable = True

  inputs = xx.inputs[0]
  nn = xx.outputs[0]
  nn = layers.GlobalAveragePooling2D()(nn)
  nn = layers.Dropout(0.1)(nn)
  embedding = layers.Dense(512, name='embedding')(nn)
  basic_model = keras.models.Model(inputs, embedding)

  ''' Callbacks '''
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

  # lfw_eval = epoch_eval_callback('/datasets/faces_emore/lfw.bin')
  lfw_eval = epoch_eval_callback('/datasets/faces_emore/lfw.bin', save_model=None)
  agedb_30_eval = epoch_eval_callback('/datasets/faces_emore/agedb_30.bin', save_model=None)
  cfp_fp_eval = epoch_eval_callback('/datasets/faces_emore/cfp_fp.bin', save_model=None)

  # reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=5, verbose=1)
  def scheduler(epoch):
      lr = 0.001 if epoch < 10 else 0.001 * np.exp(0.2 * (10 - epoch))
      print('\nLearning rate for epoch {} is {}'.format(epoch + 1, lr))
      return lr
  lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)

  model_checkpoint = ModelCheckpoint("./keras_checkpoints.h5", verbose=1)
  # model_checkpoint = ModelCheckpoint("./keras_checkpoints_res_arcface", 'val_loss', verbose=1, save_best_only=True)
  callbacks = [lr_scheduler, lfw_eval, agedb_30_eval, cfp_fp_eval, model_checkpoint]

  ''' Model with bottleneck '''
  class NormDense(tf.keras.layers.Layer):
      def __init__(self, classes=1000, **kwargs):
          super(NormDense, self).__init__(**kwargs)
          self.output_dim = classes
      def build(self, input_shape):
          self.w = self.add_weight(name='norm_dense_w', shape=(input_shape[-1], self.output_dim), initializer='random_normal', trainable=True)
          super(NormDense, self).build(input_shape)
      def call(self, inputs, **kwargs):
          norm_w = tf.nn.l2_normalize(self.w, axis=0)
          # inputs = tf.nn.l2_normalize(inputs, axis=1)
          return tf.matmul(inputs, norm_w)
      def compute_output_shape(self, input_shape):
          shape = tf.TensorShape(input_shape).as_list()
          shape[-1] = self.output_dim
          return tf.TensorShape(shape)
      def get_config(self):
          base_config = super(NormDense, self).get_config()
          base_config['output_dim'] = self.output_dim
      @classmethod
      def from_config(cls, config):
          return cls(**config)

  inputs = basic_model.inputs[0]
  embedding = basic_model.outputs[0]
  output = NormDense(classes, name='norm_dense')(embedding)
  concate = layers.concatenate([embedding, output], name='concate')
  model = keras.models.Model(inputs, concate)
  # model.load_weights('nn.h5')
  model.summary()

  ''' Loss function wrapper '''
  def logits_accuracy(y_true, y_pred):
      logits = y_pred[:, 512:]
      return keras.metrics.categorical_accuracy(y_true, logits)
  ```
  ```py
  import multiprocessing as mp
  mp.set_start_method('forkserver')
  hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, initial_epoch=11, use_multiprocessing=True, workers=4)

  # hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch)
  hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps)

  hist = model.fit(train_ds, epochs=200, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, validation_data=val_ds, validation_steps=validation_steps)
  ```
  ```py
  from tensorflow.keras import layers
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

  ''' Basic model '''
  # Multi GPU
  mirrored_strategy = tf.distribute.MirroredStrategy()
  with mirrored_strategy.scope():
      # xx = keras.applications.ResNet101V2(include_top=False, weights='imagenet')
      # xx = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
      xx = tf.keras.applications.ResNet50V2(include_top=False, weights='imagenet')
      xx.trainable = True

      inputs = xx.inputs[0]
      nn = xx.outputs[0]
      nn = layers.GlobalAveragePooling2D()(nn)
      nn = layers.Dropout(0.1)(nn)
      embedding = layers.Dense(512, name='embedding')(nn)
      basic_model = keras.models.Model(inputs, embedding)

      basic_model.load_weights('basic_model_arc_8_split.h5')
      basic_model.trainable = False
  ```
## Softmax train
  ```py
  def softmax_loss(y_true, y_pred):
      logits = y_pred[:, 512:]
      return keras.losses.categorical_crossentropy(y_true, logits, from_logits=True)

  with mirrored_strategy.scope():
      model.compile(optimizer='adamax', loss=softmax_loss, metrics=[logits_accuracy])
  ```
  ```py
  Epoch 1/200
  43216/43216 [==============================] - 8650s 200ms/step - loss: 3.8481 - accuracy: 0.4496 - val_loss: 2.6660 - val_accuracy: 0.5180
  Epoch 2/200
  43216/43216 [==============================] - 8792s 203ms/step - loss: 0.9634 - accuracy: 0.8118 - val_loss: 1.2425 - val_accuracy: 0.7599
  Epoch 3/200
  43216/43216 [==============================] - 8720s 202ms/step - loss: 0.6660 - accuracy: 0.8676 - val_loss: 1.3942 - val_accuracy: 0.7380
  Epoch 4/200
  43216/43216 [==============================] - 8713s 202ms/step - loss: 0.5394 - accuracy: 0.8920 - val_loss: 0.6720 - val_accuracy: 0.8733
  Epoch 5/200
  43216/43216 [==============================] - 8873s 205ms/step - loss: 0.4662 - accuracy: 0.9063 - val_loss: 0.7837 - val_accuracy: 0.8540
  ```
## Arcface loss
  ```py
  # def arcface_loss(y_true, y_pred, margin1=1.0, margin2=0.2, margin3=0.3, scale=64.0):
  def arcface_loss(y_true, y_pred, margin1=0.9, margin2=0.4, margin3=0.15, scale=64.0):
      # y_true = tf.squeeze(y_true)
      # y_true = tf.cast(y_true, tf.int32)
      # y_true = tf.argmax(y_true, 1)
      # cos_theta = tf.nn.l2_normalize(logits, axis=1)
      # theta = tf.acos(cos_theta)
      # mask = tf.one_hot(y_true, epth=norm_logits.shape[-1])
      embedding = y_pred[:, :512]
      logits = y_pred[:, 512:]
      norm_emb = tf.norm(embedding, axis=1, keepdims=True)
      norm_logits = logits / norm_emb
      theta = tf.acos(norm_logits)
      cond = tf.where(tf.greater(theta * margin1 + margin3, np.pi), tf.zeros_like(y_true), y_true)
      cond = tf.cast(cond, dtype=tf.bool)
      m1_theta_plus_m3 = tf.where(cond, theta * margin1 + margin3, theta)
      cos_m1_theta_plus_m3 = tf.cos(m1_theta_plus_m3)
      arcface_logits = tf.where(cond, cos_m1_theta_plus_m3 - margin2, cos_m1_theta_plus_m3) * scale
      tf.assert_equal(tf.math.is_nan(tf.reduce_mean(arcface_logits)), False)
      return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=True)

  with mirrored_strategy.scope():
      model.compile(optimizer='adamax', loss=arcface_loss, metrics=[logits_accuracy])
  ```
  ```py
  Epoch 1/200
  43216/43216 [==============================] - 9148s 212ms/step - loss: 34.7171 - accuracy: 0.0088 - val_loss: 14.4753 - val_accuracy: 0.0010
  Epoch 2/200
  43216/43216 [==============================] - 8995s 208ms/step - loss: 14.4870 - accuracy: 0.0022 - val_loss: 14.2573 - val_accuracy: 0.0118
  Epoch 3/200
  43216/43216 [==============================] - 8966s 207ms/step - loss: 14.5741 - accuracy: 0.0146 - val_loss: 14.6334 - val_accuracy: 0.0156
  Epoch 4/200
  43216/43216 [==============================] - 8939s 207ms/step - loss: 14.6519 - accuracy: 0.0175 - val_loss: 14.3232 - val_accuracy: 0.0158
  Epoch 5/200
  43216/43216 [==============================] - 9122s 211ms/step - loss: 14.6973 - accuracy: 0.0198 - val_loss: 15.0081 - val_accuracy: 0.0210
  ```
## Arcface loss 2
  ```py
  def arcface_loss(labels, norm_logits, s=64.0, m=0.45):
      # labels = tf.squeeze(labels)
      # labels = tf.cast(labels, tf.int32)
      # norm_logits = tf.nn.l2_normalize(logits, axis=1)

      cos_m = tf.math.cos(m)
      sin_m = tf.math.sin(m)
      mm = sin_m * m
      threshold = tf.math.cos(np.pi - m)

      cos_t2 = tf.square(norm_logits)
      sin_t2 = tf.subtract(1., cos_t2)
      sin_t = tf.sqrt(sin_t2)
      cos_mt = s * tf.subtract(tf.multiply(norm_logits, cos_m), tf.multiply(sin_t, sin_m), name='cos_mt')
      cond_v = norm_logits - threshold
      cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
      keep_val = s * (norm_logits - mm)
      cos_mt_temp = tf.where(cond, cos_mt, keep_val)
      # mask = tf.one_hot(labels, depth=norm_logits.shape[-1])
      mask = tf.cast(labels, tf.float32)
      inv_mask = tf.subtract(1., mask)
      s_cos_t = tf.multiply(s, norm_logits)
      arcface_logits = tf.add(tf.multiply(s_cos_t, inv_mask), tf.multiply(cos_mt_temp, mask))
      # return tf.keras.losses.sparse_categorical_crossentropy(labels, arcface_logits, from_logits=True)
      return tf.keras.losses.categorical_crossentropy(labels, arcface_logits, from_logits=True)
  ```
## Center loss
  ```py
  class Save_Numpy_Callback(tf.keras.callbacks.Callback):
      def __init__(self, save_file, save_tensor):
          super(Save_Numpy_Callback, self).__init__()
          self.save_file = os.path.splitext(save_file)[0]
          self.save_tensor = save_tensor

      def on_epoch_end(self, epoch=0, logs=None):
          np.save(self.save_file, self.save_tensor.numpy())

  class Center_loss(keras.losses.Loss):
      def __init__(self, num_classes, feature_dim=512, alpha=0.5, factor=1.0, initial_file=None):
          super(Center_loss, self).__init__()
          self.alpha = alpha
          self.factor = factor
          centers = tf.Variable(tf.zeros([num_classes, feature_dim]))
          if initial_file:
              if os.path.exists(initial_file):
                  aa = np.load(initial_file)
                  centers.assign(aa)
              self.save_centers_callback = Save_Numpy_Callback(initial_file, centers)
          self.centers = centers

      def call(self, y_true, y_pred):
          labels = tf.argmax(y_true, axis=1)
          centers_batch = tf.gather(self.centers, labels)
          # loss = tf.reduce_mean(tf.square(y_pred - centers_batch))
          loss = tf.reduce_mean(tf.square(y_pred - centers_batch), axis=-1)

          # Update centers
          # diff = (1 - self.alpha) * (centers_batch - y_pred)
          diff = centers_batch - y_pred
          unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
          appear_times = tf.gather(unique_count, unique_idx)
          appear_times = tf.reshape(appear_times, [-1, 1])

          diff = diff / tf.cast((1 + appear_times), tf.float32)
          diff = self.alpha * diff
          # print(centers_batch.shape, self.centers.shape, labels.shape, diff.shape)
          self.centers.assign(tf.tensor_scatter_nd_sub(self.centers, tf.expand_dims(labels, 1), diff))
          # centers_batch = tf.gather(self.centers, labels)

          return loss * self.factor

  center_loss = Center_loss(classes, factor=1.0, initial_file='./centers.npy')
  callbacks.append(center_loss.save_centers_callback)

  def center_loss_wrapper(center_loss, other_loss):
      def _loss_func(y_true, y_pred):
          embedding = y_pred[:, :512]
          center_loss_v = center_loss(y_true, embedding)
          other_loss_v = other_loss(y_true, y_pred)
          # tf.print("other_loss: %s, cent_loss: %s" % (other_loss_v, center_loss_v))
          return other_loss_v + center_loss_v
      other_loss_name = other_loss.name if 'name' in other_loss.__dict__ else other_loss.__name__
      _loss_func.__name__ = "center_" + other_loss_name
      return _loss_func

  center_softmax_loss = center_loss_wrapper(center_loss, softmax_loss)
  center_arcface_loss = center_loss_wrapper(center_loss, arcface_loss)

  def single_center_loss(labels, prediction):
      embedding = prediction[:, :512]
      norm_logits = prediction[:, 512:]
      return center_loss(labels, embedding)

  with mirrored_strategy.scope():
      # model.compile(optimizer='adamax', loss=single_center_loss, metrics=[logits_accuracy()])
      model.compile(optimizer='adamax', loss=center_softmax_loss, metrics=[logits_accuracy()])
  ```
  ```py
  Epoch 1/200
  >>>> lfw evaluation max accuracy: 0.956833, thresh: 0.628311, overall max accuracy: 0.956833, improved = True
  43216/43216 [==============================] - 9838s 228ms/step - loss: 9.3089 - logits_accuracy: 0.0376 - val_loss: 7.7020 - val_logits_accuracy: 0.1513
  Epoch 2/200
  >>>> lfw evaluation max accuracy: 0.986000, thresh: 0.321373, overall max accuracy: 0.986000, improved = True
  43216/43216 [==============================] - 9979s 231ms/step - loss: 6.3202 - logits_accuracy: 0.4252 - val_loss: 5.1966 - val_logits_accuracy: 0.6057
  Epoch 3/200
  >>>> lfw evaluation max accuracy: 0.991667, thresh: 0.287180, overall max accuracy: 0.991667, improved = True
  43216/43216 [==============================] - 9476s 219ms/step - loss: 4.5633 - logits_accuracy: 0.7169 - val_loss: 3.9777 - val_logits_accuracy: 0.7618
  Epoch 4/200
  >>>> lfw evaluation max accuracy: 0.992333, thresh: 0.250578, overall max accuracy: 0.992333, improved = True
  43216/43216 [==============================] - 9422s 218ms/step - loss: 3.6551 - logits_accuracy: 0.8149 - val_loss: 3.2682 - val_logits_accuracy: 0.8270
  Epoch 5/200
  >>>> lfw evaluation max accuracy: 0.993500, thresh: 0.232111, overall max accuracy: 0.993500, improved = True
  43216/43216 [==============================] - 9379s 217ms/step - loss: 3.1123 - logits_accuracy: 0.8596 - val_loss: 2.8836 - val_logits_accuracy: 0.8516
  Epoch 6/200
  >>>> lfw evaluation max accuracy: 0.992500, thresh: 0.208816, overall max accuracy: 0.993500, improved = False
  43216/43216 [==============================] - 9068s 210ms/step - loss: 2.7492 - logits_accuracy: 0.8851 - val_loss: 2.5630 - val_logits_accuracy: 0.8771                                                         
  Epoch 7/200
  >>>> lfw evaluation max accuracy: 0.992667, thresh: 0.207485, overall max accuracy: 0.993500, improved = False
  43216/43216 [==============================] - 9145s 212ms/step - loss: 2.4826 - logits_accuracy: 0.9015 - val_loss: 2.3668 - val_logits_accuracy: 0.8881
  ```
## Offline Triplet loss train SUB
  ```py
  import pickle
  with open('faces_emore_img_class_shuffle.pkl', 'rb') as ff:
      aa = pickle.load(ff)
  image_names, image_classes = aa['image_names'], aa['image_classes']
  classes = np.max(image_classes) + 1
  print(len(image_names), len(image_classes), classes)
  # 5822653 5822653 85742

  from sklearn.preprocessing import normalize
  from tqdm import tqdm
  import pandas as pd

  class Triplet_datasets:
      def __init__(self, image_names, image_classes, batch_size=128, alpha=0.2, image_per_class=4, max_class=10000):
          self.AUTOTUNE = tf.data.experimental.AUTOTUNE
          self.image_dataframe = pd.DataFrame({'image_names': image_names, "image_classes" : image_classes})
          self.classes = self.image_dataframe.image_classes.unique().shape[0]
          self.image_per_class = image_per_class
          self.max_class = max_class
          self.alpha = alpha
          self.batch_size = batch_size
          self.sub_total = np.ceil(self.max_class * image_per_class / batch_size)
          # self.update_triplet_datasets()

      def update_triplet_datasets(self):
          list_ds = self.prepare_sub_list_dataset()
          anchors, poses, negs = self.mine_triplet_data_pairs(list_ds)
          # self.train_dataset, self.steps_per_epoch = self.gen_triplet_train_dataset(anchors, poses, negs)
          return self.gen_triplet_train_dataset(anchors, poses, negs)

      def image_pick_func(self, df):
          vv = df.image_names.values
          choice_replace = vv.shape[0] < self.image_per_class
          return np.random.choice(vv, self.image_per_class, replace=choice_replace)

      def process_path(self, img_name, img_shape=(112, 112)):
          parts = tf.strings.split(img_name, os.path.sep)[-2]
          label = tf.cast(tf.strings.to_number(parts), tf.int32)
          img = tf.io.read_file(img_name)
          img = tf.image.decode_jpeg(img, channels=3)
          img = tf.image.convert_image_dtype(img, tf.float32)
          img = tf.image.resize(img, img_shape)
          img = tf.image.random_flip_left_right(img)
          return img, label, img_name

      def prepare_sub_list_dataset(self):
          tt = self.image_dataframe.groupby("image_classes").apply(self.image_pick_func)
          sub_tt = tt[np.random.choice(tt.shape[0], self.max_class, replace=False)]
          cc = np.concatenate(sub_tt.values)
          list_ds = tf.data.Dataset.from_tensor_slices(cc)
          list_ds = list_ds.map(self.process_path, num_parallel_calls=self.AUTOTUNE)
          list_ds = list_ds.batch(self.batch_size)
          list_ds = list_ds.prefetch(buffer_size=self.AUTOTUNE)
          return list_ds

      def batch_triplet_image_process(self, anchors, poses, negs):
          anchor_labels = tf.zeros_like(anchors, dtype=tf.float32)
          labels = tf.concat([anchor_labels, anchor_labels + 1, anchor_labels + 2], 0)
          image_names = tf.concat([anchors, poses, negs], 0)
          images = tf.map_fn(lambda xx: self.process_path(xx)[0], image_names, dtype=tf.float32)
          # image_classes = tf.map_fn(lambda xx: tf.strings.split(xx, os.path.sep)[-2], image_names)
          # return images, labels, image_classes
          return images, labels

      def mine_triplet_data_pairs(self, list_ds):
          embs, labels, img_names = [], [], []
          for imgs, label, img_name in tqdm(list_ds, "Embedding", total=self.sub_total):
              emb = basic_model.predict(imgs)
              embs.extend(emb)
              labels.extend(label.numpy())
              img_names.extend(img_name.numpy())
          embs = np.array(embs)
          not_nan_choice = np.isnan(embs).sum(1) == 0
          embs = embs[not_nan_choice]
          embs = normalize(embs)
          labels = np.array(labels)[not_nan_choice]
          img_names = np.array(img_names)[not_nan_choice]

          '''
          where we have same label: pos_idx --> [10, 11, 12, 13]
          image names: pose_imgs --> ['a', 'b', 'c', 'd']
          anchor <--> pos: {10: [11, 12, 13], 11: [12, 13], 12: [13]}
          distance of anchor and pos: stack_pos_dists -->
              [[10, 11], [10, 12], [10, 13], [11, 12], [11, 13], [12, 13]]
          anchors image names: stack_anchor_name --> ['a', 'a', 'a', 'b', 'b', 'c']
          pos image names: stack_pos_name --> ['b', 'c', 'd', 'c', 'd', 'd']
          distance between anchor and all others: stack_dists -->
              [d(10), d(10), d(10), d(11), d(11), d(12)]
          distance between pos and neg for all anchor: neg_pos_dists -->
              [d([10, 11]) - d(10), d([10, 12]) - d(10), d([10, 13]) - d(10),
               d([11, 12]) - d(11), d([11, 13]) - d(11),
               d([12, 13]) - d(12)]
          valid pos indexes: neg_valid_x --> [0, 0, 0, 1, 1, 1, 2, 5, 5, 5]
          valid neg indexss: neg_valid_y --> [1022, 312, 3452, 6184, 294, 18562, 82175, 9945, 755, 8546]
          unique valid pos indexes: valid_pos --> [0, 1, 2, 5]
          random valid neg indexs in each pos: valid_neg --> [1022, 294, 82175, 8546]
          anchor names: stack_anchor_name[valid_pos] --> ['a', 'a', 'a', 'c']
          pos names: stack_pos_name[valid_pos] --> ['b', 'c', 'd', 'd']
          '''
          anchors, poses, negs = [], [], []
          for label in tqdm(np.unique(labels), "Mining triplet pairs"):
          # for label in np.unique(labels):
              pos_idx = np.where(labels == label)[0]
              pos_imgs = img_names[pos_idx]
              total = pos_idx.shape[0]
              pos_embs = embs[pos_idx[:-1]]
              dists = np.dot(pos_embs, embs.T)
              pos_dists = [dists[id, pos_idx[id + 1:]] for id in range(total - 1)]
              stack_pos_dists = np.expand_dims(np.hstack(pos_dists), -1)

              elem_repeats = np.arange(1, total)[::-1]
              stack_anchor_name = pos_imgs[:-1].repeat(elem_repeats, 0)
              stack_pos_name = np.hstack([pos_imgs[ii:] for ii in range(1, total)])
              stack_dists = dists.repeat(elem_repeats, 0)

              neg_pos_dists = stack_pos_dists - stack_dists - self.alpha
              neg_pos_dists[:, pos_idx] = 1
              neg_valid_x, neg_valid_y = np.where(neg_pos_dists < 0)

              if len(neg_valid_x) > 0:
                  valid_pos = np.unique(neg_valid_x)
                  valid_neg = [np.random.choice(neg_valid_y[neg_valid_x == ii]) for ii in valid_pos]
                  anchors.extend(stack_anchor_name[valid_pos])
                  poses.extend(stack_pos_name[valid_pos])
                  negs.extend(img_names[valid_neg])
                  # self.minning_print_func(pos_imgs, valid_pos, valid_neg, stack_anchor_name, stack_pos_name, labels, stack_dists)
          print(">>>> %d triplets found." % (len(anchors)))
          return anchors, poses, negs

      def gen_triplet_train_dataset(self, anchors, poses, negs):
          num_triplets = len(anchors)
          train_dataset = tf.data.Dataset.from_tensor_slices((anchors, poses, negs))
          train_dataset = train_dataset.shuffle(num_triplets + 1)
          train_dataset = train_dataset.batch(self.batch_size)
          train_dataset = train_dataset.map(self.batch_triplet_image_process, num_parallel_calls=self.AUTOTUNE)
          train_dataset = train_dataset.prefetch(buffer_size=self.AUTOTUNE)
          steps_per_epoch = np.ceil(num_triplets / self.batch_size)
          return train_dataset, steps_per_epoch

      def minning_print_func(self, pose_imgs, valid_pos, valid_neg, stack_anchor_name, stack_pos_name, labels, stack_dists):
          img2idx = dict(zip(pose_imgs, range(len(pose_imgs))))
          valid_anchor_idx = [img2idx[stack_anchor_name[ii]] for ii in valid_pos]
          valid_pos_idx = [img2idx[stack_pos_name[ii]] for ii in valid_pos]
          print("anchor: %s" % (list(zip(valid_anchor_idx, labels[pos_idx[valid_anchor_idx]]))))
          print("pos: %s" % (list(zip(valid_pos_idx, labels[pos_idx[valid_pos_idx]]))))
          print("neg: %s" % (labels[valid_neg]))
          print("pos dists: %s" % ([stack_dists[ii, pos_idx[jj]] for ii, jj in zip(valid_pos, valid_pos_idx)]))
          print("neg dists: %s" % ([stack_dists[ii, jj] for ii, jj in zip(valid_pos, valid_neg)]))
          print()

  def triplet_loss(labels, embeddings, alpha=0.2):
      labels = tf.squeeze(labels)
      labels.set_shape([None])
      anchor_emb = tf.nn.l2_normalize(embeddings[labels == 0], 1)
      pos_emb = tf.nn.l2_normalize(embeddings[labels == 1], 1)
      neg_emb = tf.nn.l2_normalize(embeddings[labels == 2], 1)
      pos_dist = tf.reduce_sum(tf.multiply(anchor_emb, pos_emb), -1)
      neg_dist = tf.reduce_sum(tf.multiply(anchor_emb, neg_emb), -1)
      basic_loss = neg_dist - pos_dist + alpha
      return tf.reduce_mean(tf.maximum(basic_loss, 0.0), axis=0)

  basic_model.compile(optimizer='adamax', loss=triplet_loss)
  triplet_datasets = Triplet_datasets(image_names, image_classes, image_per_class=5, max_class=10000)
  for epoch in range(100):
      train_dataset, steps_per_epoch = triplet_datasets.update_triplet_datasets()
      basic_model.fit(train_dataset, epochs=1, verbose=1, callbacks=callbacks, steps_per_epoch=steps_per_epoch, initial_epoch=epoch, use_multiprocessing=True, workers=4)
  ```
  ```py
  def mine_triplet_data_pairs(embs, labels, img_names, alpha=0.2):
      anchors, poses, negs = [], [], []
      for idx, (emb, label) in enumerate(zip(embs, labels)):
          dist = np.dot(emb, embs.T)
          pos_indexes = np.where(labels == label)[0]
          pos_indexes = pos_indexes[pos_indexes > idx]
          neg_indxes = np.where(labels != label)[0]
          for pos in pos_indexes:
              if pos == idx:
                  continue
              pos_dist = dist[pos]
              neg_valid = neg_indxes[pos_dist - dist[neg_indxes] < alpha]
              if neg_valid.shape[0] == 0:
                  continue
              neg_random = np.random.choice(neg_valid)
              anchors.append(img_names[idx])
              poses.append(img_names[pos])
              negs.append(img_names[neg_random])
              print("label: %d, pos: %d, %f, neg: %d, %f" % (label, labels[pos], dist[pos], labels[neg_random], dist[neg_random]))
      return anchors, poses, negs
  ```
## tf-insightface train
  - Arcface loss
    ```py
    epoch: 25, step: 60651, loss = 15.109766960144043, logit_loss = 15.109766960144043, center_loss = 0
    epoch: 25, step: 60652, loss = 17.565662384033203, logit_loss = 17.565662384033203, center_loss = 0
    Saving checkpoint for epoch 25 at /home/tdtest/workspace/tf_insightface/recognition/mymodel-26
    Time taken for epoch 25 is 8373.555536031723 sec
    ```
  - Arcface center loss
    ```py
    epoch: 0, step: 60652, loss = 11.264640808105469, logit_loss = 11.262413024902344, center_loss = 0.002227420685812831
    Saving checkpoint for epoch 0 at /home/tdtest/workspace/tf_insightface/recognition/mymodel-1
    Time taken for epoch 0 is 8373.187638521194 sec
    ```
## FUNC
  - **tf.compat.v1.scatter_sub** 将 `ref` 中 `indices` 指定位置的值减去 `updates`，会同步更新 `ref`
    ```py
    scatter_sub(ref, indices, updates, use_locking=False, name=None)
    ```
    ```py
    ref = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8],dtype = tf.int32)
    indices = tf.constant([4, 3, 1, 7],dtype = tf.int32)
    updates = tf.constant([9, 10, 11, 12],dtype = tf.int32)
    print(tf.compat.v1.scatter_sub(ref, indices, updates).numpy())
    # [ 1 -9  3 -6 -4  6  7 -4]
    print(ref.numpy())
    [ 1 -9  3 -6 -4  6  7 -4]
    ```
  - **tf.tensor_scatter_nd_sub** 多维数据的 `tf.compat.v1.scatter_sub`
    ```py
    tensor = tf.ones([8], dtype=tf.int32)
    indices = tf.constant([[4], [3], [1] ,[7]])
    updates = tf.constant([9, 10, 11, 12])
    print(tf.tensor_scatter_nd_sub(tensor, indices, updates).numpy())
    # [ 1 -9  3 -6 -4  6  7 -4]
    ```
  - **tf.gather** 根据 `indices` 切片选取 `params` 中的值
    ```py
    gather_v2(params, indices, validate_indices=None, axis=None, batch_dims=0, name=None)
    ```
    ```py
    print(tf.gather([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [0, 1, 0]).numpy())
    # [[1 2 3] [4 5 6] [1 2 3]]
    ```
  - **l2 normalize**
    ```py
    aa = [x1, x2]
    bb = [[y1, y2], [y3, y4]]

    ''' tf.nn.l2_normalize(tf.matmul(aa, bb)) '''
    tf.matmul(aa, bb) = [x1 * y1 + x2 * y3, x1 * y2 + x2 * y4]
    tf.nn.l2_normalize(tf.matmul(aa, bb)) = [
        (x1 * y1 + x2 * y3) / sqrt((x1 * y1 + x2 * y3) ** 2 + (x1 * y2 + x2 * y4) ** 2)
        (x1 * y2 + x2 * y4) / sqrt((x1 * y1 + x2 * y3) ** 2 + (x1 * y2 + x2 * y4) ** 2)
    ]

    ''' tf.matmul(tf.nn.l2_normalize(aa), tf.nn.l2_normalize(bb)) '''
    tf.nn.l2_normalize(aa) = [x1 / sqrt(x1 ** 2 + x2 ** 2), x2 / sqrt(x1 ** 2 + x2 ** 2)]
    tf.nn.l2_normalize(bb) = [[y1 / sqrt(y1 ** 2 + y3 ** 2), y2 / sqrt(y2 ** 2 + y4 ** 2)],
                              [y3 / sqrt(y1 ** 2 + y3 ** 2), y4 / sqrt(y2 ** 2 + y4 ** 2)]]
    tf.matmul(tf.nn.l2_normalize(aa), tf.nn.l2_normalize(bb)) = [
        (x1 * y1 + x2 * y3) / sqrt((x1 ** 2 + x2 ** 2) * (y1 ** 2 + y3 ** 2)),
        (x1 * y2 + x2 * y4) / sqrt((x1 ** 2 + x2 ** 2) * (y2 ** 2 + y4 ** 2))
    ]
    ```
    ```py
    aa = tf.convert_to_tensor([[1, 2]], dtype='float32')
    bb = tf.convert_to_tensor(np.arange(4).reshape(2, 2), dtype='float32')
    print(aa.numpy())
    # [[1. 2.]]
    print(bb.numpy())
    # [[0. 1.] [2. 3.]]

    print(tf.matmul(aa, bb).numpy())
    # [[4. 7.]]
    print(tf.nn.l2_normalize(tf.matmul(aa, bb), axis=1).numpy())
    # [[0.49613893 0.8682431 ]]

    print(tf.nn.l2_normalize(aa, 1).numpy())
    # [[0.4472136 0.8944272]]
    print(tf.nn.l2_normalize(bb, 0).numpy())
    # [[0.         0.31622776] [1.         0.94868326]]
    print(tf.matmul(tf.nn.l2_normalize(aa, 1), tf.nn.l2_normalize(bb, 0)).numpy())
    # [[0.8944272  0.98994946]]
    ```
***
- [TensorFlow Addons Losses: TripletSemiHardLoss](https://www.tensorflow.org/addons/tutorials/losses_triplet)
- [TensorFlow Addons Layers: WeightNormalization](https://www.tensorflow.org/addons/tutorials/layers_weightnormalization)
