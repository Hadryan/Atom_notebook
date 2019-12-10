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
## Loading data and basic model
  ```py
  ''' flow_from_directory '''
  import glob2
  from keras.preprocessing.image import ImageDataGenerator
  from tensorflow.keras import layers
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

  path = '/datasets/faces_emore_112x112_folders/'
  image_gen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
  train_data_gen = image_gen.flow_from_directory(path, class_mode='categorical', target_size=(112, 112), batch_size=96, seed=1, subset='training')
  val_data_gen = image_gen.flow_from_directory(path, class_mode='categorical', target_size=(112, 112), batch_size=96, seed=1, subset='validation')
  classes = train_data_gen.num_classes

  ''' flow_from_dataframe '''
  path = '/home/tdtest/workspace/insightface-master/faces_emore_img/faces_emore_img.foo'
  with open(path, 'r') as ff:
      image_names = np.array([ii.strip() for ii in ff.readlines()])
  image_classes = np.array([os.path.basename(os.path.dirname(ii)) for ii in image_names])
  print(image_names.shape, image_classes.shape)

  data_df = pd.DataFrame({"image_names": image_names, "image_classes": image_classes})
  image_gen = ImageDataGenerator(rescale=1./255, validation_split=0.1)
  train_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col="image_classes", class_mode='categorical', target_size=(112, 112), batch_size=128, seed=1, subset='training', validate_filenames=False)
  # Found 5240388 non-validated image filenames belonging to 85742 classes.
  val_data_gen = image_gen.flow_from_dataframe(data_df, directory=None, x_col='image_names', y_col="image_classes", class_mode='categorical', target_size=(112, 112), batch_size=128, seed=1, subset='validation', validate_filenames=False)
  # Found 582265 non-validated image filenames belonging to 85742 classes.

  classes = data_df.image_classes.unique().shape[0]

  ''' Basic model '''
  # xx = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
  xx = tf.keras.applications.ResNet50V2(input_shape=(112, 112, 3), include_top=False, weights='imagenet')
  xx.trainable = True
  inputs = layers.Input(shape=(112, 112, 3))
  nn = xx(inputs)
  nn = layers.GlobalAveragePooling2D()(nn)
  nn = layers.BatchNormalization()(nn)
  nn = layers.Dropout(0.1)(nn)
  embedding = layers.Dense(512, name='embedding')(nn)
  # logits = layers.Dense(classes, activation='softmax', name='logits')(embedding)
  logits = layers.Dense(classes, name='logits')(embedding)
  norm_logits = layers.BatchNormalization(name='norm_logits')(logits)
  ```
## Softmax train
  ```py
  model = keras.models.Model(inputs, norm_logits)
  model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
  model.summary()

  reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=5, verbose=1)
  model_checkpoint = ModelCheckpoint("./keras_checkpoints_res_soft", 'val_loss', verbose=1, save_best_only=True)
  callbacks = [model_checkpoint, reduce_lr]
  hist = model.fit_generator(train_data_gen, validation_data=val_data_gen, epochs=200, verbose=1, callbacks=callbacks)
  ```
## Arcface loss train
  ```py
  # arc_loss = arcface_loss(labels, embedding, norm_logits)
  def arcface_loss(labels, embedding, norm_logits, margin1=1.0, margin2=0.2, margin3=0.3, scale=64.0):
      # embedding = prediction[:, :512]
      # norm_logits = prediction[:, 512:]
      norm_x = tf.norm(embedding, axis=1, keepdims=True)
      cos_theta = norm_logits / norm_x
      theta = tf.acos(cos_theta)
      zeros = tf.zeros_like(labels)
      cond = tf.where(tf.greater(theta * margin1 + margin3, np.pi), zeros, labels)
      cond = tf.cast(cond, dtype=tf.bool)
      m1_theta_plus_m3 = tf.where(cond, theta * margin1 + margin3, theta)
      cos_m1_theta_plus_m3 = tf.cos(m1_theta_plus_m3)
      arcface_logits = tf.where(cond, cos_m1_theta_plus_m3 - margin2, cos_m1_theta_plus_m3) * scale
      labels = tf.argmax(labels, 1)
      # print(">>>> labels", labels)
      # print(">>>> arcface_logits", arcface_logits)
      return tf.keras.losses.sparse_categorical_crossentropy(labels, arcface_logits, from_logits=True)

  def single_arcface_loss(labels, prediction, center_loss_factor=1.0):
      embedding = prediction[:, :512]
      norm_logits = prediction[:, 512:]
      return arcface_loss(labels, embedding, norm_logits)

  def logits_accuracy(y_true, y_pred):
      norm_logits = y_pred[:, 512:]
      return keras.metrics.categorical_accuracy(y_true, norm_logits)

  concate = layers.concatenate([embedding, norm_logits], name='concate')
  model = keras.models.Model(inputs, concate)
  model.compile(optimizer='adam', loss=single_arcface_loss, metrics=[logits_accuracy])
  model.summary()

  reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=5, verbose=1)
  model_checkpoint = ModelCheckpoint("./keras_checkpoints_res_arcface", 'val_loss', verbose=1, save_best_only=True)
  callbacks = [model_checkpoint, reduce_lr]
  hist = model.fit_generator(train_data_gen, validation_data=val_data_gen, epochs=200, verbose=1, callbacks=callbacks)
  ```
## Center loss
  ```py
  import functools

  def _center_loss_func(labels, features, alpha, num_classes, centers, feature_dim):      
      # assert feature_dim == features.get_shape()[1]    
      # labels = K.reshape(labels, [-1])
      # labels = tf.to_int32(labels)
      # centers_batch = tf.gather(centers, labels)
      # global centers
      # print("centers: %s, sum: %s" % (centers, tf.reduce_sum(centers)))
      # labels = tf.cast(labels, tf.int32)
      labels = tf.argmax(labels, axis=1)
      centers_batch = tf.gather(centers, labels)
      diff = (1 - alpha) * (centers_batch - features)
      # print(centers_batch.shape, centers.shape, labels.shape, diff.shape)
      # centers = tf.compat.v1.scatter_sub(centers, labels, diff)
      centers.assign(tf.tensor_scatter_nd_sub(centers, tf.expand_dims(labels, 1), diff))
      # centers_batch = tf.gather(centers, labels)
      loss = tf.reduce_mean(tf.square(features - centers_batch))
      return loss

  def get_center_loss(num_classes, feature_dim=512, alpha=0.9):
      """Center loss based on the paper "A Discriminative
         Feature Learning Approach for Deep Face Recognition"
         (http://ydwen.github.io/papers/WenECCV16.pdf)
      """
      # Each output layer use one independed center: scope/centers
      centers = tf.Variable(tf.zeros([num_classes, feature_dim]))
      @functools.wraps(_center_loss_func)
      def center_loss(y_true, y_pred):
          return _center_loss_func(y_true, y_pred, alpha, num_classes, centers, feature_dim)
      return center_loss

  center_loss = get_center_loss(num_classes=classes)
  def center_arcface_loss(labels, prediction, center_loss_factor=1.0):
      embedding = prediction[:, :512]
      norm_logits = prediction[:, 512:]
      arc_loss = arcface_loss(labels, embedding, norm_logits)
      if center_loss_factor > 0:
          cent_loss = center_loss(labels, embedding)
          print("arcface_loss = %s, cent_loss = %s" % (arc_loss, cent_loss))
          return arc_loss + cent_loss * center_loss_factor
      else:
          return arc_loss

  def single_center_loss(labels, prediction, center_loss_factor=1.0):
      embedding = prediction[:, :512]
      norm_logits = prediction[:, 512:]
      return center_loss(labels, embedding)

  concate = layers.concatenate([embedding, norm_logits], name='concate')
  model = keras.models.Model(inputs, concate)
  # model.compile(optimizer='adam', loss=single_center_loss)
  model.compile(optimizer='adam', loss=center_arcface_loss, metrics=[logits_accuracy])
  model.summary()

  reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=5, verbose=1)
  model_checkpoint = ModelCheckpoint("./keras_checkpoints_res_arcface", 'val_loss', verbose=1, save_best_only=True)
  callbacks = [model_checkpoint, reduce_lr]
  hist = model.fit_generator(train_data_gen, validation_data=val_data_gen, epochs=200, verbose=1, callbacks=callbacks)
  ```
## Triplet loss train
```py

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
***

# PLATE MASK
## skimage 角点检测
  ```py
  def plate_mask_corner(polygon_points, image_shape=(128, 384), plot_image=True):
      mask = polygon2mask(image_shape, polygon_points)
      cc = corner_peaks(corner_harris(mask, method='eps'), min_distance=1, num_peaks=4, exclude_border=False)
      if plot_image:
          fig = plt.figure()
          plt.imshow(mask)
          plt.scatter(cc[:, 1], cc[:, 0])
      return cc
  ```
## 取巻积后所有最小值点
  ```py
  def plate_mask_to_scatter(polygon_points, conv_kernel_size=[9, 9], corner_min=None, corner_max=None, min_dist=100, image_shape=(128, 384), plot_image=True):
      if corner_min == None:
          corner_min = conv_kernel_size[0]
      if corner_max == None:
          corner_max = (conv_kernel_size[0] * conv_kernel_size[1]) // 2
      mask = polygon2mask(image_shape, polygon_points)
      cc = convolve2d(mask, np.ones(conv_kernel_size), mode='same')
      bb = np.where(mask, cc, np.zeros_like(cc))
      # print(pd.value_counts(bb.flatten()).sort_index())
      # print(np.logical_and(bb <corner_ max, bb > corner_min).sum())
      tt = np.where(np.logical_and(bb < corner_max, bb > corner_min))

      ''' Group by dists '''
      tt_coor = np.array(list(zip(tt[0].tolist(), tt[1].tolist())))
      dd = [np.expand_dims(tt_coor[0], 0)]
      base = [tt_coor[0]]
      for ii in tt_coor[1:]:
          dists = ((ii - base) ** 2).sum(-1)
          if np.any(dists < 100):
              id = dists.argmin()
              dd[id] = np.vstack([dd[id], ii])
          else:
              base.append(ii)
              dd.append(np.expand_dims(ii, 0))

      ''' Return min conv value of each group '''
      rr = [ii[np.argmin([bb[ixx, iyy] for ixx, iyy in ii])] for ii in dd]
      rr = np.vstack(rr)
      if plot_image:
          fig = plt.figure()
          plt.imshow(mask)
          plt.scatter(rr[:, 1], rr[:, 0])
      return rr
  ```
## 巻积后按区域划分后的四个最小值点
  ```py
  def plate_mask_to_scatter_2(polygon_points, conv_kernel_size=[9, 9], image_shape=(128, 384), plot_image=True):
      mask = polygon2mask(image_shape, polygon_points)
      cc = convolve2d(mask, np.ones(conv_kernel_size), mode='same')
      bb = np.where(mask, cc, np.zeros_like(cc))
      # print(pd.value_counts(bb.flatten()))

      half_x = image_shape[1] // 2 + 1
      half_y = image_shape[0] // 2 + 1
      rr = []
      for ixx in [0, half_x]:
          for iyy in [0, half_y]:
              # print("ixx = %d, iyy = %d" % (ixx, iyy))
              sub_bb = bb[iyy : iyy + half_y, ixx : ixx + half_x]
              sub_bb = np.where(sub_bb == 0, np.ones_like(sub_bb) * 255, sub_bb)
              tt_y, tt_x = np.where(sub_bb == sub_bb.min())
              point = [tt_y.mean() + iyy, tt_x.mean() + ixx]
              print("point = %s, sub_bb.min = %d" % (point, sub_bb.min()))
              rr.append(point)
      rr = np.array(rr)
      if plot_image:
          fig = plt.figure()
          plt.imshow(mask)
          plt.scatter(rr[:, 1], rr[:, 0])
      return rr
  ```
## 测试
  ```py
  def coord_sort(coord):
      coord_sort_1 = sorted(coord.tolist(), key=lambda ii: ii[0] + ii[1])
      coord_sort_2 = sorted(coord_sort_1[:2], key=lambda ii: ii[0])
      coord_sort_2.extend(sorted(coord_sort_1[2:], key=lambda ii: ii[0]))
      return np.array(coord_sort_2)

  def test_with_masks(path, test_func, test_num=None):
      dists = []
      pps = []
      ccs = []
      tests = os.listdir(path)
      if test_num:
          tests = tests[:test_num]
      for ii in tests:
          pp = np.array(os.path.splitext(ii)[0].split('_')[1:]).astype('float').reshape(-1, 2)[:, ::-1]
          rr = test_func(pp)
          pps.append(pp)
          ccs.append(rr)

          pp_sort = coord_sort(pp)
          rr_sort = coord_sort(rr)
          dd = ((pp_sort - rr_sort) ** 2).sum()
          dists.append(dd)
          print('pp = %s, rr = %s, dist = %.2f' % (pp.tolist(), rr.tolist(), dd))
      dda = np.array(dists)
      ppa = np.array(pps)
      cca = np.array(ccs)

      print("Max 10 dist: %s" % (np.sort(dda)[-10:]))
      return ppa, cca, dda

  test_func = lambda pp: plate_mask_to_scatter(pp, plot_image=False)
  path = './mask_128_384/'
  ppa, cca, dda = test_with_masks(path, test_func)
  # Max 10 dist: [13.36 13.36 14.02 14.02 14.64 14.7  15.4  16.08 16.08 16.14]

  dda.max()
  np.sort(dda)[-10:]
  np.sort(dda)[-50:]
  print(ppa[np.argsort(dda)[-10:]])
  # [[[13.2, 90.2], [59.8, 30.6], [114.8, 293.8], [68.2, 353.4]],
  #  [[13.2, 90.2], [59.8, 30.6], [114.8, 293.8], [68.2, 353.4]],
  #  [[15.1, 88.1], [56.8, 37.5], [112.9, 295.9], [71.2, 346.5]],
  #  [[15.1, 88.1], [56.8, 37.5], [112.9, 295.9], [71.2, 346.5]],
  #  [[10.9, 85.4], [60.8, 25.9], [117.1, 298.6], [67.2, 358.1]],
  #  [[11.9, 80.3], [58.9, 26.8], [116.1, 303.7], [69.1, 357.2]],
  #  [[14.1, 74.1], [53.8, 30.2], [113.9, 309.9], [74.2, 353.8]],
  #  [[12.9, 32.1], [67.6, 83.6], [115.1, 351.9], [60.4, 300.4]],
  #  [[12.9, 32.1], [67.6, 83.6], [115.1, 351.9], [60.4, 300.4]],
  #  [[14.1, 85.1], [60.8, 32.1], [113.9, 298.9], [67.2, 351.9]]]
  ```
***
