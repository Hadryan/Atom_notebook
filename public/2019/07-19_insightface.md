# ___2019 - 07 - 19 Insightface___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2019 - 07 - 19 Insightface___](#2019-07-19-insightface)
  - [目录](#目录)
  - [Related projects](#related-projects)
  - [Insightface MXNet 模型使用](#insightface-mxnet-模型使用)
  	- [MXNet](#mxnet)
  	- [模型加载与特征提取](#模型加载与特征提取)
  	- [模型训练](#模型训练)
  - [Highest Accuracy](#highest-accuracy)
  	- [Deepinsight Accuracy](#deepinsight-accuracy)
  	- [Mobilenet 256 Embeddings](#mobilenet-256-embeddings)
  	- [Mobilenet 512 embeddings](#mobilenet-512-embeddings)
  	- [Vargface](#vargface)
  - [MTCNN](#mtcnn)
  	- [Testing function](#testing-function)
  	- [facenet mtcnn](#facenet-mtcnn)
  	- [insightface mtcnn](#insightface-mtcnn)
  	- [MTCNN-Tensorflow](#mtcnn-tensorflow)
  	- [mtcnn.MTCNN](#mtcnnmtcnn)
  	- [caffe MTCNN](#caffe-mtcnn)
  	- [人脸对齐](#人脸对齐)
  - [Docker 封装](#docker-封装)
  - [视频中识别人脸保存](#视频中识别人脸保存)

  <!-- /TOC -->
***

# Insightface MXNet 模型使用
## Insightface
  - [deepinsight/insightface](https://github.com/deepinsight/insightface)
## MXNet
  ```sh
  # 安装 cuda-10-0 对应的 mxnet 版本
  pip install mxnet-cu100
  ```
## 模型加载与特征提取
  ```py
  # cd ~/workspace/face_recognition_collection/insightface/deploy
  import face_model
  import argparse
  import cv2
  import os

  home_path = os.environ.get("HOME")
  args = argparse.ArgumentParser().parse_args([])
  args.image_size = '112,112'
  args.model = os.path.join(home_path, 'workspace/models/insightface_mxnet_model/model-r100-ii/model,0')
  args.ga_model = os.path.join(home_path, "workspace/models/insightface_mxnet_model/gamodel-r50/model,0")
  args.gpu = 0
  args.det = 0
  args.flip = 0
  args.threshold = 1.05
  model = face_model.FaceModel(args)

  img = cv2.imread('./Tom_Hanks_54745.png')
  bbox, points = model.detector.detect_face(img, det_type = model.args.det)

  import matplotlib.pyplot as plt
  aa = bbox[0, :4].astype(np.int)
  bb = points[0].astype(np.int).reshape(2, 5).T

  # landmarks
  plt.imshow(img)
  plt.scatter(ii[:, 0], ii[:, 1])

  # cropped image
  plt.imshow(img[aa[1]:aa[3], aa[0]:aa[2], :])

  # By face_preprocess.preprocess
  cd ../src/common
  import face_preprocess
  cc = face_preprocess.preprocess(img, aa, bb, image_size='112,112')
  plt.imshow(cc)

  # export image feature, OUT OF MEMORY
  emb = model.get_feature(model.get_input(img))
  ```
## 模型训练
  ```sh
  # My 2-stage pipeline:
  # Train softmax with lr=0.1 for 120K iterations.
  # LRSTEPS='240000,360000,440000'
  CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 0 --prefix "$PREFIX" --per-batch-size 128 --lr-steps "$LRSTEPS" --margin-s 32.0 --margin-m 0.1 --ckpt 2 --emb-size 128 --fc7-wd-mult 10.0 --wd 0.00004 --max-steps 140002

  # Switch to ArcFace loss to do normal training with '100K,140K,160K' iterations.
  # LRSTEPS='100000,140000,160000'
  CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --data-dir $DATA_DIR --network "$NETWORK" --loss-type 4 --prefix "$PREFIX" --per-batch-size 128 --lr-steps "$LRSTEPS" --margin-s 64.0 --margin-m 0.5 --ckpt 1 --emb-size 128 --fc7-wd-mult 10.0 --wd 0.00004 --pretrained '../models2/model-y1-test/model,70'

  # training dataset: ms1m
  # LFW: 99.50, CFP_FP: 88.94, AgeDB30: 95.91
  ```
  **config.py 配置文件**
  ```py
  # config.py +78
  network.m1.emb_size = 512

  # config.py +117
  dataset.emore.dataset_path = '/datasets/faces_emore'
  dataset.emore.num_classes = 85742
  ...
  dataset.glint.dataset_path = '/datasets/faces_glint'
  dataset.glint.num_classes = 180855

  # config.py +147
  loss.arcface = edict()
  loss.arcface.loss_name = 'margin_softmax'
  loss.arcface.loss_s = 64.0
  loss.arcface.loss_m1 = 1.0
  loss.arcface.loss_m2 = 0.5
  loss.arcface.loss_m3 = 0.0
  ```
  **mobilenet 模型训练**
  ```sh
  export MXNET_ENABLE_GPU_P2P=0

  # arcface train
  CUDA_VISIBLE_DEVICES='0,1' python -u train_parall.py --network m1 --loss arcface --dataset emore --per-batch-size 96

  # triplet fine-tune
  CUDA_VISIBLE_DEVICES='0,1' python -u train.py --network m1 --loss arcface --dataset emore --per-batch-size 96 --pretrained ./models/m1-arcface-emore/model --lr 0.0001
  CUDA_VISIBLE_DEVICES='1' python -u train.py --network m1 --loss triplet --dataset emore --per-batch-size 150 --pretrained ./models/m1-triplet-emore_97083/model --lr 0.0001 --lr-steps '1000,100000,160000,220000,280000,340000'
  CUDA_VISIBLE_DEVICES='0' python -u train.py --network m1 --loss triplet --dataset glint --per-batch-size 150 --pretrained ./models/m1-triplet-emore_290445/model --pretrained-epoch 602 --lr 0.0001 --lr-steps '1000,100000,160000,220000,280000,340000'
  ```
  **Vargfacenet 模型训练**
  ```sh
  # Vargfacenet
  CUDA_VISIBLE_DEVICES='0,1' python3 -u train_parall.py --network vargfacenet --loss softmax --dataset emore --per-batch-size 96
  CUDA_VISIBLE_DEVICES='1' python3 -u train.py --network vargfacenet --loss arcface --dataset glint --per-batch-size 150 --pretrained ./models/vargfacenet-softmax-emore/model --pretrained-epoch 166 --lr 0.0001 --lr-steps '100000,160000,220000,280000,340000'
  ```
## 代码分析
  - **config**
    ```py
    # config.py
    config.bn_mom = 0.9
    config.net_output = 'E'
    config.ce_loss = True
    config.fc7_lr_mult = 1.0
    config.fc7_wd_mult = 1.0
    config.fc7_no_bias = False

    network.m1.net_name = 'fmobilenet'
    network.m1.emb_size = 256
    network.m1.net_output = 'GDC'
    network.m1.net_multiplier = 1.0

    network.vargfacenet.net_name = 'vargfacenet'
    network.vargfacenet.net_multiplier = 1.25
    network.vargfacenet.emb_size = 512
    network.vargfacenet.net_output='J'

    loss.arcface.loss_name = 'margin_softmax'
    loss.arcface.loss_s = 64.0
    loss.arcface.loss_m1 = 1.0
    loss.arcface.loss_m2 = 0.5
    loss.arcface.loss_m3 = 0.0

    loss.triplet.loss_name = 'triplet'
    loss.triplet.images_per_identity = 5
    loss.triplet.triplet_alpha = 0.3
    loss.triplet.triplet_bag_size = 7200
    loss.triplet.triplet_max_ap = 0.0
    loss.triplet.per_batch_size = 60
    loss.triplet.lr = 0.05
    ```
  - **symbol**
    ```py
    # symbol_utils.py
    def Linear(data, num_filter=1, kernel=(1, 1), stride=(1, 1), pad=(0, 0), num_group=1, name=None, suffix=''):
        conv = mx.sym.Convolution(data=data, num_filter=num_filter, kernel=kernel, num_group=num_group, stride=stride, pad=pad, no_bias=True, name='%s%s_conv2d' %(name, suffix))
        bn = mx.sym.BatchNorm(data=conv, name='%s%s_batchnorm' %(name, suffix), fix_gamma=False,momentum=bn_mom)    
        return bn

    def get_fc1(last_conv, num_classes, fc_type, input_channel=512):
      elif fc_type=='E':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        body = mx.symbol.Dropout(data=body, p=0.4)
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
      elif fc_type=='FC':
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
      elif fc_type=="GDC": #mobilefacenet_v1
        conv_6_dw = Linear(last_conv, num_filter=input_channel, num_group=input_channel, kernel=(7,7), pad=(0, 0), stride=(1, 1), name="conv_6dw7_7")  
        conv_6_f = mx.sym.FullyConnected(data=conv_6_dw, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=conv_6_f, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
      elif fc_type=='J':
        fc1 = mx.sym.FullyConnected(data=body, num_hidden=num_classes, name='pre_fc1')
        fc1 = mx.sym.BatchNorm(data=fc1, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='fc1')
    ```
  - **fmobilenet**
    ```py
    # fmobilenet.py
    def get_symbol():
      conv_14 = Conv(conv_14_dw, num_filter=bf*32, kernel=(1, 1), pad=(0, 0), stride=(1, 1), name="conv_14") # 7/7
      body = conv_14
      fc1 = symbol_utils.get_fc1(body, num_classes, fc_type)
      return fc1
    ```
  - **train_parall**
    ```py
    # train_parall.py
    def get_symbol_embedding():
      embedding = eval(config.net_name).get_symbol()
      all_label = mx.symbol.Variable('softmax_label')
      #embedding = mx.symbol.BlockGrad(embedding)
      all_label = mx.symbol.BlockGrad(all_label)
      out_list = [embedding, all_label]
      out = mx.symbol.Group(out_list)
      return out

    def get_symbol_arcface(args):
      embedding = mx.symbol.Variable('data')
      all_label = mx.symbol.Variable('softmax_label')
      gt_label = all_label
      is_softmax = True
      #print('call get_sym_arcface with', args, config)
      _weight = mx.symbol.Variable("fc7_%d_weight"%args._ctxid, shape=(args.ctx_num_classes, config.emb_size),
          lr_mult=config.fc7_lr_mult, wd_mult=config.fc7_wd_mult)
      if config.loss_name=='softmax': #softmax
        fc7 = mx.sym.FullyConnected(data=embedding, weight = _weight, no_bias = True, num_hidden=args.ctx_num_classes, name='fc7_%d'%args._ctxid)
      elif config.loss_name=='margin_softmax':
        _weight = mx.symbol.L2Normalization(_weight, mode='instance')
        nembedding = mx.symbol.L2Normalization(embedding, mode='instance', name='fc1n_%d'%args._ctxid)
        fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.ctx_num_classes, name='fc7_%d'%args._ctxid)
        if config.loss_m1!=1.0 or config.loss_m2!=0.0 or config.loss_m3!=0.0:
          gt_one_hot = mx.sym.one_hot(gt_label, depth = args.ctx_num_classes, on_value = 1.0, off_value = 0.0)
          if config.loss_m1==1.0 and config.loss_m2==0.0:
            _one_hot = gt_one_hot*args.margin_b
            fc7 = fc7-_one_hot
          else:
            fc7_onehot = fc7 * gt_one_hot
            cos_t = fc7_onehot
            t = mx.sym.arccos(cos_t)
            if config.loss_m1!=1.0:
              t = t*config.loss_m1
            if config.loss_m2!=0.0:
              t = t+config.loss_m2
            margin_cos = mx.sym.cos(t)
            if config.loss_m3!=0.0:
              margin_cos = margin_cos - config.loss_m3
            margin_fc7 = margin_cos
            margin_fc7_onehot = margin_fc7 * gt_one_hot
            diff = margin_fc7_onehot - fc7_onehot
            fc7 = fc7+diff
        fc7 = fc7*config.loss_s

      out_list = []
      out_list.append(fc7)
      if config.loss_name=='softmax': #softmax
        out_list.append(gt_label)
      out = mx.symbol.Group(out_list)
      return out

    def train_net(args):
      esym = get_symbol_embedding()
      asym = get_symbol_arcface

      opt = optimizer.SGD(learning_rate=base_lr, momentum=base_mom, wd=base_wd, rescale_grad=_rescale)
    ```
***

# Highest Accuracy
## Deepinsight Accuracy

  | Method        | LFW(%) | CFP-FP(%) | AgeDB-30(%) | MegaFace(%)          |
  | ------------- | ------ | --------- | ----------- | -------------------- |
  | LResNet100E   | 99.77  | 98.27     | 98.28       | 98.47                |
  | LResNet50E    | 99.80  | 92.74     | 97.76       | 97.64                |
  | LResNet34E    | 99.65  | 92.12     | 97.70       | 96.70                |
  | MobileFaceNet | 99.50  | 88.94     | 95.91       | -----                |
  | VarGfaceNet   | 99.783 | 98.400    | 98.067      | 88.334 **DeepGlint** |
## Mobilenet 256 Embeddings

  | Step      | fc7_acc  | lfw     | cfp_fp  | agedb_30 | SUM     |
  | --------- | -------- | ------- | ------- | -------- | ------- |
  | **Loss**  | arcface  | **DS**  | glint   |          |         |
  | 1660K     | 0.25     | 0.99567 | 0.89529 | 0.96683  | 2.85779 |
  | **Loss**  | triplet  | **DS**  | emore   |          |         |
  | 840       | 0.062472 | 0.99633 | 0.93429 | 0.97083  | 2.90145 |
  | **Loss**  | triplet  | **DS**  | glint   |          |         |
  | 960[40]   | 0.064614 | 0.99617 | 0.93686 | 0.97017  | 2.90319 |
  | 2575[107] | 0.064761 | 0.99667 | 0.93829 | 0.96950  | 2.90445 |
## Mobilenet 512 embeddings
  | Step      | fc7_acc  | lfw     | cfp_fp  | agedb_30 | SUM     |
  | --------- | -------- | ------- | ------- | -------- | ------- |
  | **Loss**  | arcface  | **DS**  | emore   |          |         |
  | 1204K     | 0.015625 | 0.99533 | 0.93671 | 0.96367  | 2.89571 |
  | **Loss**  | triplet  | **DS**  | glint   |          |         |
  | 25[1]     | 0.146767 | 0.99567 | 0.93971 | 0.96500  | 2.90038 |
  | 532[20]   | 0.149680 | 0.99650 | 0.94614 | 0.96600  | 2.90864 |
  | 613[23]   | 0.146067 | 0.99683 | 0.94957 | 0.96300  | 2.90940 |
  | 668[25]   | 0.147614 | 0.99633 | 0.94757 | 0.96617  | 2.91007 |
  | 914[34]   | 0.148697 | 0.99650 | 0.94886 | 0.96517  | 2.91052 |
  | 996[37]   | 0.138909 | 0.99667 | 0.95014 | 0.96467  | 2.91148 |
  | 2809[102] | 0.146283 | 0.99600 | 0.95071 | 0.96783  | 2.91455 |
  | **Loss**  | triplet  | **DS**  | emore   |          |         |
  | 1697[65]  | 0.155924 | 0.99667 | 0.95129 | 0.96817  | 2.91612 |
## Vargface

  | Step     | fc7_acc    | lfw     | cfp_fp  | agedb_30 | SUM     |
  | -------- | ---------- | ------- | ------- | -------- | ------- |
  | **Loss** | softmax    | **DS**  | emore   |          |         |
  | 9.68K    | 0.265625   | 0.98383 | 0.82914 | 0.85117  |         |
  | 25.66K   | 0.28645834 | 0.98333 | 0.83729 | 0.85717  |         |
  | 62K      | 0.25520834 | 0.98067 | 0.83429 | 0.86517  |         |
  | 72K      | 0.3125     | 0.97683 | 0.81329 | 0.87217  |         |
  | 270K     | 0.7395833  | 0.99517 | 0.95086 | 0.93267  |         |
  | 332K     | 0.703125   | 0.99583 | 0.94857 | 0.93350  |         |
  | **Loss** | triplet    | **DS**  | glint   |          |         |
  | 175[10]  | 0.070560   | 0.99567 | 0.94314 | 0.95033  | 2.88914 |
  | 361[20]  | 0.056305   | 0.99683 | 0.94414 | 0.94867  | 2.88964 |
  | 648[35]  | 0.064737   | 0.99567 | 0.94700 | 0.95250  | 2.89517 |
  | **Loss** | triplet    | **DS**  | emore   |          |         |
***

# MTCNN
## Testing function
  ```py
  import skimage
  import cv2
  import matplotlib.pyplot as plt

  def test_mtcnn_multi_face(img_name, detector, image_type="RGB"):
      fig = plt.figure()
      ax = fig.add_subplot()

      if image_type == "RGB":
          imgm = skimage.io.imread(img_name)
          ax.imshow(imgm)
      else:
          imgm = cv2.imread(img_name)
          ax.imshow(imgm[:, :, ::-1])

      bb, pp = detector(imgm)
      print(bb.shape, pp.shape)

      for cc in bb:
          rr = plt.Rectangle(cc[:2], cc[2] - cc[0], cc[3] - cc[1], fill=False, color='r')
          ax.add_patch(rr)
      return bb, pp
  ```
  ![](images/mtcnn_test_image.jpg)
## facenet mtcnn
  - Q: ValueError: Object arrays cannot be loaded when allow_pickle=False
    ```py
    # vi /home/leondgarse/workspace/face_recognition_collection/facenet/src/align/detect_face.py +85
    data_dict = np.load(data_path, allow_pickle=True, encoding='latin1').item() #pylint: disable=no-member
    ```
  ```py
  # cd ~/workspace/face_recognition_collection/facenet/src

  import tensorflow as tf
  import align.detect_face
  from skimage.io import imread

  with tf.Graph().as_default():
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth = True)
      config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
      sess = tf.Session(config=config)
      with sess.as_default():
          pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
  # For test
  minsize = 40  # minimum size of face
  threshold = [0.9, 0.6, 0.7]  # three steps's threshold
  factor = 0.709  # scale factor

  def face_detection_align(img):
      return align.detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

  img = imread('../../test_img/Anthony_Hopkins_0002.jpg')
  print(face_detection_align(img))
  # array([[ 72.06663263,  62.10347486, 170.06547058, 188.18554652, 0.99998772]]),
  # array([[102.54911], [148.13242], [124.94654], [105.55612], [145.82028], [113.74067],
  #      [113.77531], [137.39977], [159.59608], [159.15378]], dtype=float32))
  %timeit face_detection_align(img)
  # 13.5 ms ± 660 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  bb, pp = test_mtcnn_multi_face('../../test_img/Fotos_anuales_del_deporte_de_2012.jpg', face_detection_align, image_type="RGB")
  # (12, 5) (10, 12)
  # 75.9 ms ± 483 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
  ```
  ![](images/facenet_mtcnn_multi.jpg)
## insightface mtcnn
  ```py
  # cd ~/workspace/face_recognition_collection/insightface/deploy

  from mtcnn_detector import MtcnnDetector
  import cv2

  det_threshold = [0.6,0.7,0.8]
  mtcnn_path = './mtcnn-model'
  detector = MtcnnDetector(model_folder=mtcnn_path, num_worker=2, accurate_landmark = False, threshold=det_threshold, minsize=40)

  img = cv2.imread('../../test_img/Anthony_Hopkins_0002.jpg')
  print(detector.detect_face(img, det_type=0))
  # array([[ 71.97946675,  64.52986962, 170.51717885, 187.63137624, 0.99999261]]),
  # array([[102.174866, 147.42386 , 124.979   , 104.82917 , 145.53633 , 113.806526,
  #     113.922585, 137.24968 , 160.5097  , 160.15164 ]], dtype=float32))
  %timeit detector.detect_face(img, det_type=0)
  # 23.5 ms ± 691 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

  bb, pp = test_mtcnn_multi_face('../../test_img/Fotos_anuales_del_deporte_de_2012.jpg', detector.detect_face, image_type="BGR")
  # (12, 5) (12, 10)
  # 122 ms ± 5.73 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
  ```
  ![](images/insightface_mtcnn_multi.jpg)
## MTCNN-Tensorflow
  - [AITTSMD/MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)
  ```py
  # cd ~/workspace/face_recognition_collection/MTCNN-Tensorflow/

  from Detection.MtcnnDetector import MtcnnDetector
  from Detection.detector import Detector
  from Detection.fcn_detector import FcnDetector
  from train_models.mtcnn_model import P_Net, R_Net, O_Net
  import cv2

  # thresh = [0.9, 0.6, 0.7]
  thresh = [0.6, 0.7, 0.8]
  # min_face_size = 24
  min_face_size = 40
  stride = 2
  slide_window = False
  shuffle = False

  #vis = True
  detectors = [None, None, None]
  prefix = ['data/MTCNN_model/PNet_landmark/PNet', 'data/MTCNN_model/RNet_landmark/RNet', 'data/MTCNN_model/ONet_landmark/ONet']
  epoch = [18, 14, 16]

  model_path = ['%s-%s' % (x, y) for x, y in zip(prefix, epoch)]
  PNet = FcnDetector(P_Net, model_path[0])
  detectors[0] = PNet
  RNet = Detector(R_Net, 24, 1, model_path[1])
  detectors[1] = RNet
  ONet = Detector(O_Net, 48, 1, model_path[2])
  detectors[2] = ONet
  mtcnn_detector = MtcnnDetector(detectors=detectors, min_face_size=min_face_size,
                                 stride=stride, threshold=thresh, slide_window=slide_window)

  img = cv2.imread('../test_img/Anthony_Hopkins_0002.jpg')
  mtcnn_detector.detect(img)
  %timeit mtcnn_detector.detect(img)
  # 37.5 ms ± 901 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

  bb, pp = test_mtcnn_multi_face('../test_img/Fotos_anuales_del_deporte_de_2012.jpg', mtcnn_detector.detect, image_type="BGR")
  # (10, 5) (10, 10)
  # 190 ms ± 4.31 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
  ```
  ![](images/MTCNN-Tensorflow_mtcnn_multi.jpg)
## mtcnn.MTCNN
  - [ipazc/mtcnn](https://github.com/ipazc/mtcnn)
  ```py
  # cd ~/workspace/face_recognition_collection
  from mtcnn.mtcnn import MTCNN
  from skimage.io import imread

  detector = MTCNN(steps_threshold=[0.6, 0.7, 0.7], min_face_size=40)
  img = imread('test_img/Anthony_Hopkins_0002.jpg')
  print(detector.detect_faces(img))
  # [{'box': [71, 60, 100, 127], 'confidence': 0.9999961853027344,
  #  'keypoints': {'left_eye': (102, 114), 'right_eye': (148, 114), 'nose': (125, 137),
  #                'mouth_left': (105, 160), 'mouth_right': (146, 160)}}]
  %timeit detector.detect_faces(img)
  # 14.5 ms ± 421 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  imgm = imread("test_img/Fotos_anuales_del_deporte_de_2012.jpg")
  print(len(detector.detect_faces(imgm)))
  # 12

  aa = detector.detect_faces(imgm)
  bb = [ii['box']for ii in aa]
  fig = plt.figure()
  ax = fig.add_subplot()
  ax.imshow(imgm)
  for cc in bb:
      rr = plt.Rectangle(cc[:2], cc[2], cc[3], fill=False, color='r')
      ax.add_patch(rr)

  %timeit detector.detect_faces(imgm)
  # 77.5 ms ± 164 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
  ```
  ```py
  for i in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']:
      points.append(lms[i][0])
      points.append(lms[i][1])
  ```
  ![](images/MTCNN_package_mtcnn_multi.jpg)
## caffe MTCNN
  ```py
  # cd ~/workspace/face_recognition_collection/MTCNN/python-caffe
  import skimage, cv2
  from MtcnnDetector import FaceDetector
  img = imread('../../test_img/Fotos_anuales_del_deporte_de_2012.jpg')
  detector = FaceDetector(minsize=40, fastresize=False, model_dir='../../insightface/deploy/mtcnn-model/')
  detector.detectface(img)
  def test_mtcnn_multi_face(img_name, detector, image_type="RGB"):
      fig = plt.figure()
      ax = fig.add_subplot()

      if image_type == "RGB":
          imgm = skimage.io.imread(img_name)
          ax.imshow(imgm)
      else:
          imgm = cv2.imread(img_name)
          ax.imshow(imgm[:, :, ::-1])

      bb, pp, _ = detector(imgm)
      print(bb.shape, pp.shape)

      for cc in bb:
          rr = plt.Rectangle(cc[:2], cc[2] - cc[0], cc[3] - cc[1], fill=False, color='r')
          ax.add_patch(rr)
      return bb, pp
  test_mtcnn_multi_face(img_name, detector.detectface)
  ```
## 人脸对齐
  ```py
  ''' 截取人脸位置图片 '''
  from mtcnn.mtcnn import MTCNN
  from skimage.io import imread
  from skimage.transform import resize
  detector = MTCNN(steps_threshold=[0.6, 0.7, 0.7], min_face_size=40)

  img = imread('/home/leondgarse/Atom_notebook/public/2019/images/mtcnn_test_image.jpg')
  ret = detector.detect_faces(img)
  bbox = [[ii["box"][0], ii["box"][1], ii["box"][0] + ii["box"][2], ii["box"][1] + ii["box"][3]] for ii in ret]
  nimgs = [resize(img[bb[1]: bb[3], bb[0]: bb[2]], (112, 112)) for bb in bbox]
  fig = plt.figure(figsize=(8, 1))
  plt.imshow(np.hstack(nimgs))
  plt.axis('off')
  plt.tight_layout()
  ```
  ![](images/tf_pb_model_faces.jpg)
  ```py
  from skimage.transform import SimilarityTransform
  import cv2

  def face_align_landmarks(img, landmarks, image_size=(112, 112)):
      ret = []
      for landmark in landmarks:
          src = np.array(
              [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]],
              dtype=np.float32,
          )

          dst = landmark.astype(np.float32)
          tform = SimilarityTransform()
          tform.estimate(dst, src)
          M = tform.params[0:2, :]
          ret.append(cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0))

      return np.array(ret)

  points = np.array([list(ii["keypoints"].values()) for ii in ret])
  nimgs = face_align_landmarks(img, points)
  fig = plt.figure(figsize=(8, 1))
  plt.imshow(np.hstack(nimgs))
  plt.axis('off')
  plt.tight_layout()
  ```
  ![](images/tf_pb_align_faces.jpg)
***

# Docker 封装
  ```sh
  sudo apt-get install -y nvidia-docker2
  docker run --runtime=nvidia -v /home/tdtest/workspace/:/home/tdtest/workspace -it tensorflow/tensorflow:latest-gpu-py3 bash

  pip install --upgrade pip
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple sklearn scikit-image waitress python-crontab opencv-python mtcnn requests

  apt update && apt install python3-opencv

  nohup ./server_flask.py -l 0 > app.log 2>&1 &

  docker ps -a | grep tensorflow | cut -d ' ' -f 1
  docker exec  -p 8082:9082 -it `docker ps -a | grep tensorflow | cut -d ' ' -f 1` bash

  docker commit `docker ps -a | grep tensorflow | cut -d ' ' -f 1` insightface
  docker run -e CUDA_VISIBLE_DEVICES='1' -v /home/tdtest/workspace/:/workspace -it -p 9082:8082 -w /workspace/insightface-master insightface:latest ./server_flask.py
  ```
***

# 视频中识别人脸保存
  ```py
  "rtsp://admin:admin111@192.168.0.65:554/cam/realmonitor?channel=1&subtype=1"
  import cv2
  import sys

  sys.path.append("../")
  from face_model.face_model import FaceModel
  fm = FaceModel(None)

  def video_capture_local_mtcnn(fm, dest_path, dest_name_pre, video_src=0, skip_frame=5):
      cap = cv2.VideoCapture(video_src)
      pic_id = 0
      cur_frame = 0
      while(cap.isOpened()):
          ret, frame = cap.read()
          if ret == True:
              frame = cv2.flip(frame, 0)

              if cur_frame == 0:
                  frame_rgb = frame[:, :, ::-1]
                  bbs, ccs, pps = fm.get_face_location(frame_rgb)
                  if len(bbs) != 0:
                      nns = fm.face_align_landmarks(frame_rgb, pps)
                      for nn in nns:
                          fname = os.path.join(dest_path, dest_name_pre + "_{}.png".format(pic_id))
                          print("fname = %s" % fname)
                          plt.imsave(fname, nn)
                          pic_id += 1

              cv2.imshow('frame', frame)
              cur_frame = (cur_frame + 1) % skip_frame
              key = cv2.waitKey(1) & 0xFF
              if key == ord('q'):
                  break
          else:
              break

      cap.release()
      cv2.destroyAllWindows()
  ```
***
