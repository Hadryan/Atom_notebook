# ___2019 - 07 - 19 Insightface___
***

# Related projects
  - [deepinsight/insightface](https://github.com/deepinsight/insightface)
  - [AITTSMD/MTCNN-Tensorflow](https://github.com/AITTSMD/MTCNN-Tensorflow)
  - [ipazc/mtcnn](https://github.com/ipazc/mtcnn)
  - [erikbern/ann-benchmarks](https://github.com/erikbern/ann-benchmarks)
***

# Insightface MXNet 模型使用
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
## TensorFlow mtcnn pb
  ```py
  # cd ~/workspace/face_recognition_collection/MTCNN/tensorflow
  import cv2
  from mtcnn import MTCNN

  det = MTCNN('./mtcnn.pb')
  img = cv2.imread('../../test_img/Anthony_Hopkins_0002.jpg')

  det.detect(img)
  # (array([[ 65.71266,  74.45414, 187.65063, 172.71921]], dtype=float32),
  #  array([0.99999845], dtype=float32),
  #  array([[113.4738  , 113.50406 , 138.02603 , 159.49994 , 158.71802 ,
  #          102.397964, 147.4054  , 125.014786, 105.924614, 145.5773  ]],
  #        dtype=float32))
  ```
  ```py
  bb, cc, pp = det.detect(img)
  bb = np.array([[ii[1], ii[0], ii[3], ii[2]] for ii in bb])
  # Out[21]: array([[ 74.45414,  65.71266, 172.71921, 187.65063]], dtype=float32)
  pp = np.array([ii.reshape(2, 5)[::-1].T for ii in pp])
  # array([[[102.397964, 113.4738  ],
  #         [147.4054  , 113.50406 ],
  #         [125.014786, 138.02603 ],
  #         [105.924614, 159.49994 ],
  #         [145.5773  , 158.71802 ]]], dtype=float32)
  ```
***

# Insightface caffe MTCNN model to TensorFlow
## caffe
  - [BVLC/caffe/docker](https://github.com/BVLC/caffe/tree/master/docker)
  ```sh
  docker run --rm -u $(id -u):$(id -g) -v $(pwd):$(pwd) -w $(pwd) bvlc/caffe:cpu caffe train --solver=example_solver.prototxt
  ```
## MMDNN 转化
  - [microsoft/MMdnn](https://github.com/microsoft/MMdnn)
  ```sh
  pip install mmdnn
  python -m mmdnn.conversion._script.convertToIR -f mxnet -n det1-symbol.json -w det1-0001.params -d det1 --inputShape 3,112,112
  mmconvert -sf mxnet -iw det1-0001.params -in det1-symbol.json -df tensorflow -om det1 --inputShape 3,224,224
  ```
  ```sh
  cd ~/workspace/face_recognition_collection/facenet/src
  cp align align_bak -r

  cd ~/workspace/face_recognition_collection/insightface/deploy/mtcnn-model
  mmtoir -f caffe -n det1.prototxt -w det1.caffemodel -o det1
  mmtoir -f caffe -n det2.prototxt -w det2.caffemodel -o det2
  mmtoir -f caffe -n det3.prototxt -w det3.caffemodel -o det3

  mmtocode -f tensorflow --IRModelPath det1.pb --IRWeightPath det1.npy --dstModelPath det1.py
  mmtocode -f tensorflow --IRModelPath det2.pb --IRWeightPath det2.npy --dstModelPath det2.py
  mmtocode -f tensorflow --IRModelPath det3.pb --IRWeightPath det3.npy --dstModelPath det3.py

  cp *.npy ~/workspace/face_recognition_collection/facenet/src/align/
  ```
## 参数调整
  ```py
  aa = np.load('align/det1.npy', allow_pickle=True).item()
  bb = np.load('align_bak/det1.npy', allow_pickle=True, encoding='latin1').item()

  for kk, vv in aa.items():
      print(kk, vv.keys())
  print()

  for kk, vv in bb.items():
      print(kk, vv.keys())
  print()

  for ii in ['conv1', 'conv2', 'conv3', 'conv4-1', 'conv4-2']:
      aa[ii]['biases'] = aa[ii]['bias']
      aa[ii].pop('bias')

  for ii in ['PReLU1', 'PReLU2', 'PReLU3']:
      aa[ii]['alpha'] = aa[ii]['gamma']
      aa[ii].pop('gamma')

  for ii in ['conv1', 'conv2', 'conv3', 'conv4-1', 'conv4-2']:
      aa[ii]['biases'] = np.squeeze(aa[ii]['biases'])

  np.save('align_2/det1', aa)

  aa = np.load('align_2/det1.npy', allow_pickle=True).item()
  for kk, vv in aa.items():
      print(kk, vv.keys())
  ```
  ```py
  aa = np.load('align/det2.npy', allow_pickle=True).item()
  bb = np.load('align_bak/det2.npy', allow_pickle=True, encoding='latin1').item()

  for kk, vv in aa.items():
      print(kk, vv.keys())
  print()

  for kk, vv in bb.items():
      print(kk, vv.keys())
  print()

  for ii in ['conv1', 'conv2', 'conv3', 'conv4_1', 'conv5-1_1', 'conv5-2_1']:
      aa[ii]['biases'] = aa[ii]['bias']
      aa[ii].pop('bias')

  for ii in ['prelu1', 'prelu2', 'prelu3', 'prelu4']:
      aa[ii]['alpha'] = aa[ii]['gamma']
      aa[ii].pop('gamma')

  aa['conv4'] = aa['conv4_1']
  aa.pop('conv4_1')
  aa['conv5-1'] = aa['conv5-1_1']
  aa.pop('conv5-1_1')
  aa['conv5-2'] = aa['conv5-2_1']
  aa.pop('conv5-2_1')

  for ii in ['conv1', 'conv2', 'conv3']:
      aa[ii]['biases'] = np.squeeze(aa[ii]['biases'])

  np.save('align_2/det2', aa)

  aa = np.load('align_2/det2.npy', allow_pickle=True).item()
  for kk, vv in aa.items():
      print(kk, vv.keys())
  ```
  ```py
  aa = np.load('align/det3.npy', allow_pickle=True).item()
  bb = np.load('align_bak/det3.npy', allow_pickle=True, encoding='latin1').item()

  for kk, vv in aa.items():
      print(kk, vv.keys())
  print()

  for kk, vv in bb.items():
      print(kk, vv.keys())
  print()

  for ii in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5_1', 'conv6-1_1', 'conv6-2_1', 'conv6-3_1']:
      aa[ii]['biases'] = aa[ii]['bias']
      aa[ii].pop('bias')

  for ii in ['prelu1', 'prelu2', 'prelu3', 'prelu4', 'prelu5']:
      aa[ii]['alpha'] = aa[ii]['gamma']
      aa[ii].pop('gamma')

  aa['conv5'] = aa['conv5_1']
  aa.pop('conv5_1')
  aa['conv6-1'] = aa['conv6-1_1']
  aa.pop('conv6-1_1')
  aa['conv6-2'] = aa['conv6-2_1']
  aa.pop('conv6-2_1')
  aa['conv6-3'] = aa['conv6-3_1']
  aa.pop('conv6-3_1')

  for ii in ['conv1', 'conv2', 'conv3', 'conv4']:
      aa[ii]['biases'] = np.squeeze(aa[ii]['biases'])

  np.save('align_2/det3', aa)

  aa = np.load('align_2/det3.npy', allow_pickle=True).item()
  for kk, vv in aa.items():
      print(kk, vv.keys())
  ```
## TensorFlow MTCNN model to MTCNN package model
  ```py
  aa = np.load('/opt/anaconda3/lib/python3.7/site-packages/mtcnn/data/mtcnn_weights.npy', allow_pickle=True).item()
  bb_1 = np.load('det1.npy', allow_pickle=True, encoding='latin1').item()

  for kk, vv in aa['PNet'].items():
      print(kk, vv.keys())
  print()

  for kk, vv in bb_1.items():
      print(kk, vv.keys())
  print()

  for ss, dd in zip(['PReLU1', 'PReLU2', 'PReLU3'], ['prelu1', 'prelu2', 'prelu3']):
      bb_1[dd] = bb_1[ss]
      bb_1.pop(ss)

  bb_2 = np.load('det2.npy', allow_pickle=True, encoding='latin1').item()

  for kk, vv in aa['ONet'].items():
      print(kk, vv.keys())
  print()

  for kk, vv in bb_2.items():
      print(kk, vv.keys())
  print()
  np.save('align_2/det1', aa)

  aa = np.load('align_2/det1.npy', allow_pickle=True).item()
  for kk, vv in aa.items():
      print(kk, vv.keys())
  ```
  ```py
  bb = np.load('det2.npy', allow_pickle=True, encoding='latin1').item()

  for kk, vv in aa['ONet'].items():
      print(kk, vv.keys())
  print()

  for kk, vv in bb.items():
      print(kk, vv.keys())
  print()

  for ii in ['conv1', 'conv2', 'conv3', 'conv4_1', 'conv5-1_1', 'conv5-2_1']:
      aa[ii]['biases'] = aa[ii]['bias']
      aa[ii].pop('bias')

  for ii in ['prelu1', 'prelu2', 'prelu3', 'prelu4']:
      aa[ii]['alpha'] = aa[ii]['gamma']
      aa[ii].pop('gamma')

  aa['conv4'] = aa['conv4_1']
  aa.pop('conv4_1')
  aa['conv5-1'] = aa['conv5-1_1']
  aa.pop('conv5-1_1')
  aa['conv5-2'] = aa['conv5-2_1']
  aa.pop('conv5-2_1')

  for ii in ['conv1', 'conv2', 'conv3']:
      aa[ii]['biases'] = np.squeeze(aa[ii]['biases'])

  np.save('align_2/det2', aa)

  aa = np.load('align_2/det2.npy', allow_pickle=True).item()
  for kk, vv in aa.items():
      print(kk, vv.keys())
  ```
  ```py
  bb_3 = np.load('det3.npy', allow_pickle=True, encoding='latin1').item()

  for kk, vv in aa['RNet'].items():
      print(kk, vv.keys())
  print()

  for kk, vv in bb_3.items():
      print(kk, vv.keys())
  print()

  for ii in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5_1', 'conv6-1_1', 'conv6-2_1', 'conv6-3_1']:
      aa[ii]['biases'] = aa[ii]['bias']
      aa[ii].pop('bias')

  for ii in ['prelu1', 'prelu2', 'prelu3', 'prelu4', 'prelu5']:
      aa[ii]['alpha'] = aa[ii]['gamma']
      aa[ii].pop('gamma')

  aa['conv5'] = aa['conv5_1']
  aa.pop('conv5_1')
  aa['conv6-1'] = aa['conv6-1_1']
  aa.pop('conv6-1_1')
  aa['conv6-2'] = aa['conv6-2_1']
  aa.pop('conv6-2_1')
  aa['conv6-3'] = aa['conv6-3_1']
  aa.pop('conv6-3_1')

  for ii in ['conv1', 'conv2', 'conv3', 'conv4']:
      aa[ii]['biases'] = np.squeeze(aa[ii]['biases'])

  np.save('align_2/det3', aa)

  aa = np.load('align_2/det3.npy', allow_pickle=True).item()
  for kk, vv in aa.items():
      print(kk, vv.keys())
  ```
***

# Insightface MXNET model to TensorFlow pb model
## MMDNN 转化模型
  ```sh
  cd model-r100-ii/

  # 一次转化
  mmconvert -sf mxnet -in model-symbol.json -iw model-0000.params -df tensorflow -om resnet100 --dump_tag SERVING --inputShape 3,112,112

  # 分步执行
  mmtoir -f mxnet -n model-symbol.json -w model-0000.params -d resnet100 --inputShape 3,112,112
  mmtocode -f tensorflow --IRModelPath resnet100.pb --IRWeightPath resnet100.npy --dstModelPath tf_resnet100.py
  mmtomodel -f tensorflow -in tf_resnet100.py -iw resnet100.npy -o tf_resnet100 --dump_tag SERVING
  ```
## TensorFlow 加载 PB 模型
  ```py
  ''' 截取人脸位置图片 '''
  from mtcnn.mtcnn import MTCNN
  from skimage.io import imread
  from skimage.transform import resize
  detector = MTCNN(steps_threshold=[0.6, 0.7, 0.7], min_face_size=40)

  img = imread('/home/leondgarse/workspace/face_recognition_collection/test_img/Fotos_anuales_del_deporte_de_2012.jpg')
  ret = detector.detect_faces(img)
  bbox = [[ii["box"][0], ii["box"][1], ii["box"][2] + ii["box"][0], ii["box"][3] + ii["box"][1]] for ii in ret]
  nimgs = [resize(img[bb[1]: bb[3], bb[0]: bb[2]], (112, 112)) for bb in bbox]
  fig = plt.figure(figsize=(8, 1))
  plt.imshow(np.hstack(nimgs))
  plt.axis('off')
  plt.tight_layout()

  ''' 提取特征值 '''
  sess = tf.InteractiveSession()
  meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], "./resnet100")
  x = sess.graph.get_tensor_by_name("data:0")
  y = sess.graph.get_tensor_by_name("fc1/add_1:0")

  emb = sess.run(y, feed_dict={x: nimgs})
  print(emb.shape)
  # (11, 512)
  ```
  ![](images/tf_pb_model_faces.jpg)
## 人脸对齐
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
## Tensorflow Serving server
  - `saved_model_cli` 显示模型 signature_def 信息
    ```sh
    cd /home/leondgarse/workspace/models/insightface_mxnet_model/model-r100-ii/tf_resnet100
    tree
    # .
    # ├── 1
    # │   ├── saved_model.pb
    # │   └── variables
    # │       ├── variables.data-00000-of-00001
    # │       └── variables.index

    saved_model_cli show --dir ./1
    # The given SavedModel contains the following tag-sets:
    # serve

    saved_model_cli show --dir ./1 --tag_set serve
    # The given SavedModel MetaGraphDef contains SignatureDefs with the following keys:
    # SignatureDef key: "serving_default"

    saved_model_cli show --dir ./1 --tag_set serve --signature_def serving_default
    # The given SavedModel SignatureDef contains the following input(s):
    #   inputs['input'] tensor_info:
    #       dtype: DT_FLOAT
    #       shape: (-1, 112, 112, 3)
    #       name: data:0
    # The given SavedModel SignatureDef contains the following output(s):
    #   outputs['output'] tensor_info:
    #       dtype: DT_FLOAT
    #       shape: (-1, 512)
    #       name: fc1/add_1:0
    # Method name is: tensorflow/serving/predict
    ```
  - `tensorflow_model_server` 启动服务
    ```sh
    # model_base_path 需要绝对路径
    tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=arcface --model_base_path=/home/leondgarse/workspace/models/insightface_mxnet_model/model-r100-ii/tf_resnet100
    ```
  - `requests` 请求返回特征值结果
    ```py
    import json
    import requests
    from skimage.transform import resize

    rr = requests.get("http://localhost:8501/v1/models/arcface")
    print(rr.json())
    # {'model_version_status': [{'version': '1', 'state': 'AVAILABLE', 'status': {'error_code': 'OK', 'error_message': ''}}]}

    x = plt.imread('grace_hopper.jpg')
    print(x.shape)
    # (600, 512, 3)

    xx = resize(x, [112, 112])
    data = json.dumps({"signature_name": "serving_default", "instances": [xx.tolist()]})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/arcface:predict',data=data, headers=headers)
    rr = json_response.json()
    print(rr.keys(), np.shape(rr['predictions']))
    # dict_keys(['predictions']) (1, 512)
    ```
  - `MTCNN` 提取人脸位置后请求结果
    ```py
    from mtcnn.mtcnn import MTCNN

    img = plt.imread('grace_hopper.jpg')
    detector = MTCNN(steps_threshold=[0.6, 0.7, 0.7])
    aa = detector.detect_faces(img)
    bb = aa[0]['box']
    cc = img[bb[1]: bb[1] + bb[3], bb[0]: bb[0] + bb[2]]
    dd = resize(cc, [112, 112])

    data = json.dumps({"signature_name": "serving_default", "instances": [dd.tolist()]})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/arcface:predict',data=data, headers=headers)
    rr = json_response.json()
    print(rr.keys(), np.shape(rr['predictions']))
    # dict_keys(['predictions']) (1, 512)
    ```
***

# Test commands
## timeit test
  - mxnet, threads = 1, 1:18.42
  - mxnet, threads = 20, 0:56.99
  - TensorFlow, threads = 1, 1:27.22
  - TensorFlow, threads = 20, 0:45.21
  ```py
  In [13]: %timeit np.savez('1', feature=bb['feature'], lab=bb['lab'])
  1 loop, best of 5: 11.1 s per loop

  In [14]: %timeit with open('1.pkl', 'wb') as ff: pickle.dump(bb, ff)
  1 loop, best of 5: 9.77 s per loop

  In [15]: %timeit np.save('1', bb)
  1 loop, best of 5: 37.6 s per loop

  In [25]: %timeit with open('1.pkl', 'rb') as ff: dd = pickle.load(ff)
  1 loop, best of 5: 3.7 s per loop

  %timeit bb = np.load('1.npz')['feature']
  1 loop, best of 5: 4.75 s per loop
  ```
  ```py
  In [23]: %timeit mtcnn.detect(img)
  16.7 ms ± 383 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  In [24]: %timeit mtcnn.detect(img)
  16.9 ms ± 369 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  In [25]: %timeit detector.detectface(img)
  30.5 ms ± 107 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

  In [26]: %timeit detector.detectface(img)
  30.3 ms ± 114 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

  In [27]: %timeit detector.detectface(img)
  29.7 ms ± 385 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
  ```
  ```py
  In [22]: %timeit -n 100 det_tfo.detect_faces(imr)
  73.5 ms ± 607 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  In [23]: %timeit -n 100 det_caffe.detectface(imb)
  30.9 ms ± 650 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  In [24]: %timeit -n 100 det_tfp.detect(imb)
  17.1 ms ± 488 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  In [9]: %timeit -n 100 det_tfp.detect(imb)
  35.6 ms ± 316 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
  ```
***

# Docker
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
