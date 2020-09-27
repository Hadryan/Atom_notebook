# ___2020 - 08 - 21 Face Detect___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2020 - 08 - 21 Face Detect___](#2020-08-21-face-detect)
  - [目录](#目录)
  - [MTCNN](#mtcnn)
  - [Yoloface](#yoloface)
  	- [Pytorch Inference](#pytorch-inference)
  	- [Convert ONNX](#convert-onnx)
  	- [Yolo decode](#yolo-decode)
  	- [CPP yolo decode](#cpp-yolo-decode)
  	- [Convert Tensorflow](#convert-tensorflow)
  	- [TFLite inference](#tflite-inference)
  - [Centerface](#centerface)
  	- [keras](#keras)
  	- [CPP](#cpp)
  - [Face detection test](#face-detection-test)
  - [Retinaface](#retinaface)
  	- [Pytorch](#pytorch)
  	- [Keras](#keras)
  - [Face recognition test](#face-recognition-test)

  <!-- /TOC -->
***

# Basic
## Image show function
  ```py
  def show_result(image, bbs, ccs=[], pps=[]):
      plt.figure()
      plt.imshow(image)
      for id, bb in enumerate(bbs):
          plt.plot([bb[0], bb[2], bb[2], bb[0], bb[0]], [bb[1], bb[1], bb[3], bb[3], bb[1]])
          if len(ccs) != 0:
              plt.text(bb[0], bb[1], '{:.4f}'.format(ccs[id]))
          if len(pps) != 0:
              pp = pps[id]
              if len(pp.shape) == 2:
                  plt.scatter(pp[:, 0], pp[:, 1], s=8)
              else:
                  plt.scatter(pp[::2], pp[1::2], s=8)
      plt.axis('off')
      plt.tight_layout()
  ```
## Face align landmarks
  ```py
  from skimage import transform
  def face_align_landmarks_sk(img, landmarks, image_size=(112, 112), method='similar'):
      tform = transform.AffineTransform() if method == 'affine' else transform.SimilarityTransform()
      src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
      ret = []
      for landmark in landmarks:
          # landmark = np.array(landmark).reshape(2, 5)[::-1].T
          tform.estimate(landmark, src)
          ret.append(transform.warp(img, tform.inverse, output_shape=image_size))
      return (np.array(ret) * 255).astype(np.uint8)
  ```
***

***
# MTCNN
## facenet mtcnn
  - Q: ValueError: Object arrays cannot be loaded when allow_pickle=False
    ```py
    # vi /home/leondgarse/workspace/face_recognition_collection/facenet/src/align/detect_face.py +85
    data_dict = np.load(data_path, allow_pickle=True, encoding='latin1').item() #pylint: disable=no-member
    ```
  ```py
  # cd ~/workspace/face_recognition_collection/facenet/src

  import tensorflow as tf
  tf.__version__
  # 1.15.0

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

  img = imread('/home/leondgarse/workspace/test_images/Anthony_Hopkins_0002.jpg')
  print(face_detection_align(img))
  # array([[ 72.88286112,  62.74520046, 173.87759207, 188.97606539, 0.99998605]])
  # array([[102.59091 ], [147.37492 ], [125.6481  ], [106.344215], [146.40628 ], [113.58914 ], [113.859146], [137.90286 ], [159.99037 ], [159.17723 ]])
  %timeit face_detection_align(img)
  # 13.5 ms ± 660 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  imm = imread('/home/leondgarse/workspace/test_images/Fotos_anuales_del_deporte_de_2012.jpg')
  bbs, pps = face_detection_align(imm)
  show_result(imm, bbs[:, :4], bbs[:, 4], pps=pps.T.reshape(-1, 2, 5).transpose(0, 2, 1))
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

  img = cv2.imread('/home/leondgarse/workspace/test_images/Anthony_Hopkins_0002.jpg')
  print(detector.detect_face(img, det_type=0))
  # array([[ 71.97946675,  64.52986962, 170.51717885, 187.63137624, 0.99999261]]),
  # array([[102.174866, 147.42386 , 124.979   , 104.82917 , 145.53633 , 113.806526,
  #     113.922585, 137.24968 , 160.5097  , 160.15164 ]], dtype=float32))
  %timeit detector.detect_face(img, det_type=0)
  # 23.5 ms ± 691 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

  imm = cv2.imread('/home/leondgarse/workspace/test_images/Fotos_anuales_del_deporte_de_2012.jpg')
  bbs, pps = detector.detect_face(imm)
  show_result(imm[:, :, ::-1], bbs[:, :4], bbs[:, 4], pps=pps.reshape(-1, 2, 5).transpose(0, 2, 1))
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

  img = cv2.imread('/home/leondgarse/workspace/test_images/Anthony_Hopkins_0002.jpg')
  mtcnn_detector.detect(img)
  %timeit mtcnn_detector.detect(img)
  # 37.5 ms ± 901 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

  imm = cv2.imread('/home/leondgarse/workspace/test_images/Fotos_anuales_del_deporte_de_2012.jpg')
  bbs, pps = mtcnn_detector.detect(imm)
  show_result(imm[:, :, ::-1], bbs[:, :4], bbs[:, 4], pps=pps.reshape(-1, 5, 2))
  ```
  ![](images/MTCNN-Tensorflow_mtcnn_multi.jpg)
## mtcnn.MTCNN
  - [ipazc/mtcnn](https://github.com/ipazc/mtcnn)
  ```py
  !pip install mtcnn

  from mtcnn.mtcnn import MTCNN
  from skimage.io import imread

  detector = MTCNN(steps_threshold=[0.6, 0.7, 0.7], min_face_size=40)
  img = imread('/home/leondgarse/workspace/test_images/Anthony_Hopkins_0002.jpg')
  print(detector.detect_faces(img))
  # [{'box': [71, 60, 100, 127], 'confidence': 0.9999961853027344,
  #  'keypoints': {'left_eye': (102, 114), 'right_eye': (148, 114), 'nose': (125, 137),
  #                'mouth_left': (105, 160), 'mouth_right': (146, 160)}}]
  %timeit detector.detect_faces(img)
  # 14.5 ms ± 421 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

  imm = imread('/home/leondgarse/workspace/test_images/Fotos_anuales_del_deporte_de_2012.jpg')
  aa = detector.detect_faces(imm)
  bbs = [ii['box'] for ii in aa]
  bbs = np.array([[ii[0], ii[1], ii[0] + ii[2], ii[1] + ii[3]] for ii in bbs])
  ccs = np.array([ii['confidence'] for ii in aa])
  pps = np.array([[ii['keypoints'][kk] for kk in ['left_eye', 'right_eye', 'nose', 'mouth_left', 'mouth_right']] for ii in aa])
  show_result(imm, bbs, ccs, pps)
  ```
  ![](images/MTCNN_package_mtcnn_multi.jpg)
## MTCNN multiple versions
  - [Github imistyrain/MTCNN](https://github.com/imistyrain/MTCNN.git)
***

# Yoloface
## Pytorch Inference
  - [Github ouyanghuiyu/yolo-face-with-landmark](https://github.com/ouyanghuiyu/yolo-face-with-landmark)
  - **Torch model inference**
    ```py
    ! cd yolo-face-with-landmark

    import torch
    from nn.models import DarknetWithShh
    from nn.backbone.mobilenetv3 import mobilenetv3_small, mobilenetv3_large, mobilenetv3_large_full
    from hyp import hyp

    device = torch.device('cpu')
    backone = mobilenetv3_small()
    weights = "./weights/mbv3_small_1_final.pt"

    net = DarknetWithShh(backone, hyp, onnx_export=False).to(device)
    net.load_state_dict(torch.load(weights, map_location=device)['model'])
    net.eval()

    image = plt.imread('src/test.jpg')
    imm = np.expand_dims(np.transpose(image[:480, :, :] / 255, (2, 0, 1)), axis=0)
    tim = torch.from_numpy(imm).to(device).float()

    net_preds = net(tim)
    print(net_preds[0].shape, [ii.shape for ii in net_preds[1]])
    # torch.Size([1, 30240, 16]) [torch.Size([1, 3, 60, 128, 16]), torch.Size([1, 3, 30, 64, 16]), torch.Size([1, 3, 15, 32, 16])]

    pred = non_max_suppression(net_preds[0], 0.25, 0.35, multi_label=False, classes=0, agnostic=False,land=True ,point_num=5)
    det = pred[0].cpu().detach().numpy()
    print(det.shape)
    # (3, 16)
    ```
    **Single test function**
    ```py
    from utils.utils import non_max_suppression

    def det_test(net, image, device=torch.device('cpu')):
        imm = np.expand_dims(np.transpose(image / 255, (2, 0, 1)), axis=0)
        tim = torch.from_numpy(imm).to(device).float()
        net_preds = net(tim)
        pred = non_max_suppression(net_preds[0], 0.25, 0.35, multi_label=False, classes=0, agnostic=False,land=True ,point_num=5)
        ret = pred[0].cpu().detach().numpy()
        bbs, ccs, pps = ret[:, :4], ret[:, 4], ret[:, 5:-1]
        return bbs, ccs, pps

    bbs, ccs, pps = det_test(net, image[:480, :, :])
    show_result(image, bbs, ccs, pps)
    ```
    ![](images/yolo_pytorch_out.jpeg)
## Convert ONNX
  - **Convert torch model to ONNX**
    ```py
    ! pip install torchstat onnx-simplifier

    import torch
    from nn.models import DarknetWithShh
    from nn.backbone.mobilenetv3 import mobilenetv3_small, mobilenetv3_large, mobilenetv3_large_full
    from hyp import hyp

    device = torch.device('cpu')
    net = DarknetWithShh(mobilenetv3_small(), hyp, onnx_export=True).to(device)
    net.load_state_dict(torch.load("./weights/mbv3_small_1_final.pt", map_location=device)['model'])
    # net.eval()

    # image = torch.from_numpy(np.ones((1, 3, 224, 224))).to(device).float()
    # net_preds = net(tim)

    output_onnx = 'onnx/mbv3_small_1.onnx'
    input_names = ["input0"]
    output_names = ["s8", "s16" , "s32"]
    # inputs = torch.randn(1, 3, 256, 160).to(device) # (1280, 800) / 5
    inputs = torch.randn(1, 3, 480, 1024).to(device)
    # torch.onnx.export(net, inputs, output_onnx, verbose=False, input_names=input_names, output_names=output_names, keep_initializers_as_inputs=False)
    # torch.onnx.export(net, inputs, output_onnx, verbose=False, input_names=input_names, output_names=output_names, do_constant_folding=False)
    torch.onnx.export(net, inputs, output_onnx, verbose=False, input_names=input_names, output_names=output_names)
    # os.system("python -m onnxsim {} {} --input-shape 1,3,256,160".format(output_onnx,output_onnx))
    ```
    ```py
    def yolo_convert_to_onnx(weight, input_shape=(256, 160)):
        width_mult = 0.75 if '75' in weight else 1
        model = mobilenetv3_small if "small" in weight else mobilenetv3_large
        light_head = True if "light" in weight else False
        model_name = os.path.splitext(os.path.basename(weight))[0]

        device = torch.device('cpu')
        net = DarknetWithShh(model(width_mult=width_mult), hyp, light_head=light_head, onnx_export=True).to(device)
        net.load_state_dict(torch.load(weight, map_location=device)['model'])
        # net.eval()

        # image = torch.from_numpy(np.ones((1, 3, 224, 224))).to(device).float()
        # net_preds = net(tim)

        output_onnx = 'onnx/{}.onnx'.format(model_name)
        input_names = ["input0"]
        output_names = ["s8", "s16" , "s32"]
        # inputs = torch.randn(1, 3, 256, 160).to(device) # (1280, 800) / 5
        inputs = torch.randn(1, 3, *input_shape).to(device)
        # torch.onnx.export(net, inputs, output_onnx, verbose=False, input_names=input_names, output_names=output_names, keep_initializers_as_inputs=False)
        # torch.onnx.export(net, inputs, output_onnx, verbose=False, input_names=input_names, output_names=output_names, do_constant_folding=False)
        torch.onnx.export(net, inputs, output_onnx, verbose=False, input_names=input_names, output_names=output_names)
        return output_onnx

    import glob2
    [yolo_convert_to_onnx(ii) for ii in glob2.glob('./weights/*.pt')]
    ```
    **Export dynamic_axes**
    ```py
    # torch.onnx.export(net, inputs, output_onnx, verbose=False, input_names=input_names, output_names=output_names, dynamic_axes={input_names[0]: {2: "h", 3: "w"}})
    torch.onnx.export(net, inputs, 'mbv3_small_1_D.onnx', verbose=False, input_names=input_names, output_names=output_names, dynamic_axes={ii: {2: "h", 3: "w"} for ii in input_names + output_names})
    ```
  - **ONNX inference**
    ```py
    import onnxruntime
    sess = onnxruntime.InferenceSession('mbv3_small_1.onnx')
    print(sess.get_inputs()[0].shape, [(ii.name, ii.shape) for ii in sess.get_outputs()])
    # [1, 3, 480, 1024] [('s8', [1, 48, 60, 128]), ('s16', [1, 48, 30, 64]), ('s32', [1, 48, 15, 32])]
    onnx_preds = sess.run(None, {sess.get_inputs()[0].name: np.ones([1, 3, 256, 160]).astype('float32')})
    ```
    **Using dynamic_axes**
    ```py
    import onnxruntime
    sess = onnxruntime.InferenceSession('mbv3_small_1_D.onnx')
    print(sess.get_inputs()[0].shape, [(ii.name, ii.shape) for ii in sess.get_outputs()])
    # [1, 3, 'h', 'w'] [('s8', [1, 48, 'h', 'w']), ('s16', [1, 48, 'h', 'w']), ('s32', [1, 48, 'h', 'w'])]
    print([ii.shape for ii in sess.run(None, {sess.get_inputs()[0].name: np.ones([1, 3, 256, 160]).astype('float32')})])
    # [(1, 48, 32, 20), (1, 48, 16, 10), (1, 48, 8, 5)]
    print([ii.shape for ii in sess.run(None, {sess.get_inputs()[0].name: np.ones([1, 3, 512, 160]).astype('float32')})])
    # [(1, 48, 64, 20), (1, 48, 32, 10), (1, 48, 16, 5)]
    print([ii.shape for ii in sess.run(None, {sess.get_inputs()[0].name: np.ones([1, 3, 512, 320]).astype('float32')})])
    # [(1, 48, 64, 40), (1, 48, 32, 20), (1, 48, 16, 10)]
    ```
## Yolo decode
  - **Numpy decode ONNX output**
    ```py
    pred_anchors = [
        [[12, 12], [20, 20], [32, 32]],
        [[48, 48], [72, 72], [128, 128]],
        [[192, 192], [320, 320], [480, 480]],
    ]
    strides = [8, 16, 32]

    def sigmoid(xx):
        return 1 / (1 + np.exp(-1 * xx))

    def np_create_grids(ng=(13, 13)):
        nx, ny = ng  # x and y grid size

        # build xy offsets
        yv, xv = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        grid = np.stack((xv, yv), 2).reshape((1, 1, ny, nx, 2)).astype("float32")
        return grid

    def np_yolodecode(pred, pred_anchor, stride, pred_format="NCHW"):
        na = 3 # number of anchors (3)
        nc = 1 # number of classes (1)
        point_num = 5 # number of landmark points (5)
        no = nc + 5 + point_num * 2 # number of outputs (16)
        anchors = np.array(pred_anchor)
        anchor_vec = anchors / stride
        anchor_wh = anchor_vec.reshape(1, na, 1, 1, 2)

        if pred_format == "NCHW":
            # (1, 48, y, x) --> (1, 3, 16, y, x) --> (1, 3, y, x, 16)
            bs, _, ny, nx = pred.shape
            pred = np.transpose(pred.reshape(bs, na, no, ny, nx), (0, 1, 3, 4, 2))
        else:
            # (1, y, x, 48) --> (1, y, x, 3, 16) --> (1, 3, y, x, 16)
            bs, ny, nx, _ = pred.shape
            pred = np.transpose(pred.reshape(bs, ny, nx, na, no), (0, 3, 1, 2, 4))
        grid = np_create_grids((nx, ny))

        xy = (sigmoid(pred[..., :2]) + grid) * stride
        wh = (np.exp(pred[..., 2:4]) * anchor_wh) * stride
        lds = [((pred[..., 4 + ii*2 : 4 + (ii + 1)*2] * anchor_wh) + grid) * stride for ii in range(point_num)]
        conf = sigmoid(pred[..., 4 + point_num * 2:])
        yolo_out = np.concatenate([xy, wh, *lds, conf], -1)
        return yolo_out.reshape(bs, -1, no)
    ```
  - **Add yolodecode with tensorflow NMS function**
    ```py
    def decode_model_output_tf(preds, pred_format="NHWC"):
        if not isinstance(preds[0], np.ndarray):
            preds = [pp.numpy() for pp in preds]

        ''' Yolo decode '''
        yolo_out = [np_yolodecode(pp, aa, ss, pred_format=pred_format) for pp, aa, ss in zip(preds, pred_anchors, strides)]
        yolo_out = np.concatenate(yolo_out, 1)

        ''' NMS filter '''
        bbt = yolo_out[0, :, :4]  # bbt is [x_center, y_center, w, h]
        bbt[:, :2] = bbt[:, :2] - bbt[:, 2:] / 2
        bbt[:, 2:] = bbt[:, :2] + bbt[:, 2:]
        cct = yolo_out[0, :, 14]
        ppt = yolo_out[0, :, 4:14]

        # Input bbox should be [y1, x1, y2, x2], but [x1, y1, x2, y2] also works
        rr = tf.image.non_max_suppression(bbt, cct, max_output_size=15, iou_threshold=0.35, score_threshold=0.25).numpy()
        bbs, pps, ccs = bbt[rr], ppt[rr], cct[rr]
        return bbs, pps, ccs
    ```
  - **Inference using dynamic_axes model**
    ```py
    import onnxruntime
    sess = onnxruntime.InferenceSession('mbv3_small_1_D.onnx')

    image = plt.imread('src/test.jpg')
    imm = np.expand_dims(np.transpose(image[:480, :, :] / 255, (2, 0, 1)), axis=0)
    onnx_preds = sess.run(None, {sess.get_inputs()[0].name: imm.astype('float32')})
    print([ii.shape for ii in onnx_preds])
    # [(1, 48, 60, 128), (1, 48, 30, 64), (1, 48, 15, 32)]

    bbs, pps, ccs = decode_model_output_tf(onnx_preds, pred_format="NCHW")
    show_result(image, bbs, ccs, pps)
    ```
    ![](images/yolo_onnx_out.jpeg)
## CPP yolo decode
  ```cpp
  float score_threh = 0.15;
  std::vector<int> minsize0 = {12  , 20 , 32 };
  std::vector<int> minsize1 = {48  , 72 , 128};
  std::vector<int> minsize2 = {196 , 320, 480};

  ex.extract("s32", pred3);
  ex.extract("s16", pred2);
  ex.extract("s8", pred1);

  yolodecode(pred1, minsize0, prebox, score_threh, in.w, in.h);
  yolodecode(pred2, minsize1, prebox, score_threh, in.w, in.h);
  yolodecode(pred3, minsize2, prebox, score_threh, in.w, in.h);

  void yolodecode(ncnn::Mat pre,std::vector<int> anchor, std::vector<Bbox> & prebox, float confidence_threshold, int net_w, int net_h) {
      int w = pre.w;
      int h = pre.h;

      for (int c = 0 ; c < anchor.size() ; c++) {
           float bias = float(anchor[c]);
           int channel_cur = c * 16;
           float stridew = net_w / w;
           float strideh = net_h / h;

           // i / j 表示横 / 竖排的第 n 个框，bias 表示当前框的大小
           // 计算的结果为缩放后图像上的坐标位置
           for (int i = 0 ; i < h ; i++) {
               for (int j = 0 ; j < w ; j++) {
                   int idx = i * w + j;
                   float confidence = sigmoid(pre.channel(channel_cur + 14)[idx]);
                   // float* ptr_class_score  = pre.channel(channel_cur + 15)[idx];
                   if (confidence >= confidence_threshold) {
                          // region box
                          Bbox temp_box;
                          temp_box.s = confidence;
                          float bbox_cx = (float(j) + sigmoid(pre.channel(channel_cur + 0)[idx])) * stridew;
                          float bbox_cy = (float(i) + sigmoid(pre.channel(channel_cur + 1)[idx])) * strideh;
                          float bbox_w = static_cast<float>(exp(pre.channel(channel_cur + 2)[idx]) * bias);
                          float bbox_h = static_cast<float>(exp(pre.channel(channel_cur + 3)[idx]) * bias);

                          temp_box.x1 = bbox_cx - bbox_w * 0.5f;
                          temp_box.y1 = bbox_cy - bbox_h * 0.5f;
                          temp_box.x2 = bbox_cx + bbox_w * 0.5f;
                          temp_box.y2 = bbox_cy + bbox_h * 0.5f;

                          for (int k = 0 ; k < 5 ; k++) {
                              temp_box.point[k]._x = float(j) * stridew + pre.channel(channel_cur + 4 + k * 2)[idx] * bias;
                              temp_box.point[k]._y = float(i) * strideh + pre.channel(channel_cur + 4 + k * 2 + 1)[idx] * bias;
                          }
                          prebox.push_back(temp_box);
                   }
               }
           }
      }
  }
  ```
## Convert Tensorflow
  - **Converter source**: `/opt/anaconda3/lib/python3.7/site-packages/onnx2keras/converter.py`
  - **Q / A**
    ```py
    ''' Q:
      DEBUG:onnx2keras:mul:Convert inputs to Keras/TF layers if needed.
      ...
      ValueError: dim 1 not in the interval [-1, 0]
    '''
    ''' A:
      vi /opt/anaconda3/lib/python3.7/site-packages/onnx2keras/elementwise_layers.py
      +106     if len(input_0.shape) == 0:
      +107         input_0 = keras.backend.expand_dims(input_0, 0)
      +108     if len(input_1.shape) == 0:
      +109         input_1 = keras.backend.expand_dims(input_1, 0)
    '''
    ```
    ```py
    ''' Q:
      WARNING:onnx2keras:upsample:!!! EXPERIMENTAL SUPPORT (upsample) !!!
      ...
      AttributeError: Unsupported number of inputs
    '''
    ''' A:
      vi /opt/anaconda3/lib/python3.7/site-packages/onnx2keras/upsampling_layers.py
      21     if len(node.input) != 1:
      -22         raise AttributeError('Unsupported number of inputs')
      +22         if node.input[-1] in layers and isinstance(layers[node.input[-1]], np.ndarray):
      +23             params['scales'] = layers[node.input[-1]]
      +24         else:
      +25             raise AttributeError('Unsupported number of inputs')
    '''
    ```
  - Replace `F.interpolate size` with fixed `scale_factor` or a `torch.nn.UpsamplingNearest2d` layer
    ```py
    vi nn/models.py
    -96         up3 = F.interpolate(output3_, size=[output2_.size(2), output2_.size(3)], mode="nearest")
    +96         up3 = F.interpolate(output3_, scale_factor=2, mode="nearest")
    -100         up2 = F.interpolate(output2, size=[output1_.size(2), output1_.size(3)], mode="nearest")
    +100         up2 = F.interpolate(output2_, scale_factor=2, mode="nearest")
    ```
  - **TF 2 onnx2keras**
    ```py
    tf.__version__
    # 2.3.0
    from onnx2keras import onnx_to_keras
    import onnx
    onnx_model = onnx.load('mbv3_small_1.onnx')
    # k_model = onnx_to_keras(onnx_model, [onnx_model.graph.input[0].name], change_ordering=False)"renumerate"
    # name_policy: [None, "short" or "renumerate"], change_ordering: change ordering to HWC
    k_model = onnx_to_keras(onnx_model, [onnx_model.graph.input[0].name], name_policy="renumerate", change_ordering=True)
    print(k_model.input_shape, k_model.output_shape)
    # (None, 480, 1024, 3) [(None, 60, 128, 48), (None, 30, 64, 48), (None, 15, 32, 48)]

    k_model.save('yolo_{}.h5'.format(k_model.input_shape[1]))
    ```
  - **TF 1 onnx_tf**
    ```py
    tf.__version__
    # 1.15.0
    import onnx
    from onnx_tf.backend import prepare
    onnx_model = onnx.load('mbv3_small_1.onnx')
    tf_rep = prepare(onnx_model)
    print(tf_rep.inputs, tf_rep.outputs)
    # ['input0'] ['s8', 's16', 's32', '985']
    print(tf_rep.graph.get_operation_by_name(tf_rep.inputs[0]).outputs[0])
    # Tensor("input0:0", shape=(1, 3, 256, 160), dtype=float32)

    tf_rep.run(np.ones([1, 3, 256, 160]))
    tf_rep.export_graph("yolo.pb")
    ```
  - **Inference 480x1024 image**
    ```py
    mm = keras.models.load_model('yolo_480.h5')
    image = plt.imread('src/test.jpg')
    print(image.shape)
    # (484, 1024, 3)

    preds = mm(tf.expand_dims(image[:480, :, :] / 255.0, 0))
    print([ii.shape.as_list() for ii in preds])
    # [[1, 60, 128, 48], [1, 30, 64, 48], [1, 15, 32, 48]]

    bbs, pps, ccs = decode_model_output_tf(preds)
    print(bbs.shape, pps.shape, ccs.shape)
    # (3, 4) (3, 10) (3,)
    show_result(image, bbs, ccs, pps)
    ```
    ![](images/yolo_keras_out.jpeg)
  - **Inference 256x160 image**
    ```py
    mm = keras.models.load_model('yolo_256.h5')

    # Resize to (256, 160)
    from skimage.transform import resize
    image = plt.imread('src/test.jpg')
    iaa = resize(image, (int(image.shape[0] / image.shape[1] * 160), 160))  # Also convert pixel value to (0, 1)
    ibb = np.vstack([iaa, np.zeros([256 - iaa.shape[0], 160, 3])])
    preds = mm(tf.expand_dims(ibb, 0))
    print([ii.shape.as_list() for ii in preds])
    # [[1, 32, 20, 48], [1, 16, 10, 48], [1, 8, 5, 48]]

    bbs, pps, ccs = decode_model_output_tf(preds)
    print(bbs.shape, pps.shape, ccs.shape)
    # (2, 4) (2, 10) (2,)

    show_result(iaa, bbs, ccs, pps)
    ```
    ![](images/yolo_keras_256_out.jpeg)
## TF yolodecode layer
  ```py
  pred_anchors = np.array([[[12, 12], [20, 20], [32, 32]],
      [[48, 48], [72, 72], [128, 128]],
      [[192, 192], [320, 320], [480, 480]]])
  strides = [8, 16, 32]
  ny_256 = [32, 16, 8]
  nx_256 = [20, 10, 5]

  class YoloDecode(keras.layers.Layer):
      def __init__(self, pred_anchor, stride, orgin_input_shape, anchor_num=3, class_num=1, point_num=5, **kwargs):
          super(YoloDecode, self).__init__(**kwargs)
          self.pred_anchor, self.stride, self.orgin_input_shape = pred_anchor, stride, orgin_input_shape
          self.anchor_num, self.class_num, self.point_num = anchor_num, class_num, point_num
          self.outputs_num = class_num + 5 + point_num * 2
          self.anchor_vec = np.array(pred_anchor) / stride
          self.anchor_wh = self.anchor_vec.reshape(1, anchor_num, 1, 1, 2)
          # for orgin_input_shape = (256, 160), nx_256 = [20, 10, 5], ny_256 = [32, 16, 8]
          self.nx, self.ny = orgin_input_shape[1] // stride, orgin_input_shape[0] // stride
          self.grid = self.create_grids(self.nx, self.ny)

      def create_grids(self, nx, ny):
          # build xy offsets
          yv, xv = tf.meshgrid(tf.range(ny), tf.range(nx), indexing='ij')
          grid = tf.cast(tf.reshape(tf.stack((xv, yv), 2), (1, 1, ny, nx, 2)), "float32")
          return grid

      def call(self, inputs):
          nn = tf.reshape(inputs, (1, self.ny, self.nx, self.anchor_num, self.outputs_num))
          preds = tf.transpose(nn, (0, 3, 1, 2, 4))

          xy = (tf.nn.sigmoid(tf.gather(preds, [0, 1], axis=-1)) + self.grid) * self.stride # Center (x, y)
          wh = (tf.exp(tf.gather(preds, [2, 3], axis=-1)) * self.anchor_wh) * self.stride
          lds = [((tf.gather(preds, [4 + ii * 2, 4 + ii * 2 + 1], axis=-1) * self.anchor_wh) + self.grid) * self.stride for ii in range(self.point_num)]
          conf = tf.sigmoid(tf.gather(preds, [14, 15], axis=-1))
          yolo_out = tf.concat([xy, wh, *lds, conf], axis=-1)
          yolo_out = tf.reshape(yolo_out, (1, self.anchor_num * self.nx * self.ny, self.outputs_num))
          return yolo_out

      def compute_output_shape(self, input_shape):
          return (input_shape[0], self.anchor_num * self.nx * self.ny, self.outputs_num)
      def get_config(self):
          config = super(YoloDecode, self).get_config()
          config.update({
              "pred_anchor": self.pred_anchor,
              "stride": self.stride,
              "orgin_input_shape": self.orgin_input_shape,
              "anchor_num": self.anchor_num,
              "class_num": self.class_num,
              "point_num": self.point_num,
          })
          return config
      @classmethod
      def from_config(cls, config):
          return cls(**config)

  nns = [YoloDecode(np.array(pred_anchors[ii]), strides[ii], (256, 160)) for ii in range(3)]
  tt = keras.models.Model(mm.inputs[0], tf.concat([nns[ii](mm.outputs[ii]) for ii in range(3)], axis=1))
  print(tt(np.ones([1, 256, 160, 3])).shape)
  # (1, 2520, 16)
  ```
  ```py
  class YoloNMS(keras.layers.Layer):
      def __init__(self, max_output_size=15, iou_threshold=0.35, score_threshold=0.25, **kwargs):
          super(YoloNMS, self).__init__(**kwargs)
          self.max_output_size, self.iou_threshold, self.score_threshold = max_output_size, iou_threshold, score_threshold
          self.points_idx = np.arange(4, 14)
      def call(self, inputs):
          xyC = tf.gather(inputs, [0, 1], axis=-1)[0]
          wh = tf.gather(inputs, [2, 3], axis=-1)[0]
          xyS = xyC - wh / 2
          xyE = xyS + wh
          bbt = tf.concat([xyS, xyE], axis=-1)
          ppt = tf.gather(inputs, self.points_idx, axis=-1)[0]
          cct = tf.squeeze(tf.gather(inputs, [14], axis=-1)[0])
          rr = tf.image.non_max_suppression(bbt, cct, max_output_size=self.max_output_size, iou_threshold=self.iou_threshold, score_threshold=self.score_threshold)
          bbs, pps, ccs = tf.gather(bbt, rr, axis=0), tf.gather(ppt, rr, axis=0), tf.gather(cct, rr, axis=0)
          return bbs, pps, ccs

      # def compute_output_shape(self, input_shape):
      #     return (None, 4), (None, 10), ()
      def get_config(self):
          config = super(YoloNMS, self).get_config()
          config.update({
              "max_output_size": self.max_output_size,
              "iou_threshold": self.iou_threshold,
              "score_threshold": self.score_threshold,
          })
          return config
      @classmethod
      def from_config(cls, config):
          return cls(**config)

  nns = [YoloDecode(np.array(pred_anchors[ii]), strides[ii], (256, 160)) for ii in range(3)]
  nn = tf.concat([nns[ii](mm.outputs[ii]) for ii in range(3)], axis=1)
  nn = YoloNMS()(nn)
  tt = keras.models.Model(mm.inputs[0], nn)
  ```
## TFLite inference
  - **Convert TF 2 model to TF 1**
    ```py
    tf.__version__
    # 2.3.0
    model_name = 'yolo_256'

    mm = keras.models.load_model(model_name + '.h5')
    mm.save_weights("weights_only.h5")
    json_config = mm.to_json()
    with open('model_config.json', 'w') as json_file:
        json_file.write(json_config)

    !sed -i 's/"ragged": false, //' model_config.json
    !sed -i 's/"class_name": "Functional"/"class_name": "Model"/g' model_config.json
    !sed -i 's/"groups": 1, //g' model_config.json
    ```
  - **TFLite with xnnpack** Convert to `TF 1.15` keras model, and then `tflite model`, for `xnnpack` can only be used on this version of `tflite`
    ```py
    tf.__version__
    # 1.15.0

    ''' Save TF 1.15 keras model'''
    model_name = 'yolo_256_tf15'

    with open('model_config.json') as json_file:
        json_config = json_file.read()
    new_model = keras.models.model_from_json(json_config)
    new_model.load_weights('weights_only.h5')
    # inputs = keras.layers.Input([256, 160, 3])
    # bb = keras.models.Model(inputs, new_model(inputs))
    # new_model = bb
    new_model.save(model_name + '.h5')

    ''' Convert to TFLite float16 model '''
    converter = tf.lite.TFLiteConverter.from_keras_model_file(model_name + '.h5')
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    open(model_name + '.tflite', 'wb').write(converter.convert())

    !adb push {model_name}.tflite /data/local/tmp
    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/{model_name}.tflite --num_threads=1 --use_xnnpack=true
    # The input model file size (MB): 1.01603
    # Inference timings in us: Init: 32973, First inference: 74176, Warmup (avg): 59476.1, Inference (avg): 57819.2
    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/{model_name}.tflite --num_threads=4 --use_xnnpack=true
    # Inference timings in us: Init: 24946, First inference: 33790, Warmup (avg): 34094.2, Inference (avg): 37156.2
    ```
## Clone model convert reshape layer
  ```py
  def convert_reshape(layer):
      # print(layer.name)
      if isinstance(layer, keras.layers.Reshape):
          if len(layer.input_shape) == 4 and layer.input_shape[2:] == (1, 1) and len(layer.output_shape) == 2 and layer.output_shape[-1] == layer.input_shape[1]:
              print(">>>> Convert Reshape to Flatten <<<<")
              return keras.layers.Flatten()
          elif len(layer.output_shape) == 4 and layer.output_shape[1:3] == (1, 1) and len(layer.input_shape) == 2 and layer.input_shape[-1] == layer.output_shape[-1]:
              print(">>>> Convert Reshape to expand_dims <<<<")
              return keras.layers.Lambda(lambda xx: tf.expand_dims(tf.expand_dims(xx, 1), 1))
          else:
              print(">>>> Something else... <<<<")
              return layer
      return layer

  mm = keras.models.load_model('aa.h5', compile=False)
  mmn = keras.models.clone_model(mm, clone_function=convert_reshape)
  ```
## NCNN
  ```sh
  cd ncnn_project/
  # MODEL_NAME="small"
  MODEL_NAME="large"

  python -m onnxsim ../mbv3_${MODEL_NAME}_1.onnx $mbv3_${MODEL_NAME}_1.onnx --input-shape 1,3,256,160
  ~/workspace/ncnn/build/tools/onnx/onnx2ncnn $mbv3_${MODEL_NAME}_1.onnx

  mkdir -p ~/go/src/tdFace.mobile/assets/yoloface_${MODEL_NAME}
  cp ncnn.bin ~/go/src/tdFace.mobile/assets/yoloface_${MODEL_NAME}/ncnn.bin
  cp ncnn.param ~/go/src/tdFace.mobile/assets/yoloface_${MODEL_NAME}/ncnn.param

  ~/workspace/ncnn/build/tools/ncnnoptimize ncnn.param ncnn.bin ncnn_opt.param ncnn_opt.bin 65536
  mkdir -p ~/go/src/tdFace.mobile/assets/yoloface_${MODEL_NAME}_fp16
  cp ncnn_opt.bin ~/go/src/tdFace.mobile/assets/yoloface_${MODEL_NAME}_fp16/ncnn.bin
  cp ncnn_opt.param ~/go/src/tdFace.mobile/assets/yoloface_${MODEL_NAME}_fp16/ncnn.param
  ```
  **ARM ncnn inference**
  ```sh
  adb shell 'cd /data/mobile_test; LD_LIBRARY_PATH=./ ./test_yoloface'
  # Repeat: [20], Total: 930.81401ms, Mean: 46.500000ms

  adb shell 'cd /data/local/tmp; LD_LIBRARY_PATH=./ ./test_yoloface -model_folder assets/yoloface_mbv3_small_1_final -num_threads 1 -max_border_size 320'
  adb shell 'cd /data/local/tmp; LD_LIBRARY_PATH=./ ./test_yoloface -model_folder assets/yoloface_mbv3_small_1_final -num_threads 4 -max_border_size 320'
  adb shell 'cd /data/local/tmp; LD_LIBRARY_PATH=./ ./test_yoloface -model_folder assets/yoloface_mbv3_small_1_final_fp16 -num_threads 1 -max_border_size 320'
  adb shell 'cd /data/local/tmp; LD_LIBRARY_PATH=./ ./test_yoloface -model_folder assets/yoloface_mbv3_small_1_final_fp16 -num_threads 4 -max_border_size 320'

  adb shell 'cd /data/local/tmp; LD_LIBRARY_PATH=./ ./test_yoloface -model_folder assets/yoloface_mbv3_large_1_final -num_threads 1 -max_border_size 320'
  adb shell 'cd /data/local/tmp; LD_LIBRARY_PATH=./ ./test_yoloface -model_folder assets/yoloface_mbv3_large_1_final -num_threads 4 -max_border_size 320'
  adb shell 'cd /data/local/tmp; LD_LIBRARY_PATH=./ ./test_yoloface -model_folder assets/yoloface_mbv3_large_1_final_fp16 -num_threads 1 -max_border_size 320'
  adb shell 'cd /data/local/tmp; LD_LIBRARY_PATH=./ ./test_yoloface -model_folder assets/yoloface_mbv3_large_1_final_fp16 -num_threads 4 -max_border_size 320'
  ```

  | model               | num_threads | max_border_size | Time mean   | max_border_size | Time mean    |
  | ------------------- | ----------- | --------------- | ----------- | --------------- | ------------ |
  | yoloface_small      | 1           | 256             | 43.450001ms | 320             | 55.950001ms  |
  | yoloface_small      | 4           | 256             | 25.500000ms | 320             | 30.299999ms  |
  | yoloface_small_fp16 | 1           | 256             | 41.150002ms | 320             | 51.950001ms  |
  | yoloface_small_fp16 | 4           | 256             | 24.000000ms | 320             | 28.799999ms  |
  | yoloface_large      | 1           | 256             | 98.650002ms | 320             | 140.000000ms |
  | yoloface_large      | 4           | 256             | 48.049999ms | 320             | 66.500000ms  |
  | yoloface_large_fp16 | 1           | 256             | 93.949997ms | 320             | 129.050003ms |
  | yoloface_large_fp16 | 4           | 256             | 45.549999ms | 320             | 61.299999ms  |

  | max_border_size | Time mean   |
  | --------------- | ----------- |
  | 320             | 28.100000ms |
  | 352             | 32.450001ms |
  | 384             | 40.750000ms |
  | 416             | 44.950001ms |
  | 480             | 57.450001ms |

  | model         | large, threads 1 | large, threads 4 | small, threads 1 | small, threads 4 |
  | ------------- | ---------------- | ---------------- | ---------------- | ---------------- |
  | 1             | 136.899994ms     | 64.199997ms      | 54.150002ms      | 29.250000ms      |
  | 1_light       | 125.500000ms     | 55.900002ms      | 44.650002ms      | 21.850000ms      |
  | 75            | 114.349998ms     | 55.400002ms      | 50.549999ms      | 29.299999ms      |
  | 75_light      | 102.050003ms     | 47.700001ms      | 38.750000ms      | 21.600000ms      |
  | 1_fp16        | 128.550003ms     | 60.250000ms      | 51.650002ms      | 28.549999ms      |
  | 1_light_fp16  | 117.550003ms     | 51.700001ms      | 39.349998ms      | 21.250000ms      |
  | 75_fp16       | 106.400002ms     | 52.650002ms      | 47.450001ms      | 27.000000ms      |
  | 75_light_fp16 | 94.949997ms      | 43.750000ms      | 35.799999ms      | 19.500000ms      |
***

# Centerface
## ONNX inference
  - [Github CenterFace](https://github.com/Star-Clouds/CenterFace.git)
  ```py
  !cd prj-python

  import cv2
  import scipy.io as sio
  import os
  from centerface import CenterFace
  cf = CenterFace(landmarks=True)
  imm = cv2.imread('000388.jpg')
  h, w, _ = imm.shape
  # h, w = 480, 640
  dets, lms = cf(imm, h, w, threshold=0.35)
  print(dets.shape, lms.shape)
  # (146, 5) (146, 10)
  ```
  ```py
  !cd prj-python

  import cv2
  frame = cv2.imread('000388.jpg')
  print(frame.shape)
  # (672, 1024, 3)

  frame = frame[:672, :, :]
  blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0, size=frame.shape[:2], mean=(0, 0, 0), swapRB=True, crop=False)
  print(blob.shape)
  # (1, 3, 1024, 672)

  net = cv2.dnn.readNetFromONNX('../onnx/centerface.onnx')
  net.setInput(blob)
  heatmap, scale, offset, lms = net.forward(["537", "538", "539", '540'])
  print(heatmap.shape, scale.shape, offset.shape, lms.shape)
  # (1, 1, 256, 168) (1, 2, 256, 168) (1, 2, 256, 168) (1, 10, 256, 168)
  ```
## Convert Tensorflow
  - **Convert ONNX model to keras**
    ```py
    !cd models/keras

    from onnx2keras import onnx_to_keras
    import onnx
    onnx_model = onnx.load('../onnx/centerface.onnx')
    # mm = onnx_to_keras(onnx_model, [onnx_model.graph.input[0].name], change_ordering=False)"renumerate"
    # name_policy: [None, "short" or "renumerate"], change_ordering: change ordering to HWC
    mm = onnx_to_keras(onnx_model, [onnx_model.graph.input[0].name], name_policy="renumerate", change_ordering=True)
    print(mm.input_shape, mm.output_shape)
    # (None, 32, 32, 3) [(None, 8, 8, 1), (None, 8, 8, 2), (None, 8, 8, 2), (None, 8, 8, 10)]

    mm.save('centerface.h5')

    ''' Change input shape '''
    inputs = keras.layers.Input([256, 160, 3])
    output = mm(inputs)
    mm = keras.models.Model(inputs, output)
    ```
  - **TFLite with xnnpack** same with `Yoloface` [TFLite inference](#tflite-inference)
    ```py
    ''' Test basic input 32x32 '''
    model_name = 'centerface_tf15'

    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/{model_name}.tflite --num_threads=4 --use_xnnpack=true
    # The input model file size (MB): 3.68274
    # Inference timings in us: Init: 65956, First inference: 9017, Warmup (avg): 5473.92, Inference (avg): 5393.91
    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/{model_name}.tflite --num_threads=1 --use_xnnpack=true
    # Inference timings in us: Init: 56500, First inference: 4736, Warmup (avg): 4483.37, Inference (avg): 4462.15

    ''' Test input 256x160 '''
    model_name = 'centerface_256_tf15'

    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/{model_name}.tflite --num_threads=1 --use_xnnpack=true
    # The input model file size (MB): 3.68608
    # Inference timings in us: Init: 64183, First inference: 132582, Warmup (avg): 126273, Inference (avg): 125876
    !adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/{model_name}.tflite --num_threads=4 --use_xnnpack=true
    # Inference timings in us: Init: 59274, First inference: 71562, Warmup (avg): 64821.5, Inference (avg): 62149.7
    ```
## CPP
  - **MNN**
    ```sh
    cd prj-mnn/
    MNN_HOME=$HOME/workspace/MNN
    g++ demo.cpp cpp/centerface.cpp -Icpp -I$MNN_HOME/include -L$MNN_HOME/build -lMNN -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui
    ./a.out ../models/mnn/ ~/workspace/test_images/1.jpg
    ```
  - **NCNN**
    ```sh
    cd prj-ncnn
    NCNN_HOME=$HOME/workspace/ncnn/build/install
    g++ demo.cpp cpp/ncnn_centerface.cpp -Icpp -I$NCNN_HOME/include/ncnn -L$NCNN_HOME/lib -lncnn -lgomp -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -I$HOME/go/src/tdFace.mobile/test -DOPENCV
    ./a.out ../models/ncnn/ ~/workspace/test_images/1.jpg
    ```
    ```sh
    python -m onnxsim ../onnx/centerface_bnmerged.onnx centerface_bnmerged.onnx

    ~/workspace/ncnn/build/tools/onnx/onnx2ncnn ../onnx/centerface_bnmerged.onnx
    ~/workspace/ncnn/build/tools/ncnnoptimize ncnn.param ncnn.bin ncnn_opt.param ncnn_opt.bin 65536
    cp ncnn.* ~/go/src/tdFace.mobile/assets/centerface
    cp ncnn_opt.bin ~/go/src/tdFace.mobile/assets/centerface_fp16/ncnn.bin
    cp ncnn_opt.param ~/go/src/tdFace.mobile/assets/centerface_fp16/ncnn.param

    adb push ~/go/src/tdFace.mobile/assets/centerface* /data/local/tmp/assets
    adb shell 'cd /data/local/tmp; LD_LIBRARY_PATH=. ./benchncnn'
    # assets/yoloface_small/ncnn  min =   19.45  max =   37.71  avg =   26.06
    # assets/yoloface_small_fp16/ncnn  min =   18.26  max =   23.39  avg =   19.57
    # assets/yoloface_large/ncnn  min =   46.97  max =   74.83  avg =   53.94
    # assets/yoloface_large_fp16/ncnn  min =   42.72  max =   71.46  avg =   56.13
    # assets/centerface/ncnn  min =   58.27  max =   86.82  avg =   68.09
    # assets/centerface_fp16/ncnn  min =   58.51  max =   82.09  avg =   65.98
    ```
  - **ARM32 ncnn inference time test**
    ```sh
    SDK_HOME="$HOME/Android/Sdk/ndk/21.0.6113669/toolchains/llvm/prebuilt/linux-x86_64"
    alias AG="$SDK_HOME/bin/armv7a-linux-androideabi29-clang++"
    # alias AG="$SDK_HOME/bin/aarch64-linux-android29-clang++"

    NCNN_HOME=$HOME/workspace/ncnn/build-armv7/install
    AG demo.cpp cpp/ncnn_centerface.cpp -Icpp -I$NCNN_HOME/include/ncnn -L$NCNN_HOME/lib -lncnn -lomp -I$HOME/go/src/tdFace.mobile/test -llog -landroid

    adb push $SDK_HOME/lib64/clang/9.0.8/lib/linux/arm/libomp.so /data/local/tmp
    adb push ../models/ncnn ~/workspace/test_images/1.jpg /data/local/tmp
    adb push a.out /data/local/tmp
    adb shell 'cd /data/local/tmp; LD_LIBRARY_PATH=. ./a.out ncnn 1.jpg'
    # [480, 256] Repeat: [20], Total: 3159.758000ms, Mean: 157.987900ms
    # [320, 256] Repeat: [20], Total: 2153.169000ms, Mean: 107.658450ms
    # [320, 224] Repeat: [20], Total: 1891.514000ms, Mean: 94.575700ms
    # [320, 160] Repeat: [20], Total: 1288.198000ms, Mean: 64.409900ms
    # [256, 160] Repeat: [20], Total: 1015.478000ms, Mean: 50.773900ms
    # [256, 128] Repeat: [20], Total: 873.331000ms, Mean: 43.666550ms
    # [64, 64] Repeat: [20], Total: 152.362000ms, Mean: 7.618100ms
    # [32, 32] Repeat: [20], Total: 72.518000ms, Mean: 3.625900ms
    ```
***

# Retinaface
## Pytorch
  ```py
  !cd /media/DATA/workspace/retinaface_collection/tensorrtx/retinaface/Pytorch_Retinaface

  import torch
  from models.retinaface import RetinaFace
  cfg_re50 = {
   'name': 'Resnet50',
   'min_sizes': [[16, 32], [64, 128], [256, 512]],
   'steps': [8, 16, 32],
   'variance': [0.1, 0.2],
   'clip': False,
   'loc_weight': 2.0,
   'gpu_train': True,
   'batch_size': 24,
   'ngpu': 4,
   'epoch': 100,
   'decay1': 70,
   'decay2': 90,
   'image_size': 840,
   'pretrain': True,
   'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
   'in_channel': 256,
   'out_channel': 256
  }

  device = torch.device("cpu")
  net = RetinaFace(cfg=cfg_re50, phase='test').to(device)
  weights = './weights/Resnet50_Final.pth'
  net.load_state_dict(torch.load(weights, map_location=device)['model'])

  pretrained_path = './weights/Resnet50_Final.pth'
  pretrained_dict = torch.load(pretrained_path, map_location=device)
  def remove_prefix(state_dict, prefix):
      print('remove prefix \'{}\''.format(prefix))
      f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
      return {f(key): value for key, value in state_dict.items()}
  pretrained_dict = remove_prefix(pretrained_dict, 'module.')

  net.load_state_dict(pretrained_dict)
  net.eval()
  inputs = torch.randn(1, 3, 1024, 768).to(device)
  torch.onnx.export(net, inputs, 'retinaface.onnx', verbose=False)

  import onnxruntime
  sess = onnxruntime.InferenceSession('retinaface.onnx')
  print([(ii.name, ii.shape) for ii in sess.get_outputs()])
  print([(ii.name, ii.shape) for ii in sess.get_inputs()])
  onnx_preds = sess.run(None, {sess.get_inputs()[0].name: imm.astype('float32')})
  ```
## Keras
  - [Github bubbliiiing/retinaface-keras](https://github.com/bubbliiiing/retinaface-keras)
  ```py
  from nets.retinaface import RetinaFace
  from utils.config import cfg_mnet, cfg_re50
  # rr = RetinaFace(cfg_mnet, backbone="mobilenet", input_shape=(256, 160, 3))
  # rr.load_weights('model_data/retinaface_mobilenet025.h5', by_name=True)
  rr = RetinaFace(cfg_re50, backbone="resnet50", input_shape=(256, 160, 3), use_unsampling2d=True)
  rr.load_weights('./model_data/resnet50_fixed_2.h5')
  rr.save('model_data/retinaface_resnet50_unsampling_256.h5')
  ```
  ```py
  from tensorflow.keras.applications.imagenet_utils import preprocess_input
  from skimage.transform import resize

  mm = keras.models.load_model('model_data/retinaface_resnet50_unsampling_256.h5', custom_objects={'relu6': tf.nn.relu6})
  # image = resize(image[:, :, :3], (256, 160))
  image = imread('./img/street.jpg')

  ipp = image.copy()
  im_height, im_width = image.shape[:2]

  anchors = Anchors(image_size=(im_height, im_width)).get_anchors()
  confidence = 0.5
  nms_threshhold = 0.45
  bbox_scale = np.repeat([(ipp.shape[1], ipp.shape[0])], 2, axis=0).flatten()
  landmark_scale = np.repeat([(ipp.shape[1], ipp.shape[0])], 5, axis=0).flatten()

  # 图片预处理，归一化
  photo = np.expand_dims(preprocess_input(ipp), 0)

  preds = mm.predict(photo)
  # 将预测结果进行解码和非极大抑制
  results = detection_out(preds, anchors, confidence_threshold=confidence, nms_threshhold=nms_threshhold)
  results = np.array(results)
  results[:,:4] = results[:,:4] * bbox_scale
  results[:,5:] = results[:,5:] * landmark_scale

  bbs, ccs, pps = results[:, :4], results[:, 4], results[:, 5:]

  plt.imshow(image.astype('uint8'))
  for bb, cc, pp in zip(bbs, ccs, pps):
      plt.plot([bb[0], bb[2], bb[2], bb[0], bb[0]], [bb[1], bb[1], bb[3], bb[3], bb[1]])
      plt.text(bb[0], bb[1], '{:.4f}'.format(cc), color='r')
      plt.scatter(pp[::2], pp[1::2], s=4, c='g')
  ```
  ```py
  import itertools
  class Anchors(object):
      def __init__(self, image_size=None):
          super(Anchors, self).__init__()
          self.min_sizes = [[16, 32], [64, 128], [256, 512]]
          self.steps = [8, 16, 32]
          self.clip = False
          self.image_size = image_size
          self.feature_maps = [[int(np.ceil(self.image_size[0]/step)), int(np.ceil(self.image_size[1]/step))] for step in self.steps]

      def get_anchors(self):
          anchors = []
          for k, f in enumerate(self.feature_maps):
              min_sizes = self.min_sizes[k]
              # 每个网格点2个先验框，都是正方形
              for i, j in itertools.product(range(f[0]), range(f[1])):
                  for min_size in min_sizes:
                      s_kx = min_size / self.image_size[1]
                      s_ky = min_size / self.image_size[0]
                      dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                      dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                      for cy, cx in itertools.product(dense_cy, dense_cx):
                          anchors += [cx, cy, s_kx, s_ky]

          anchors = np.reshape(anchors,[-1,4])

          output = np.zeros_like(anchors[:,:4])
          output[:,0] = anchors[:,0] - anchors[:,2]/2
          output[:,1] = anchors[:,1] - anchors[:,3]/2
          output[:,2] = anchors[:,0] + anchors[:,2]/2
          output[:,3] = anchors[:,1] + anchors[:,3]/2

          if self.clip:
              output = np.clip(output, 0, 1)
          return output

  def detection_out(predictions, mbox_priorbox, confidence_threshold=0.4, nms_threshhold=0.45):  
      # 网络预测的结果
      mbox_loc = predictions[0][0]
      # 置信度
      mbox_conf = predictions[1][0][:,1:2]
      # ldm的调整情况
      mbox_ldm = predictions[2][0]

      decode_bbox = decode_boxes(mbox_loc, mbox_ldm, mbox_priorbox)
      conf_mask = (mbox_conf >= confidence_threshold)[:,0]
      detection = np.concatenate((decode_bbox[conf_mask][:,:4], mbox_conf[conf_mask], decode_bbox[conf_mask][:,4:]), -1)

      best_box = []
      scores = detection[:,4]
      # 根据得分对该种类进行从大到小排序。
      arg_sort = np.argsort(scores)[::-1]
      detection = detection[arg_sort]
      while np.shape(detection)[0]>0:
          # 每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
          best_box.append(detection[0])
          if len(detection) == 1:
              break
          ious = iou(best_box[-1],detection[1:])
          detection = detection[1:][ious < nms_threshhold]
      return best_box

  def decode_boxes(mbox_loc, mbox_ldm, mbox_priorbox):
      # 获得先验框的宽与高
      prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
      prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
      # 获得先验框的中心点
      prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
      prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])

      # 真实框距离先验框中心的xy轴偏移情况
      decode_bbox_center_x = mbox_loc[:, 0] * prior_width * 0.1
      decode_bbox_center_x += prior_center_x
      decode_bbox_center_y = mbox_loc[:, 1] * prior_height * 0.1
      decode_bbox_center_y += prior_center_y

      # 真实框的宽与高的求取
      decode_bbox_width = np.exp(mbox_loc[:, 2] * 0.2)
      decode_bbox_width *= prior_width
      decode_bbox_height = np.exp(mbox_loc[:, 3] * 0.2)
      decode_bbox_height *= prior_height

      # 获取真实框的左上角与右下角
      decode_bbox_xmin = decode_bbox_center_x - 0.5 * decode_bbox_width
      decode_bbox_ymin = decode_bbox_center_y - 0.5 * decode_bbox_height
      decode_bbox_xmax = decode_bbox_center_x + 0.5 * decode_bbox_width
      decode_bbox_ymax = decode_bbox_center_y + 0.5 * decode_bbox_height

      prior_width = np.expand_dims(prior_width,-1)
      prior_height = np.expand_dims(prior_height,-1)
      prior_center_x = np.expand_dims(prior_center_x,-1)
      prior_center_y = np.expand_dims(prior_center_y,-1)

      mbox_ldm = mbox_ldm.reshape([-1,5,2])
      decode_ldm = np.zeros_like(mbox_ldm)
      decode_ldm[:,:,0] = np.repeat(prior_width,5,axis=-1)*mbox_ldm[:,:,0]*0.1 + np.repeat(prior_center_x,5,axis=-1)
      decode_ldm[:,:,1] = np.repeat(prior_height,5,axis=-1)*mbox_ldm[:,:,1]*0.1 + np.repeat(prior_center_y,5,axis=-1)


      # 真实框的左上角与右下角进行堆叠
      decode_bbox = np.concatenate((decode_bbox_xmin[:, None],
                                      decode_bbox_ymin[:, None],
                                      decode_bbox_xmax[:, None],
                                      decode_bbox_ymax[:, None],
                                      np.reshape(decode_ldm,[-1,10])), axis=-1)
      # 防止超出0与1
      decode_bbox = np.minimum(np.maximum(decode_bbox, 0.0), 1.0)
      return decode_bbox

  def iou(b1,b2):
      b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
      b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

      inter_rect_x1 = np.maximum(b1_x1, b2_x1)
      inter_rect_y1 = np.maximum(b1_y1, b2_y1)
      inter_rect_x2 = np.minimum(b1_x2, b2_x2)
      inter_rect_y2 = np.minimum(b1_y2, b2_y2)

      inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                   np.maximum(inter_rect_y2 - inter_rect_y1, 0)

      area_b1 = (b1_x2-b1_x1)*(b1_y2-b1_y1)
      area_b2 = (b2_x2-b2_x1)*(b2_y2-b2_y1)

      iou = inter_area/np.maximum((area_b1+area_b2-inter_area),1e-6)
      return iou
  ```
***

# Face detection test
  ```py
  os.chdir("../../../samba/tdFace-flask/mtcnn_tf/")
  import mtcnn
  mtcnn_det = mtcnn.MTCNN('./mtcnn.pb')
  mtcnn_det.detect_faces(imm)

  import insightface
  retina = insightface.model_zoo.face_detection.retinaface_mnet025_v1()
  retina.prepare(-1)
  retina.detect(imm)
  ```

  - **MTCNN python**
    ```py
    os.chdir('/home/leondgarse/workspace/samba/tdFace-flask')
    from mtcnn_tf.mtcnn import MTCNN

    pp = imread('test_images/test.jpg')
    det = MTCNN('./mtcnn_tf/mtcnn.pb')
    det.detect_faces(pp)
    ```
    ```py
    confid = [0.935214, 0.9988366, 0.778714]
    bbox   = [[304.18033, 381.37274, 461.27203, 514.9812], [357.4963, 813.8469, 418.02625, 870.40283], [623.5862, 1228.5925, 746.287, 1334.8964]]
    points = [[371.39066, 379.2191, 406.99472, 425.24304, 430.6188, 442.9482, 496.701, 472.00586, 433.493, 477.85278],
              [385.03262, 389.30566, 400.0184, 404.95398, 409.36685, 834.60486, 854.2477, 842.1936, 829.3925, 847.8495],
              [667.3448, 662.90393, 680.22015, 715.79315, 711.9303, 1278.7151, 1313.0042, 1310.4624, 1294.1819, 1314.4312]]
    ```
  - **Retinaface mnet v2**
    ```py
    confid = [0.995438, 0.968334, 0.935214]
    bbox   = [[301.314, 377.009, 468.395, 517.275], [360.384, 810.457, 428.049, 875.668], [383.878, 918.502, 426.882, 947.455]]
    points = [[412.175, 437.242, 609.626, 675.408, 698.357, 435.543, 713.442, 610.528, 383.064, 625.876],
              [382.554, 411.982, 478.701, 491.152, 514.878, 793.743, 919.821, 834.382, 776.301, 863.163],
              [407.271, 414.886, 455.501, 481.512, 489.468, 961.242, 956.157, 980.694, 934.81, 930.51]]
    ```
  - **MTCNN ncnn**
    ```py
    bbox   = [[291, 353, 469, 531], [353, 810, 421, 877]]
    points = [[369.34027, 373.8462, 412.09048, 427.1006, 431.2583, 439.1975, 496.24152, 475.05768, 426.5965, 476.0255],
              [377.24957, 379.67755, 394.86682, 404.6796, 408.17218, 831.5927, 854.9071, 839.1629, 827.1213, 849.32294]]
    ```
  - **Retinaface mnet v1**
    ```py
    confid = [0.99876, 0.970946, 0.968176]
    bbox   = [[301.613, 370.449, 464.16, 519.373], [389.298, 915.537, 428.462, 945.48], [365.157, 808.332, 427.836, 870.429]]
    points = [[374.37, 378.498, 415.735, 425.11, 429.21, 438.062, 495.49, 473.448, 425.179, 472.81],
              [402.625, 405.244, 413.758, 417.896, 420.327, 939.76, 938.98, 942.569, 934.049, 932.757],
              [388.867, 391.66, 405.041, 411.323, 413.503, 825.623, 849.172, 833.371, 824.378, 841.637]]
    ```
  - **Fast MTCNN**
    ```py
    bbox   = [[300, 365, 465, 519], [358, 804, 426, 868]]
    points = [[351.832, 358.374, 392.036, 422.125, 425.595, 426.558, 493.061, 470.133, 425.86, 477.172],
              [380.635, 378.517, 393.588, 402.717, 400.555, 821.393, 848.771, 837.606, 824.914, 850.827]]
    ```
  - **TensorRT Resnet50 Retinaface**
    ```py
    # image resize to 640 x 480
    confid = [0.999367, 0.9837, 0.292366]
    bbox   = [[297.329, 368.313, 460.907, 508.905], [631.101, 1209.59, 729.906, 1295.55], [383.694, 910.691, 419.172, 940.814]]
    points = [[366.921, 373.209, 407.907, 422.979, 428.457, 445.047, 491.792, 476.919, 435.203, 475.148],
              [660.883, 668.741, 676.638, 697.016, 702.062, 1226.69, 1245.87, 1215.45, 1216.04, 1227.2],
              [403.335, 403.363, 411.059, 416.415, 416.208, 919.986, 933.348, 927.906, 922.223, 932.094]]
    ```
    ```py
    # image resize to 1024 x 768
    confid = [0.998839, 0.995743, 0.972942, 0.704532]
    bbox   = [[298.86, 369.962, 461.99, 518.878], [628.826, 1206.82, 727.035, 1296.12], [365.398, 807.16, 426.747, 870.231], [384.857, 918.394, 422.749, 945.925]]
    points = [[366.801, 378.701, 402.8, 422.121, 431.771, 436.892, 495.485, 471.716, 433.112, 476.609],
              [658.944, 671.21, 680.043, 697.777, 706.938, 1223.22, 1247.03, 1212.29, 1212.06, 1226.47],
              [387.325, 396.311, 405.779, 408.413, 415.295, 830.972, 852.654, 837.531, 823.55, 838.6],
              [401.003, 404.001, 411.397, 415.328, 417.592, 939.79, 941.663, 942.861, 934.488, 934.689]]
    ```
***

# Face recognition test
  ```py
  import glob2
  import insightface
  from sklearn.preprocessing import normalize

  imms = glob2.glob('./*.jpg')
  imgs = [imread(ii)[:, :, :3] for ii in imms]
  det = insightface.model_zoo.face_detection.retinaface_mnet025_v1()
  det.prepare(-1)
  idds = {nn: ii for nn, ii in zip(imms, imgs)}
  dds = {nn: det.detect(ii[:, :, ::-1]) for nn, ii in zip(imms, imgs)}

  nimgs = np.array([face_align_landmarks_sk(idds[kk], vv[1])[0] for kk, vv in dds.items() if len(vv[1]) != 0])
  plt.imshow(np.hstack(nimgs))
  plt.tight_layout()
  nimgs_norm = (nimgs[:, :, :, :3] - 127.5) / 127

  ees = normalize(mm(nimgs_norm))
  np.dot(ees, ees.T)
  [kk for kk, vv in dds.items() if len(vv[1]) != 0]

  mm = face_model.FaceModel()
  ees = normalize(mm.interp(nimgs_norm))
  ```
  ```py
  mm = keras.models.load_model("../Keras_insightface/checkpoints/mobilenet_adamw_BS256_E80_arc_trip128_basic_agedb_30_epoch_89_batch_15000_0.953333.h5")
  mm = keras.models.load_model("../Keras_insightface/checkpoints/mobilenet_adamw_BS256_E80_arc_trip_basic_agedb_30_epoch_114_batch_5000_0.954500.h5")
  mm = keras.models.load_model("../Keras_insightface/checkpoints/mobilenet_adamw_BS256_E80_arc_c64_basic_agedb_30_epoch_103_batch_5000_0.953667.h5")
  ```
  ```py
  mmns = [
      "T_mobilenetv3L_adamw_5e5_arc_trip64_BS1024_basic_agedb_30_epoch_125_batch_2000_0.953833.h5",
      "T_mobilenet_adamw_5e5_arc_trip64_BS1024_basic_agedb_30_epoch_114_batch_4000_0.952000.h5",
      "mobilenet_adamw_BS256_E80_arc_tripD_basic_agedb_30_epoch_123_0.955333.h5",
      "keras_se_mobile_facenet_emore_triplet_basic_agedb_30_epoch_100_0.958333.h5",
      "keras_se_mobile_facenet_emore_IV_basic_agedb_30_epoch_48_0.957833.h5",
  ]
  for mmn in mmns:
      mm = keras.models.load_model("../Keras_insightface/checkpoints/" + mmn)
      ees = normalize(mm(nimgs_norm))
      np.dot(ees, ees.T)
      print(">>>>", mmn)
      print(np.dot(ees, ees.T))
  ```
  ```py
  # Resnest101
  array([[1.        , 0.1558015 , 0.15456064, 0.13313395, 0.16850811, 0.71167467],
         [0.1558015 , 1.        , 0.19925219, 0.20134368, 0.37451308, 0.15747676],
         [0.15456064, 0.19925219, 1.        , 0.33097531, 0.60421063, 0.18203224],
         [0.13313395, 0.20134368, 0.33097531, 1.        , 0.26904165, 0.19666012],
         [0.16850811, 0.37451308, 0.60421063, 0.26904165, 1.        , 0.28212154],
         [0.71167467, 0.15747676, 0.18203224, 0.19666012, 0.28212154, 1.        ]])

  # Mobilenet
  array([[1.        , 0.29774572, 0.09815505, 0.15720329, 0.17226598, 0.72166827],
         [0.29774572, 1.        , 0.34869616, 0.1973212 , 0.45947366, 0.25320448],
         [0.09815505, 0.34869616, 1.        , 0.26055445, 0.5881257 , 0.18348087],
         [0.15720329, 0.1973212 , 0.26055445, 1.        , 0.30415184, 0.24423493],
         [0.17226598, 0.45947366, 0.5881257 , 0.30415184, 1.        , 0.14440171],
         [0.72166827, 0.25320448, 0.18348087, 0.24423493, 0.14440171, 1.        ]])

  # MXNet resnet101
  array([[0.9999998 , 0.24837211, 0.1303953 , 0.21556515, 0.23038697, 0.7216288 ],
         [0.24837211, 1.        , 0.34982952, 0.17353785, 0.3337898 , 0.28773898],
         [0.1303953 , 0.34982952, 1.        , 0.21442819, 0.71804786, 0.13172513],
         [0.21556515, 0.17353785, 0.21442819, 1.        , 0.21935105, 0.24629946],
         [0.23038697, 0.3337898 , 0.71804786, 0.21935105, 1.0000001 , 0.20667821],
         [0.7216288 , 0.28773898, 0.13172513, 0.24629946, 0.20667821, 1.0000001 ]])
  ```
  ```py
  # MXNet resnet101
  array([[0.9999999 , 0.12421323, 0.36465803, 0.47377837, 0.21408823, 0.35779008, 0.23447886, 0.30594954, 0.54214454],
         [0.12421323, 0.9999999 , 0.2692185 , 0.07800525, 0.12620914, 0.24759631, 0.32376158, 0.30534476, 0.07428315],
         [0.36465803, 0.2692185 , 0.99999994, 0.30244786, 0.2776268 , 0.7250889 , 0.12520632, 0.19832084, 0.4338043 ],
         [0.47377837, 0.07800525, 0.30244786, 1.0000001 , 0.17683765, 0.31931424, 0.20238054, 0.20936331, 0.5031358 ],
         [0.21408823, 0.12620914, 0.2776268 , 0.17683765, 1.0000002 , 0.2316848 , 0.25115657, 0.22379392, 0.20974487],
         [0.35779008, 0.24759631, 0.7250889 , 0.31931424, 0.2316848 , 0.9999999 , 0.12169009, 0.22123177, 0.4395604 ],
         [0.23447886, 0.32376158, 0.12520632, 0.20238054, 0.25115657, 0.12169009, 1.0000001 , 0.7024463 , 0.17882678],
         [0.30594954, 0.30534476, 0.19832084, 0.20936331, 0.22379392, 0.22123177, 0.7024463 , 0.99999994, 0.19508064],
         [0.54214454, 0.07428315, 0.4338043 , 0.5031358 , 0.20974487, 0.4395604 , 0.17882678, 0.19508064, 1.        ]])

  # Resnest101
  array([[1.        , 0.09653112, 0.34567068, 0.35957315, 0.09873741, 0.3647217 , 0.32881509, 0.32723336, 0.51869996],
         [0.09653112, 1.        , 0.14862806, 0.20942087, 0.1459171 , 0.11868739, 0.15710884, 0.29315572, 0.06172519],
         [0.34567068, 0.14862806, 1.        , 0.23773459, 0.20453881, 0.64531339, 0.15824395, 0.25539223, 0.35883342],
         [0.35957315, 0.20942087, 0.23773459, 1.        , 0.34161757, 0.27761874, 0.06426631, 0.11394879, 0.29227217],
         [0.09873741, 0.1459171 , 0.20453881, 0.34161757, 1.        , 0.10085443, 0.33029246, 0.26362839, 0.20034649],
         [0.3647217 , 0.11868739, 0.64531339, 0.27761874, 0.10085443, 1.        , 0.12520004, 0.12369476, 0.2990151 ],
         [0.32881509, 0.15710884, 0.15824395, 0.06426631, 0.33029246, 0.12520004, 1.        , 0.60281276, 0.16362239],
         [0.32723336, 0.29315572, 0.25539223, 0.11394879, 0.26362839, 0.12369476, 0.60281276, 1.        , 0.17242818],
         [0.51869996, 0.06172519, 0.35883342, 0.29227217, 0.20034649, 0.2990151 , 0.16362239, 0.17242818, 1.        ]])

  # Mobilenet triplet 128
  array([[1.        , 0.41836702, 0.41450289, 0.5285159 , 0.39815743, 0.59143796, 0.39294739, 0.39449755, 0.58467363],
         [0.41836702, 1.        , 0.41646123, 0.3481795 , 0.34727307, 0.36440841, 0.47861499, 0.62125792, 0.25343142],
         [0.41450289, 0.41646123, 1.        , 0.30024778, 0.39975276, 0.78776638, 0.24419089, 0.27779964, 0.32785577],
         [0.5285159 , 0.3481795 , 0.30024778, 1.        , 0.28179755, 0.39157149, 0.22987079, 0.32495614, 0.52379275],
         [0.39815743, 0.34727307, 0.39975276, 0.28179755, 1.        , 0.44091393, 0.35736952, 0.41333531, 0.40843123],
         [0.59143796, 0.36440841, 0.78776638, 0.39157149, 0.44091393, 1.        , 0.33504632, 0.34104602, 0.52528929],
         [0.39294739, 0.47861499, 0.24419089, 0.22987079, 0.35736952, 0.33504632, 1.        , 0.68957433, 0.40338012],
         [0.39449755, 0.62125792, 0.27779964, 0.32495614, 0.41333531, 0.34104602, 0.68957433, 1.        , 0.43240437],
         [0.58467363, 0.25343142, 0.32785577, 0.52379275, 0.40843123, 0.52528929, 0.40338012, 0.43240437, 1.        ]])

  # Mobilenet triplet 64
  array([[1.        , 0.31283942, 0.36502096, 0.40771537, 0.29296445, 0.52262759, 0.19741207, 0.1872439 , 0.43291409],
         [0.31283942, 1.        , 0.24724888, 0.2878407 , 0.24942819, 0.3370249 , 0.44580004, 0.45603498, 0.21947548],
         [0.36502096, 0.24724888, 1.        , 0.32684813, 0.2597726 , 0.6709674 , 0.1321767 , 0.2345897 , 0.2343204 ],
         [0.40771537, 0.2878407 , 0.32684813, 1.        , 0.22050977, 0.31049397, 0.21690768, 0.25340733, 0.39517784],
         [0.29296445, 0.24942819, 0.2597726 , 0.22050977, 1.        , 0.18286361, 0.29566625, 0.36705299, 0.25452527],
         [0.52262759, 0.3370249 , 0.6709674 , 0.31049397, 0.18286361, 1.        , 0.12788039, 0.2380372 , 0.29876387],
         [0.19741207, 0.44580004, 0.1321767 , 0.21690768, 0.29566625, 0.12788039, 1.        , 0.64417225, 0.27175968],
         [0.1872439 , 0.45603498, 0.2345897 , 0.25340733, 0.36705299, 0.2380372 , 0.64417225, 1.        , 0.29206764],
         [0.43291409, 0.21947548, 0.2343204 , 0.39517784, 0.25452527, 0.29876387, 0.27175968, 0.29206764, 1.        ]])

  # Mobilenet arc only
  array([[1.        , 0.27447427, 0.33093477, 0.44366778, 0.12657129, 0.42529421, 0.23941177, 0.14257544, 0.45117096],
         [0.27447427, 1.        , 0.20500087, 0.23143572, 0.19713553, 0.30256486, 0.37154384, 0.41321225, 0.1484183 ],
         [0.33093477, 0.20500087, 1.        , 0.31254969, 0.25402285, 0.70438819, 0.14027923, 0.13860752, 0.20007214],
         [0.44366778, 0.23143572, 0.31254969, 1.        , 0.13716888, 0.27382582, 0.22073898, 0.18395958, 0.36093491],
         [0.12657129, 0.19713553, 0.25402285, 0.13716888, 1.        , 0.14737819, 0.22610791, 0.31317051, 0.17809208],
         [0.42529421, 0.30256486, 0.70438819, 0.27382582, 0.14737819, 1.        , 0.09306854, 0.15856636, 0.20832577],
         [0.23941177, 0.37154384, 0.14027923, 0.22073898, 0.22610791, 0.09306854, 1.        , 0.56017446, 0.26547573],
         [0.14257544, 0.41321225, 0.13860752, 0.18395958, 0.31317051, 0.15856636, 0.56017446, 1.        , 0.16801733],
         [0.45117096, 0.1484183 , 0.20007214, 0.36093491, 0.17809208, 0.20832577, 0.26547573, 0.16801733, 1.        ]])

  # res2-6-10-2-dim256
  array([[ 0.99999976,  0.02275476,  0.19566718,  0.39551505,  0.11382857, 0.2418991 ,  0.1621489 ,  0.20770606,  0.54685557],
         [ 0.02275476,  1.0000001 ,  0.18565616,  0.12622942,  0.07543389, 0.15768072,  0.18987912,  0.24687825,  0.00755703],
         [ 0.19566718,  0.18565616,  1.        ,  0.1135577 ,  0.17233118, 0.64797974,  0.13736339,  0.15125085,  0.28536692],
         [ 0.39551505,  0.12622942,  0.1135577 ,  0.9999999 ,  0.1203843 , 0.1491866 ,  0.13995048,  0.10306323,  0.40521592],
         [ 0.11382857,  0.07543389,  0.17233118,  0.1203843 ,  0.9999999 , 0.19507906,  0.21430498,  0.28658438,  0.23534644],
         [ 0.2418991 ,  0.15768072,  0.64797974,  0.1491866 ,  0.19507906, 1.0000001 , -0.01007168,  0.14095978,  0.29420048],
         [ 0.1621489 ,  0.18987912,  0.13736339,  0.13995048,  0.21430498, -0.01007168,  1.0000002 ,  0.58440447,  0.11922266],
         [ 0.20770606,  0.24687825,  0.15125085,  0.10306323,  0.28658438, 0.14095978,  0.58440447,  0.99999994,  0.11037809],
         [ 0.54685557,  0.00755703,  0.28536692,  0.40521592,  0.23534644, 0.29420048,  0.11922266,  0.11037809,  1.        ]])

  # res4-8-46-4-dim256
  array([[ 0.9999999 ,  0.02085986,  0.21933755,  0.3805598 ,  0.14618638, 0.26807183,  0.18817256,  0.2006281 ,  0.5925854 ],
         [ 0.02085986,  0.99999976,  0.10873765,  0.05561617,  0.0148335 , 0.10101061,  0.19646968,  0.25535637, -0.0167736 ],
         [ 0.21933755,  0.10873765,  1.        ,  0.20362249,  0.3145762 , 0.54968023,  0.11722751,  0.14334169,  0.3327023 ],
         [ 0.3805598 ,  0.05561617,  0.20362249,  0.99999994,  0.10594524, 0.14630523,  0.14106289,  0.18696074,  0.33167878],
         [ 0.14618638,  0.0148335 ,  0.3145762 ,  0.10594524,  0.9999999 , 0.10847133,  0.17860007,  0.21755098,  0.24645992],
         [ 0.26807183,  0.10101061,  0.54968023,  0.14630523,  0.10847133, 1.        , -0.05039748,  0.04296025,  0.30741566],
         [ 0.18817256,  0.19646968,  0.11722751,  0.14106289,  0.17860007, -0.05039748,  0.99999994,  0.67188215,  0.1417823 ],
         [ 0.2006281 ,  0.25535637,  0.14334169,  0.18696074,  0.21755098, 0.04296025,  0.67188215,  1.0000001 ,  0.20449154],
         [ 0.5925854 , -0.0167736 ,  0.3327023 ,  0.33167878,  0.24645992, 0.30741566,  0.1417823 ,  0.20449154,  1.        ]])
  ```
  ```py
  >>>> T_mobilenetv3L_adamw_5e5_arc_trip64_BS1024_basic_agedb_30_epoch_125_batch_2000_0.953833.h5
  [[ 1.00000000e+00  4.29695178e-01  7.41433247e-01 -7.58692589e-02 2.86694661e-01  3.34029962e-01  6.29330699e-01  1.17840146e-01 3.66203455e-02  2.99279494e-01  6.30978385e-02  6.59875585e-02 5.94344468e-02  3.53909248e-01  6.03454735e-01]
   [ 4.29695178e-01  1.00000000e+00  3.88810500e-01  9.24215575e-02 4.15714712e-01  4.63776141e-01  2.79021334e-01  1.93106090e-01 6.14039079e-02  4.36266536e-01  3.05092272e-01  2.72503414e-01 2.36863946e-02  6.27761959e-01  3.04508917e-01]
   [ 7.41433247e-01  3.88810500e-01  1.00000000e+00 -8.04484453e-02 3.43371120e-01  3.93705692e-01  6.68397522e-01  1.49977210e-01 5.28993824e-03  2.39942221e-01  9.20759722e-02  1.52024917e-01 -2.34210041e-02  3.01573152e-01  4.68741484e-01]
   [-7.58692589e-02  9.24215575e-02 -8.04484453e-02  1.00000000e+00 2.52538475e-01  1.14712040e-01 -2.36838923e-02  2.59420105e-01 7.84408310e-02  2.51792454e-01  2.80058815e-01  4.23255287e-01 1.51056557e-01  1.32943884e-01  2.65491689e-02]
   [ 2.86694661e-01  4.15714712e-01  3.43371120e-01  2.52538475e-01 1.00000000e+00  2.90129739e-01  2.17371170e-01  2.82333790e-01 -5.69787095e-02  7.85161600e-01  1.66002096e-01  2.51595070e-01 2.42157661e-02  3.74642881e-01  2.56471164e-01]
   [ 3.34029962e-01  4.63776141e-01  3.93705692e-01  1.14712040e-01 2.90129739e-01  1.00000000e+00  2.86759557e-01  2.77230969e-01 2.14522410e-01  2.02422667e-01  1.55801281e-01  1.49006801e-01 1.26711035e-01  4.63698546e-01  2.58617719e-01]
   [ 6.29330699e-01  2.79021334e-01  6.68397522e-01 -2.36838923e-02 2.17371170e-01  2.86759557e-01  1.00000000e+00  8.41929262e-02 1.78889130e-01  1.38994904e-01  6.16348781e-02  1.53765124e-01 1.03469038e-01  1.58587708e-01  5.00574712e-01]
   [ 1.17840146e-01  1.93106090e-01  1.49977210e-01  2.59420105e-01 2.82333790e-01  2.77230969e-01  8.41929262e-02  1.00000000e+00 4.31928564e-02  2.13714179e-01  2.72829371e-01  2.02678373e-01 2.12908894e-01  1.79022369e-01  1.73207374e-01]
   [ 3.66203455e-02  6.14039079e-02  5.28993824e-03  7.84408310e-02 -5.69787095e-02  2.14522410e-01  1.78889130e-01  4.31928564e-02 1.00000000e+00 -5.68161294e-02  6.44153122e-04 -2.58861422e-02 5.50812631e-01  6.06255037e-02  9.14468134e-02]
   [ 2.99279494e-01  4.36266536e-01  2.39942221e-01  2.51792454e-01 7.85161600e-01  2.02422667e-01  1.38994904e-01  2.13714179e-01 -5.68161294e-02  1.00000000e+00  1.30310857e-01  2.53251215e-01 -4.04040813e-02  4.23246367e-01  3.45164837e-01]
   [ 6.30978385e-02  3.05092272e-01  9.20759722e-02  2.80058815e-01 1.66002096e-01  1.55801281e-01  6.16348781e-02  2.72829371e-01 6.44153122e-04  1.30310857e-01  1.00000000e+00  6.27957001e-01 9.73341356e-02  3.44253814e-01  9.72629729e-02]
   [ 6.59875585e-02  2.72503414e-01  1.52024917e-01  4.23255287e-01 2.51595070e-01  1.49006801e-01  1.53765124e-01  2.02678373e-01 -2.58861422e-02  2.53251215e-01  6.27957001e-01  1.00000000e+00 5.20221954e-02  2.71369178e-01  1.03349736e-01]
   [ 5.94344468e-02  2.36863946e-02 -2.34210041e-02  1.51056557e-01 2.42157661e-02  1.26711035e-01  1.03469038e-01  2.12908894e-01 5.50812631e-01 -4.04040813e-02  9.73341356e-02  5.20221954e-02 1.00000000e+00 -5.65003059e-02  5.94529807e-02]
   [ 3.53909248e-01  6.27761959e-01  3.01573152e-01  1.32943884e-01 3.74642881e-01  4.63698546e-01  1.58587708e-01  1.79022369e-01 6.06255037e-02  4.23246367e-01  3.44253814e-01  2.71369178e-01 -5.65003059e-02  1.00000000e+00  2.86708117e-01]
   [ 6.03454735e-01  3.04508917e-01  4.68741484e-01  2.65491689e-02 2.56471164e-01  2.58617719e-01  5.00574712e-01  1.73207374e-01 9.14468134e-02  3.45164837e-01  9.72629729e-02  1.03349736e-01 5.94529807e-02  2.86708117e-01  1.00000000e+00]]

  >>>> T_mobilenet_adamw_5e5_arc_trip64_BS1024_basic_agedb_30_epoch_114_batch_4000_0.952000.h5
  [[ 1.00000000e+00  3.40127229e-01  7.42367761e-01  9.77221874e-02 2.27505597e-01  3.38114298e-01  6.62165021e-01  2.86542628e-01 6.27376177e-02  2.64824894e-01  8.98881711e-02  1.96263220e-01 1.38312027e-01  2.59788144e-01  5.67146784e-01]
   [ 3.40127229e-01  1.00000000e+00  4.63760455e-01  1.78377113e-01 4.41031952e-01  5.63879258e-01  3.30656585e-01  2.71374052e-01 2.83991623e-02  4.92803750e-01  2.83944627e-01  3.10024507e-01 -2.52927263e-03  4.23298203e-01  2.14691822e-01]
   [ 7.42367761e-01  4.63760455e-01  1.00000000e+00  1.19289468e-01 2.12030153e-01  3.79822104e-01  7.05061793e-01  1.98442319e-01 2.69456223e-02  2.59063466e-01  8.65879339e-02  1.54904172e-01 -3.60709414e-03  1.97223187e-01  4.65837005e-01]
   [ 9.77221874e-02  1.78377113e-01  1.19289468e-01  1.00000000e+00 1.94240239e-01  2.80580214e-01  1.82881875e-01  1.33683145e-01 2.35610243e-01  3.02979263e-01  3.36326864e-01  4.61680298e-01 1.55318379e-01  1.54522633e-01  1.95128360e-01]
   [ 2.27505597e-01  4.41031952e-01  2.12030153e-01  1.94240239e-01 1.00000000e+00  3.65266644e-01  1.52581522e-01  3.64339901e-01 2.33655351e-02  7.48629914e-01  2.40550686e-01  2.99422519e-01 1.00063718e-02  3.46281858e-01  1.19450206e-01]
   [ 3.38114298e-01  5.63879258e-01  3.79822104e-01  2.80580214e-01 3.65266644e-01  1.00000000e+00  4.16392402e-01  3.88906795e-01 2.51990080e-01  4.97731379e-01  2.86117684e-01  3.66880718e-01 7.92414701e-02  4.10441979e-01  3.53212579e-01]
   [ 6.62165021e-01  3.30656585e-01  7.05061793e-01  1.82881875e-01 1.52581522e-01  4.16392402e-01  1.00000000e+00  1.44187902e-01 2.07984983e-01  1.73525770e-01  2.82954211e-02  1.17332713e-01 1.20478080e-01  1.42130694e-01  5.43879167e-01]
   [ 2.86542628e-01  2.71374052e-01  1.98442319e-01  1.33683145e-01 3.64339901e-01  3.88906795e-01  1.44187902e-01  1.00000000e+00 -6.04172017e-02  4.03472338e-01  3.23831760e-01  3.09279216e-01 3.66082570e-02  3.64953442e-01  1.27283955e-01]
   [ 6.27376177e-02  2.83991623e-02  2.69456223e-02  2.35610243e-01 2.33655351e-02  2.51990080e-01  2.07984983e-01 -6.04172017e-02 1.00000000e+00  5.83566348e-02  6.13434470e-02  5.84754874e-02 6.74269715e-01  5.26982486e-02  2.11072771e-01]
   [ 2.64824894e-01  4.92803750e-01  2.59063466e-01  3.02979263e-01 7.48629914e-01  4.97731379e-01  1.73525770e-01  4.03472338e-01 5.83566348e-02  1.00000000e+00  2.76139975e-01  4.28514901e-01 -6.98304466e-04  4.47151176e-01  1.57057293e-01]
   [ 8.98881711e-02  2.83944627e-01  8.65879339e-02  3.36326864e-01 2.40550686e-01  2.86117684e-01  2.82954211e-02  3.23831760e-01 6.13434470e-02  2.76139975e-01  1.00000000e+00  6.30786455e-01 -5.77923624e-02  2.58146191e-01  7.82584319e-02]
   [ 1.96263220e-01  3.10024507e-01  1.54904172e-01  4.61680298e-01 2.99422519e-01  3.66880718e-01  1.17332713e-01  3.09279216e-01 5.84754874e-02  4.28514901e-01  6.30786455e-01  1.00000000e+00 4.80950608e-03  2.77168455e-01  1.49742426e-01]
   [ 1.38312027e-01 -2.52927263e-03 -3.60709414e-03  1.55318379e-01 1.00063718e-02  7.92414701e-02  1.20478080e-01  3.66082570e-02 6.74269715e-01 -6.98304466e-04 -5.77923624e-02  4.80950608e-03 1.00000000e+00 -5.17101664e-02  4.35576539e-02]
   [ 2.59788144e-01  4.23298203e-01  1.97223187e-01  1.54522633e-01 3.46281858e-01  4.10441979e-01  1.42130694e-01  3.64953442e-01 5.26982486e-02  4.47151176e-01  2.58146191e-01  2.77168455e-01 -5.17101664e-02  1.00000000e+00  1.68841703e-01]
   [ 5.67146784e-01  2.14691822e-01  4.65837005e-01  1.95128360e-01 1.19450206e-01  3.53212579e-01  5.43879167e-01  1.27283955e-01 2.11072771e-01  1.57057293e-01  7.82584319e-02  1.49742426e-01 4.35576539e-02  1.68841703e-01  1.00000000e+00]]

  >>>> mobilenet_adamw_BS256_E80_arc_tripD_basic_agedb_30_epoch_123_0.955333.h5
  [[ 1.          0.41995046  0.72045217  0.14276859  0.17353927  0.25630641 0.72570569  0.31259563  0.14555155  0.31257568  0.22750018  0.22630342 0.13320194  0.45702378  0.57295065]
   [ 0.41995046  1.          0.4343083   0.25114369  0.3457262   0.41600223 0.3831861   0.20682873  0.09373844  0.52028971  0.17897687  0.21537132 0.08827312  0.54672629  0.26120827]
   [ 0.72045217  0.4343083   1.          0.05976454  0.1291929   0.28230939 0.73592407  0.14744738  0.15618992  0.28359686  0.13008813  0.10878404 0.07561229  0.36095247  0.43146426]
   [ 0.14276859  0.25114369  0.05976454  1.          0.28050121  0.18802487 0.12192914  0.19248151  0.1614934   0.28875296  0.35986638  0.45585616 0.22853697  0.16217371  0.24177532]
   [ 0.17353927  0.3457262   0.1291929   0.28050121  1.          0.25971277 0.15233854  0.26623758  0.08054116  0.727568    0.11586712  0.18685093 0.07124111  0.27318993  0.26957086]
   [ 0.25630641  0.41600223  0.28230939  0.18802487  0.25971277  1.  0.28283108  0.15134039  0.23407477  0.25908109  0.13018991  0.19951519 0.06343216  0.40855905  0.41172223]
   [ 0.72570569  0.3831861   0.73592407  0.12192914  0.15233854  0.28283108 1.          0.2351004   0.28629809  0.19598139  0.17258618  0.20843697 0.22102642  0.35971177  0.56085978]
   [ 0.31259563  0.20682873  0.14744738  0.19248151  0.26623758  0.15134039 0.2351004   1.          0.11567138  0.23213774  0.28010928  0.32430955 0.16913989  0.23879342  0.19629286]
   [ 0.14555155  0.09373844  0.15618992  0.1614934   0.08054116  0.23407477 0.28629809  0.11567138  1.          0.06126115 -0.01562102  0.02848535 0.56563011  0.20275155  0.24363542]
   [ 0.31257568  0.52028971  0.28359686  0.28875296  0.727568    0.25908109 0.19598139  0.23213774  0.06126115  1.          0.1488665   0.21466271 -0.01354127  0.3675338   0.30999004]
   [ 0.22750018  0.17897687  0.13008813  0.35986638  0.11586712  0.13018991 0.17258618  0.28010928 -0.01562102  0.1488665   1.          0.65452039 0.03520288  0.22426077  0.19856089]
   [ 0.22630342  0.21537132  0.10878404  0.45585616  0.18685093  0.19951519 0.20843697  0.32430955  0.02848535  0.21466271  0.65452039  1.  0.08200381  0.28916965  0.22271837]
   [ 0.13320194  0.08827312  0.07561229  0.22853697  0.07124111  0.06343216 0.22102642  0.16913989  0.56563011 -0.01354127  0.03520288  0.08200381 1.          0.08978504  0.11078503]
   [ 0.45702378  0.54672629  0.36095247  0.16217371  0.27318993  0.40855905 0.35971177  0.23879342  0.20275155  0.3675338   0.22426077  0.28916965 0.08978504  1.          0.4145425 ]
   [ 0.57295065  0.26120827  0.43146426  0.24177532  0.26957086  0.41172223 0.56085978  0.19629286  0.24363542  0.30999004  0.19856089  0.22271837 0.11078503  0.4145425   1.        ]]

  >>>> keras_se_mobile_facenet_emore_triplet_basic_agedb_30_epoch_100_0.958333.h5
  [[1.         0.61478385 0.8973454  0.15718107 0.5926555  0.58813182 0.92300232 0.45091071 0.3164522  0.5968713  0.37408453 0.30994958 0.40522699 0.65396296 0.83518201]
   [0.61478385 1.         0.66360599 0.36545365 0.67394194 0.5910911 0.66880568 0.51033195 0.3596981  0.69880752 0.56744638 0.45521595 0.41459763 0.72988871 0.55831343]
   [0.8973454  0.66360599 1.         0.17525107 0.5584644  0.68718486 0.86782025 0.50264027 0.33017282 0.53835818 0.39080338 0.32353151 0.36372143 0.67215952 0.80071578]
   [0.15718107 0.36545365 0.17525107 1.         0.31412549 0.40733733 0.25785307 0.43937926 0.38964902 0.31744369 0.58116134 0.61701738 0.39894503 0.39162109 0.32306835]
   [0.5926555  0.67394194 0.5584644  0.31412549 1.         0.50954788 0.62572916 0.60725355 0.1909226  0.89496291 0.54618607 0.45310902 0.35389693 0.74746891 0.51459188]
   [0.58813182 0.5910911  0.68718486 0.40733733 0.50954788 1.  0.65470876 0.51524153 0.51829348 0.47027365 0.37630522 0.40607065 0.42410082 0.68561522 0.65426112]
   [0.92300232 0.66880568 0.86782025 0.25785307 0.62572916 0.65470876 1.         0.50959772 0.43159469 0.61265041 0.43510244 0.35097861 0.46370049 0.66316858 0.81414746]
   [0.45091071 0.51033195 0.50264027 0.43937926 0.60725355 0.51524153 0.50959772 1.         0.24266423 0.55925443 0.51402934 0.40525766 0.42429972 0.58113889 0.48454077]
   [0.3164522  0.3596981  0.33017282 0.38964902 0.1909226  0.51829348 0.43159469 0.24266423 1.         0.15768315 0.26993046 0.28630734 0.75963755 0.20986197 0.36556155]
   [0.5968713  0.69880752 0.53835818 0.31744369 0.89496291 0.47027365 0.61265041 0.55925443 0.15768315 1.         0.53241526 0.45763469 0.34710827 0.72942182 0.55309736]
   [0.37408453 0.56744638 0.39080338 0.58116134 0.54618607 0.37630522 0.43510244 0.51402934 0.26993046 0.53241526 1.         0.749129 0.31426637 0.56523865 0.40658124]
   [0.30994958 0.45521595 0.32353151 0.61701738 0.45310902 0.40607065 0.35097861 0.40525766 0.28630734 0.45763469 0.749129   1.  0.26827176 0.46411074 0.41938556]
   [0.40522699 0.41459763 0.36372143 0.39894503 0.35389693 0.42410082 0.46370049 0.42429972 0.75963755 0.34710827 0.31426637 0.26827176 1.         0.27497352 0.40184703]
   [0.65396296 0.72988871 0.67215952 0.39162109 0.74746891 0.68561522 0.66316858 0.58113889 0.20986197 0.72942182 0.56523865 0.46411074 0.27497352 1.         0.64524276]
   [0.83518201 0.55831343 0.80071578 0.32306835 0.51459188 0.65426112 0.81414746 0.48454077 0.36556155 0.55309736 0.40658124 0.41938556 0.40184703 0.64524276 1.        ]]

  >>>> keras_se_mobile_facenet_emore_IV_basic_agedb_30_epoch_48_0.957833.h5
  [[1.         0.52452019 0.79764335 0.01922559 0.34479842 0.41821113 0.83882856 0.17354656 0.18959271 0.36957028 0.14057233 0.10494926 0.27162717 0.49565217 0.65373726]
   [0.52452019 1.         0.57140449 0.18228006 0.47906092 0.44123847 0.5275844  0.20980341 0.20691296 0.46903743 0.34897468 0.2765814 0.194298   0.61766128 0.39762093]
   [0.79764335 0.57140449 1.         0.07529915 0.30094927 0.53102282 0.75710164 0.24882642 0.14580214 0.28065893 0.16415297 0.1474415 0.17211926 0.49134935 0.60505851]
   [0.01922559 0.18228006 0.07529915 1.         0.20317396 0.28491703 0.10819204 0.20830958 0.25527885 0.21926467 0.37335406 0.42986572 0.30167764 0.23739102 0.20924116]
   [0.34479842 0.47906092 0.30094927 0.20317396 1.         0.29358603 0.39203468 0.44237078 0.09325511 0.81655135 0.26954425 0.26361311 0.25743398 0.5550894  0.3253638 ]
   [0.41821113 0.44123847 0.53102282 0.28491703 0.29358603 1.  0.4785178  0.27395942 0.36109232 0.23813071 0.16109122 0.2450794 0.2005094  0.57975659 0.49980962]
   [0.83882856 0.5275844  0.75710164 0.10819204 0.39203468 0.4785178 1.         0.25265195 0.29839689 0.37215297 0.1977397  0.17022776 0.34383761 0.487475   0.65087879]
   [0.17354656 0.20980341 0.24882642 0.20830958 0.44237078 0.27395942 0.25265195 1.         0.01756419 0.35887119 0.35585282 0.28417292 0.20165476 0.31740708 0.22444148]
   [0.18959271 0.20691296 0.14580214 0.25527885 0.09325511 0.36109232 0.29839689 0.01756419 1.         0.04743121 0.13746886 0.15323205 0.71083048 0.12089917 0.23476041]
   [0.36957028 0.46903743 0.28065893 0.21926467 0.81655135 0.23813071 0.37215297 0.35887119 0.04743121 1.         0.23741816 0.25527783 0.20521458 0.54611248 0.35884441]
   [0.14057233 0.34897468 0.16415297 0.37335406 0.26954425 0.16109122 0.1977397  0.35585282 0.13746886 0.23741816 1.         0.69237788 0.18012348 0.29819382 0.17399155]
   [0.10494926 0.2765814  0.1474415  0.42986572 0.26361311 0.2450794 0.17022776 0.28417292 0.15323205 0.25527783 0.69237788 1.  0.15727922 0.2623953  0.2742645 ]
   [0.27162717 0.194298   0.17211926 0.30167764 0.25743398 0.2005094 0.34383761 0.20165476 0.71083048 0.20521458 0.18012348 0.15727922 1.         0.13275842 0.26789568]
   [0.49565217 0.61766128 0.49134935 0.23739102 0.5550894  0.57975659 0.487475   0.31740708 0.12089917 0.54611248 0.29819382 0.2623953 0.13275842 1.         0.47580607]
   [0.65373726 0.39762093 0.60505851 0.20924116 0.3253638  0.49980962 0.65087879 0.22444148 0.23476041 0.35884441 0.17399155 0.2742645 0.26789568 0.47580607 1.        ]]
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

# Tests
## Image rotate
```py
@tf.function
def image_rotation(imm, degree):
    if degree == 90:
        return np.transpose(imm[::-1, :, :], (1, 0, 2))
    if degree == 180:
        return imm[::-1, ::-1, :]
    if degree == 270:
        return np.transpose(imm[:, ::-1, :], (1, 0, 2))
    return imm

plt.imshow(image_rotation(imm, 90))
plt.imshow(image_rotation(imm, 180))
plt.imshow(image_rotation(imm, 270))
```
```py
mm3 = keras.Sequential([
    keras.layers.Input((None, None, 3)),
    keras.preprocessing.image.apply_affine_transform,
    mm,
    keras.layers.Lambda(tf.nn.l2_normalize, name='norm_embedding', arguments={'axis': 1})
])

converter = tf.lite.TFLiteConverter.from_keras_model(mm3)
tflite_model = converter.convert()
open('./norm_model_tf2.tflite', 'wb').write(tflite_model)

inputs = keras.layers.Input([None, None, 3])
nn = keras.preprocessing.image.apply_affine_transform(inputs)
mm = keras.models.Model(inputs, nn)
```
```py
@tf.function
def images_funcs(image, trans, type):
    ret = image
    type = type[0]
    if type == 0:
        # Resize
        ret = keras.layers.experimental.preprocessing.Resizing(trans[0], trans[1])
    elif type == 1:
        # Rotate
        angle = trans[0]
        if angle == 90:
            ret = image[::-1, :, :].transpose(1, 0, 2)
        elif angle == 180:
            ret = imm[::-1, ::-1, :]
        elif angle == 270:
            ret = imm[:, ::-1, :].transpose(1, 0, 2)
    elif type == 2:
        # Affine
        ret = keras.preprocessing.image.apply_affine_transform(image, *trans)
    return ret

image_input = keras.layers.Input([None, None, 3])
trans_input = keras.layers.Input([6])
type_input = keras.layers.Input(None)
```
```py
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


theta = tform.rotation / np.pi * 180
_tx, _ty = tform.translation
tx = np.cos(theta) * _tx + np.sin(theta) * _ty
ty = np.cos(theta) * _ty - np.sin(theta) * _tx
# tx, ty = _tx, _ty
zx, zy = tform.scale, tform.scale
transform_matrix = None

if theta != 0:
    theta = np.deg2rad(theta)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])
    transform_matrix = rotation_matrix

if tx != 0 or ty != 0:
    # np.cos(theta), -np.sin(theta), np.cos(theta) * tx - np.sin(theta) * ty
    # np.sin(theta), np.cos(theta), np.sin(theta) * tx + np.cos(theta) * ty
    # 0, 0, 1
    shift_matrix = np.array([[1, 0, tx],
                             [0, 1, ty],
                             [0, 0, 1]])
    if transform_matrix is None:
        transform_matrix = shift_matrix
    else:
        transform_matrix = np.dot(transform_matrix, shift_matrix)

if shear != 0:
    shear = np.deg2rad(shear)
    shear_matrix = np.array([[1, -np.sin(shear), 0],
                             [0, np.cos(shear), 0],
                             [0, 0, 1]])
    if transform_matrix is None:
        transform_matrix = shear_matrix
    else:
        transform_matrix = np.dot(transform_matrix, shear_matrix)

if zx != 1 or zy != 1:
    # np.cos(theta) * zx, -np.sin(theta) * zy, np.cos(theta) * tx - np.sin(theta) * ty
    # np.sin(theta) * zx, np.cos(theta) * zy, np.sin(theta) * tx + np.cos(theta) * ty
    # 0, 0, 1
    zoom_matrix = np.array([[zx, 0, 0],
                            [0, zy, 0],
                            [0, 0, 1]])
    if transform_matrix is None:
        transform_matrix = zoom_matrix
    else:
        transform_matrix = np.dot(transform_matrix, zoom_matrix)

if transform_matrix is not None:
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(
        transform_matrix, h, w)
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]

    channel_images = [ndimage.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=order,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
return x

```
# 几何变换 geometric transformations
## Face align landmarks
  ```py
  from skimage.transform import SimilarityTransform
  import cv2

  def face_align_landmarks(img, landmarks, image_size=(112, 112)):
      ret = []
      for landmark in landmarks:
          # landmark = np.array(landmark).reshape(2, 5)[::-1].T
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
  ```
  ```py
  from skimage import transform
  def face_align_landmarks_sk(img, landmarks, image_size=(112, 112), method='similar', show=True):
      tform = transform.AffineTransform() if method == 'affine' else transform.SimilarityTransform()
      src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
      ret, nns = [], []
      for landmark in landmarks:
          # landmark = np.array(landmark).reshape(2, 5)[::-1].T
          tform.estimate(landmark, src)
          ret.append(transform.warp(img, tform.inverse, output_shape=image_size))
      ret = (np.array(ret) * 255).astype(np.uint8)

      return (np.array(ret) * 255).astype(np.uint8)

  def face_align_landmarks_sk(img, landmarks, image_size=(112, 112), method='similar', order=1, show=True):
      tform = transform.AffineTransform() if method == 'affine' else transform.SimilarityTransform()
      src = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]], dtype=np.float32)
      ret, nns = [], []
      for landmark in landmarks:
          # landmark = np.array(landmark).reshape(2, 5)[::-1].T
          tform.estimate(src, landmark)
          ret.append(transform.warp(img, tform, output_shape=image_size, order=order))
          if show:
              nns.append(tform.inverse(landmark))

      ret = (np.array(ret) * 255).astype(np.uint8)
      if show:
          plt.figure()
          plt.imshow(np.hstack(ret))
          for id, ii in enumerate(nns):
              plt.scatter(ii[:, 0] + image_size[0] * id, ii[:, 1], c='r', s=8)
      return ret
  ```
  ```py
  mm3 = keras.Sequential([
      keras.layers.Input((None, None, 3)),
      keras.preprocessing.image.apply_affine_transform,
      mm,
      keras.layers.Lambda(tf.nn.l2_normalize, name='norm_embedding', arguments={'axis': 1})
  ])

  converter = tf.lite.TFLiteConverter.from_keras_model(mm3)
  tflite_model = converter.convert()
  open('./norm_model_tf2.tflite', 'wb').write(tflite_model)

  inputs = keras.layers.Input([None, None, 3])
  nn = keras.preprocessing.image.apply_affine_transform(inputs)
  mm = keras.models.Model(inputs, nn)
  ```
## ndimage affine_transform
  - 对于 `skimage.transform` 生成的转换矩阵，`ndimage.interpolation.affine_transform` 在使用时，需要将横纵坐标上的变幻对调
  - **变换 tform.parameters**
    - `rotation` 改为反向旋转
    - `translation` 对调 `xy` 变换值
    ```py
    from scipy import ndimage

    tform = transform.SimilarityTransform()
    tform.estimate(src, pps[0])

    # tt = transform.SimilarityTransform(rotation=tform.rotation*-1, scale=tform.scale, translation=tform.translation[::-1]).params
    tt = tform.params.copy()
    tt[0, -1], tt[1, -1] = tt[1, -1], tt[0, -1]
    tt[0, 1], tt[1, 0] = -1 * tt[0, 1], -1 * tt[1, 0]
    channel_images = [ndimage.interpolation.affine_transform(
        imm[:, :, ii],
        tt,
        output_shape=(112, 112),
        order=1,
        mode='nearest',
        cval=0) for ii in range(3)]
    x = np.stack(channel_images, axis=-1)
    plt.imshow(x)
    ```
  - **变换时图像转置** 使用原值的 `tform.parameters`，转置图像的宽高
    ```py
    from scipy import ndimage

    tform = transform.SimilarityTransform()
    tform.estimate(src, pps[0])

    channel_images = [ndimage.interpolation.affine_transform(
        imm[:, :, ii].T,
        tform.params,
        output_shape=(112, 112),
        order=1,
        mode='nearest',
        cval=0) for ii in range(3)]
    x = np.stack(channel_images, axis=-1)
    x = np.transpose(x, (1, 0, 2))
    plt.imshow(x)
    ```
  - **生成转置的变换矩阵** `skimage.transform` `estimate` 的参数值对调 `xy` 坐标值
    ```py
    tform.estimate(src[:, ::-1], pps[0][:, ::-1])
    channel_axis = 2
    x = np.rollaxis(imm, channel_axis, 0)  # (976, 1920, 3) --> (3, 976, 1920)
    channel_images = [ndimage.interpolation.affine_transform(
        x_channel,
        tform.params,
        output_shape=(112, 112),
        order=1,
        mode='nearest',
        cval=0) for x_channel in x]
    x = np.stack(channel_images, axis=0)  # (3, 112, 112)
    x = np.rollaxis(x, 0, channel_axis + 1) # (112, 112, 3)
    plt.imshow(x)
    ```
## affine_transform 图像缩放
  ```py
  scale = 3
  tt = transform.SimilarityTransform(scale=scale, translation=[0, 0]).params
  channel_images = [ndimage.interpolation.affine_transform(
      imm[:, :, ii],
      tt,
      output_shape=(imm.shape[0] // scale, imm.shape[1] // scale),
      order=1,
      mode='nearest',
      cval=0) for ii in range(3)]
  x = np.stack(channel_images, axis=-1)
  plt.imshow(x)
  ```
## affine_transform 图像旋转
  ```py
  theta = 90 # [90, 180, 270]
  rotation = theta / 180 * np.pi
  # translation=[imm.shape[0] * abs(cos(rotation)), imm.shape[1] * abs(sin(rotation))]
  if theta == 90:
      translation=[imm.shape[0], 0]
      output_shape = imm.shape[:2][::-1]
  elif theta == 180:
      translation=imm.shape[:2]
      output_shape = imm.shape[:2]
  elif theta == 270:
      translation=[0, imm.shape[1]]
      output_shape = imm.shape[:2][::-1]

  tt = transform.SimilarityTransform(rotation=rotation, translation=translation).params
  channel_images = [ndimage.interpolation.affine_transform(
      imm[:, :, ii],
      tt,
      output_shape=output_shape,
      order=1,
      mode='nearest',
      cval=0) for ii in range(3)]
  x = np.stack(channel_images, axis=-1)
  plt.imshow(x)
  ```
## TF function
  ```py
  class WarpAffine(keras.layers.Layer):
      def __call__(self, imm, tformP, output_shape):
          rets = []
          for xx in imm:
              x = tf.transpose(xx, (2, 0, 1))
              channel_images = [ndimage.interpolation.affine_transform(
                  x_channel,
                  tformP,
                  output_shape=output_shape,
                  order=1,
                  mode='nearest',
                  cval=0) for x_channel in x]
              x = tf.stack(channel_images, axis=0)
              x = tf.transpose(x, (1, 2, 0))
              rets.append(x)
          return rets
  ```
## Rotation
```go
func (nnInterpolator) transform_RGBA_RGBA_Src(dst *image.RGBA, dr, adr image.Rectangle, d2s *f64.Aff3, src *image.RGBA, sr image.Rectangle, bias image.Point) {
    for dy := int32(adr.Min.Y); dy < int32(adr.Max.Y); dy++ {
        dyf := float64(dr.Min.Y+int(dy)) + 0.5
        d := (dr.Min.Y+int(dy)-dst.Rect.Min.Y)*dst.Stride + (dr.Min.X+adr.Min.X-dst.Rect.Min.X)*4
        for dx := int32(adr.Min.X); dx < int32(adr.Max.X); dx, d = dx+1, d+4 {
            dxf := float64(dr.Min.X+int(dx)) + 0.5
            sx0 := int(d2s[0]*dxf+d2s[1]*dyf+d2s[2]) + bias.X
            sy0 := int(d2s[3]*dxf+d2s[4]*dyf+d2s[5]) + bias.Y
            if !(image.Point{sx0, sy0}).In(sr) {
                continue
            }
            pi := (sy0-src.Rect.Min.Y)*src.Stride + (sx0-src.Rect.Min.X)*4
            pr := uint32(src.Pix[pi+0]) * 0x101
            pg := uint32(src.Pix[pi+1]) * 0x101
            pb := uint32(src.Pix[pi+2]) * 0x101
            pa := uint32(src.Pix[pi+3]) * 0x101
            dst.Pix[d+0] = uint8(pr >> 8)
            dst.Pix[d+1] = uint8(pg >> 8)
            dst.Pix[d+2] = uint8(pb >> 8)
            dst.Pix[d+3] = uint8(pa >> 8)
        }
    }    
}
```
```cpp
void RotateDrawWithClip(
    WDIBPIXEL *pDstBase, int dstW, int dstH, int dstDelta,
    WDIBPIXEL *pSrcBase, int srcW, int srcH, int srcDelta,
    float fDstCX, float fDstCY, float fSrcCX, float fSrcCY, float fAngle, float fScale) {
    if (dstW <= 0) { return; }
    if (dstH <= 0) { return; }

    srcDelta /= sizeof(WDIBPIXEL);
    dstDelta /= sizeof(WDIBPIXEL);

    float duCol = (float)sin(-fAngle) * (1.0f / fScale);
    float dvCol = (float)cos(-fAngle) * (1.0f / fScale);
    float duRow = dvCol;
    float dvRow = -duCol;

    float startingu = fSrcCX - (fDstCX * dvCol + fDstCY * duCol);
    float startingv = fSrcCY - (fDstCX * dvRow + fDstCY * duRow);

    float rowu = startingu;
    float rowv = startingv;

    for(int y = 0; y < dstH; y++) {
        float uu = rowu;
        float vv = rowv;

        WDIBPIXEL *pDst = pDstBase + (dstDelta * y);

        for(int x = 0; x < dstW ; x++) {
            int sx = (int)uu;
            int sy = (int)vv;

            // For non-negative values we have to check u and v (not sx and sy)
            // since u = -0.25 gives sx=0 after rounsing, so 1 extra pixel line will be drawn
            // (we assume that u,v >= 0 will lead to sx,sy >= 0)

            if ((uu >= 0) && (vv >= 0) && (sx < srcW) && (sy < srcH)) {
                WDIBPIXEL *pSrc = pSrcBase + sx + (sy * srcDelta);
                *pDst++ = *pSrc++;
            } else {
                pDst++; // Skip
                //*pDst++ = VOID_COLOR; // Fill void (black)
            }

            uu += duRow;
            vv += dvRow;
        }

        rowu += duCol;
        rowv += dvCol;
    }
}
```
```py
def image_rotate(src, dstW, dstH, tf):
    # convW = cos(rotate) / scale
    dst = np.zeros([dstW, dstH, 3], dtype=src.dtype)
    ww, hh = src.shape[:2]
    ww, hh = ww - 1, hh - 1
    for ii in range(dstW):
        tw, th = tf[0] * ii, tf[3] * ii
        for jj in range(dstH):
            sw = int(tw + tf[1] * jj + tf[2])
            sh = int(th + tf[4] * jj + tf[5])
            if sw > 0 and sw < ww and sh > 0 and sh < hh:
                dst[ii, jj] = src[sw, sh]
            else:
                dst[ii, jj] = [0, 0, 0]
    return dst

angle = 60 / 180 * np.pi
tf = [cos(angle), -sin(angle), 0, sin(angle), cos(angle), 0]
```
## transform
- **transform.SimilarityTransform**
```py
__init__(self, matrix=None, scale=None, rotation=None, translation=None)
```

## NV21 to RGB
```py
import cv2
def YUVtoRGB(byteArray, width, height):
    e = width * height
    Y = byteArray[0:e]
    Y = np.reshape(Y, (height, width))

    s = e
    V = byteArray[s::2]
    V = np.repeat(V, 2, 0)
    V = np.reshape(V, (height // 2, width))
    V = np.repeat(V, 2, 0)

    U = byteArray[s+1::2]
    U = np.repeat(U, 2, 0)
    U = np.reshape(U, (height // 2, width))
    U = np.repeat(U, 2, 0)

    RGBMatrix = (np.dstack([Y,U,V])).astype(np.uint8)
    RGBMatrix = cv2.cvtColor(RGBMatrix, cv2.COLOR_YUV2RGB, 3)
    return RGBMatrix

with open('nv21.txt', 'r') as ff:
    aa = ff.read()
bb = [byte(ii) for ii in aa[1:-1].split(', ')]
with open('nv21.bin', 'wb') as ff:
    for ii in bb:
        ff.write(ii)

with open('nv21.bin', 'rb') as ff:
    cc = ff.read()
plt.imshow(YUVtoRGB([byte(ii) for ii in cc], 1280, 800))
```
