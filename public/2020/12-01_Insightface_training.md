# ___2019 - 11 - 18 Keras Insightface___
***

# 目录
  <!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

  - [___2019 - 11 - 18 Keras Insightface___](#2019-11-18-keras-insightface)
  - [目录](#目录)
  - [Test functions](#test-functions)
  	- [Decode mxnet log](#decode-mxnet-log)
  	- [Choose accuracy](#choose-accuracy)
  	- [Remove regular loss from total loss](#remove-regular-loss-from-total-loss)
  	- [Multi GPU losses test](#multi-gpu-losses-test)
  	- [Face recognition test](#face-recognition-test)
  	- [Replace ReLU with PReLU in mobilenet](#replace-relu-with-prelu-in-mobilenet)
  	- [Convolution test](#convolution-test)
  - [Model conversion](#model-conversion)
  	- [ONNX](#onnx)
  	- [TensorRT](#tensorrt)
  	- [TFlite](#tflite)
  - [Weight decay](#weight-decay)
  	- [MXNet SGD and tfa SGDW](#mxnet-sgd-and-tfa-sgdw)
  	- [L2 Regularization and Weight Decay](#l2-regularization-and-weight-decay)
  	- [SGD with momentum](#sgd-with-momentum)
  	- [Keras Model test](#keras-model-test)
  	- [MXNet model test](#mxnet-model-test)
  	- [Modify model with L2 regularizer](#modify-model-with-l2-regularizer)
  	- [Optimizers with weight decay test](#optimizers-with-weight-decay-test)
  - [Training Record](#training-record)
  	- [Loss function test on Mobilenet](#loss-function-test-on-mobilenet)
  	- [Mobilefacenet](#mobilefacenet)
  	- [Loss function test on Mobilefacenet epoch 44](#loss-function-test-on-mobilefacenet-epoch-44)
  	- [ResNet101V2](#resnet101v2)
  	- [EfficientNetB4](#efficientnetb4)
  	- [ResNeSt101](#resnest101)
  	- [Mobilenet_V2 emore BS1024 train test](#mobilenetv2-emore-bs1024-train-test)
  	- [Mobilnet emore BS256](#mobilnet-emore-bs256)
  	- [ResNet34 CASIA](#resnet34-casia)
  	- [Comparing early softmax training](#comparing-early-softmax-training)
  	- [MXNet record](#mxnet-record)
  - [Label smoothing](#label-smoothing)
  - [Nesterov](#nesterov)
  - [Sub center](#sub-center)
  - [Distillation](#distillation)
  	- [MNIST example](#mnist-example)
  	- [Embedding](#embedding)
  	- [Result](#result)
  - [IJB](#ijb)

  <!-- /TOC -->
***

# Test functions
## Decode mxnet log
  ```py
  import json

  def decode_mxnet_log(src_file, dest_file=None):
      with open(src_file, 'r') as ff:
          aa = ff.readlines()

      losses = [ii.strip() for ii in aa if 'Train-lossvalue=' in ii]
      losses = [float(ii.split('=')[-1]) for ii in losses]

      accs = [ii.strip() for ii in aa if 'Train-acc=' in ii]
      accs = [float(ii.split('=')[-1]) for ii in accs]

      lfws = [ii.strip() for ii in aa if 'Accuracy-Flip:' in ii and 'lfw' in ii]
      lfws = [float(ii.split(': ')[-1].split('+-')[0]) for ii in lfws]

      cfp_fps = [ii.strip() for ii in aa if 'Accuracy-Flip:' in ii and 'cfp_fp' in ii]
      cfp_fps = [float(ii.split(': ')[-1].split('+-')[0]) for ii in cfp_fps]

      agedb_30s = [ii.strip() for ii in aa if 'Accuracy-Flip:' in ii and 'agedb_30' in ii]
      agedb_30s = [float(ii.split(': ')[-1].split('+-')[0]) for ii in agedb_30s]

      lrs = [0.1] * 20 + [0.01] * 10 + [0.001] * (len(losses) - 20 - 10)

      bb = {
          "loss": losses,
          "accuracy": accs,
          "lr": lrs,
          "lfw": lfws,
          "cfp_fp": cfp_fps,
          "agedb_30": agedb_30s,
      }

      print({kk:len(bb[kk]) for kk in bb})

      if dest_file == None:
          dest_file = os.path.splitext(src_file)[0] + '.json'
      with open(dest_file, 'w') as ff:
          json.dump(bb, ff)
      return dest_file

  decode_mxnet_log('r34_wdm1_lazy_false.log')
  ```
## Choose accuracy
  ```py
  import json

  def choose_accuracy(aa):
      print(">>>> agedb max:")
      for pp in aa:
          with open(pp, 'r') as ff:
              hh = json.load(ff)
          nn = os.path.splitext(os.path.basename(pp))[0]
          arg_max = np.argmax(hh['agedb_30'])
          print(nn, ":", arg_max, ["%s: %.4f" % (kk, hh[kk][arg_max]) for kk in ['lfw', 'cfp_fp', 'agedb_30']])

      print(">>>> all max:")
      for pp in aa:
          with open(pp, 'r') as ff:
              hh = json.load(ff)
          nn = os.path.splitext(os.path.basename(pp))[0]
          print(nn, ":", ["%s: %.4f, %d" % (kk, max(hh[kk]), np.argmax(hh[kk])) for kk in ['lfw', 'cfp_fp', 'agedb_30']])

      print(">>>> sum max:")
      for pp in aa:
          with open(pp, 'r') as ff:
              hh = json.load(ff)
          nn = os.path.splitext(os.path.basename(pp))[0]
          arg_max = np.argmax(np.sum([hh[kk] for kk in ['lfw', 'cfp_fp', 'agedb_30']], axis=0))
          print(nn, ":", arg_max, ["%s: %.4f" % (kk, hh[kk][arg_max]) for kk in ['lfw', 'cfp_fp', 'agedb_30']])
  ```
## Remove regular loss from total loss
  ```py
  import json

  def remove_reg_loss_from_hist(src_hist, dest_hist=None):
      with open(src_hist, 'r') as ff:
          aa = json.load(ff)
      aa['loss'] = [ii - jj for ii, jj in zip(aa['loss'], aa['regular_loss'])]
      if dest_hist == None:
          dest_hist = os.path.splitext(src_hist)[0] + "_no_reg.json"
      with open(dest_hist, 'w') as ff:
          json.dump(aa, ff)
      return dest_hist

  remove_reg_loss_from_hist('checkpoints/NNNN_resnet34_MXNET_E_REG_BN_SGD_1e3_lr1e1_random0_arcT4_S32_E1_BS512_casia_3_hist.json')
  ```
## Multi GPU losses test
  ```py
  sys.path.append('..')
  import losses, train
  with tf.distribute.MirroredStrategy().scope():
      basic_model = train.buildin_models("MobileNet", dropout=0.4, emb_shape=256)
      tt = train.Train('faces_emore_test', save_path='temp_test.h5', eval_paths=['lfw.bin'], basic_model=basic_model, lr_base=0.001, batch_size=16, random_status=3)
      sch = [
          {"loss": losses.scale_softmax, "epoch": 2},
          {"loss": losses.ArcfaceLoss(), "triplet": 10, "epoch": 2},
          {"loss": losses.margin_softmax, "centerloss": 20, "epoch": 2},
          {"loss": losses.ArcfaceLoss(), "centerloss": 10, "triplet": 20, "epoch": 2},
          {"loss": losses.BatchAllTripletLoss(0.3), "alpha": 0.1, "epoch": 2},
          {"loss": losses.BatchHardTripletLoss(0.25), "centerloss": 10, "triplet": 20, "epoch": 2},
          {"loss": losses.CenterLoss(num_classes=5, emb_shape=256), "epoch": 2},
          {"loss": losses.CenterLoss(num_classes=5, emb_shape=256), "triplet": 10, "epoch": 2}
      ]
      tt.train(sch)
  ```
## Temp test
  ```py
  sys.path.append('..')
  import losses, train
  basic_model = train.buildin_models("MobileNet", dropout=0, emb_shape=256)
  tt = train.Train('faces_emore_test', save_path='temp_test.h5', eval_paths=['lfw.bin'], basic_model=basic_model, lr_base=0.001, batch_size=16, random_status=0)
  sch = [
      {"loss": losses.CenterLoss(num_classes=5, emb_shape=256), "epoch": 2},
  ]
  tt.my_evals[-1].save_model = None
  tt.basic_callbacks.pop(0) # NOT saving
  tt.basic_callbacks.pop(-1) # NO gently_stop
  tt.train(sch)
  ```
## Face recognition test
  ```py
  import glob2
  import insightface
  from sklearn.preprocessing import normalize
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

  imms = glob2.glob('./*.jpg')
  imgs = [imread(ii)[:, :, :3] for ii in imms]
  det = insightface.model_zoo.face_detection.retinaface_r50_v1()
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
## Replace ReLU with PReLU in mobilenet
  ```py
  def convert_ReLU(layer):
      # print(layer.name)
      if isinstance(layer, keras.layers.ReLU):
          print(">>>> Convert ReLU:", layer.name)
          return keras.layers.PReLU(shared_axes=[1, 2], name=layer.name)
      return layer

  mm = keras.applications.MobileNet(include_top=False, input_shape=(112, 112, 3), weights=None)
  mmn = keras.models.clone_model(mm, clone_function=convert_ReLU)
  ```
## Convolution test
  ```py
  inputs = tf.ones([1, 3, 3, 1])
  conv_valid = tf.keras.layers.Conv2D(1, 2, strides=2, padding='valid', use_bias=False, kernel_initializer='ones')
  conv_same = tf.keras.layers.Conv2D(1, 2, strides=2, padding='same', use_bias=False, kernel_initializer='ones')
  pad = keras.layers.ZeroPadding2D(padding=1)
  print(conv_valid(inputs).shape, conv_same(inputs).shape, conv_valid(pad(inputs)).shape)
  # (1, 1, 1, 1) (1, 2, 2, 1) (1, 2, 2, 1)

  print(inputs.numpy()[0, :, :, 0].tolist())
  # [[1.0, 1.0, 1.0],
  #  [1.0, 1.0, 1.0],
  #  [1.0, 1.0, 1.0]]
  print(conv_same(inputs).numpy()[0, :, :, 0].tolist())
  # [[4.0, 2.0],
  #  [2.0, 1.0]]
  print(conv_valid(pad(inputs)).numpy()[0, :, :, 0].tolist())
  # [[1.0, 2.0],
  #  [2.0, 4.0]]
  print(pad(inputs).numpy()[0, :, :, 0].tolist())
  # [[0.0, 0.0, 0.0, 0.0, 0.0],
  #  [0.0, 1.0, 1.0, 1.0, 0.0],
  #  [0.0, 1.0, 1.0, 1.0, 0.0],
  #  [0.0, 1.0, 1.0, 1.0, 0.0],
  #  [0.0, 0.0, 0.0, 0.0, 0.0]]
  ```
  ```py
  data = mx.symbol.Variable("data", shape=(1, 1, 3, 3))
  ww = mx.symbol.Variable("ww", shape=(1, 1, 2, 2))
  cc = mx.sym.Convolution(data=data, weight=ww, no_bias=True, kernel=(2, 2), stride=(2, 2), num_filter=1, pad=(1, 1))

  aa = mx.nd.ones([1, 1, 3, 3])
  bb = mx.nd.ones([1, 1, 2, 2])
  ee = cc.bind(mx.cpu(), {'data': aa, 'ww': bb})
  print(ee.forward()[0].asnumpy()[0, 0].tolist())
  # [[1.0, 2.0],
  #  [2.0, 4.0]]
  ```
***

# Model conversion
## ONNX
  - `tf2onnx` convert `saved model` to `tflite`, support `tf1.15.0`
    ```py
    tf.__version__
    # '1.15.0'

    # Convert to saved model first
    import glob2
    mm = tf.keras.models.load_model(glob2.glob('./keras_mobilefacenet_256_basic_*.h5')[0], compile=False)
    tf.keras.experimental.export_saved_model(mm, './saved_model')
    # tf.contrib.saved_model.save_keras_model(mm, 'saved_model') # TF 1.13

    ! pip install -U tf2onnx
    ! python -m tf2onnx.convert --saved-model ./saved_model --output model.onnx
    ```
  - [keras2onnx](https://github.com/onnx/keras-onnx)
    ```py
    ! pip install keras2onnx

    import keras2onnx
    import glob2
    mm = tf.keras.models.load_model(glob2.glob('./keras_mobilefacenet_256_basic_*.h5')[0], compile=False)
    onnx_model = keras2onnx.convert_keras(mm, mm.name)
    keras2onnx.save_model(onnx_model, 'mm.onnx')
    ```
## TensorRT
  - [Atom_notebook TensorRT](https://github.com/leondgarse/Atom_notebook/blob/master/public/2019/08-19_tensorrt.md)
## TFlite
  - Convert to TFlite
    ```py
    tf.__version__
    # '1.15.0'

    import glob2
    converter = tf.lite.TFLiteConverter.from_keras_model_file("checkpoints/keras_se_mobile_facenet_emore_triplet_basic_agedb_30_epoch_100_0.958333.h5")
    tflite_model = converter.convert()
    open('./model.tflite', 'wb').write(tflite_model)
    ```
    ```py
    tf.__version__
    # '2.1.0'

    import glob2
    mm = tf.keras.models.load_model(glob2.glob('./keras_mobilefacenet_256_basic_*.h5')[0], compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(mm)
    tflite_model = converter.convert()
    open('./model_tf2.tflite', 'wb').write(tflite_model)
    ```
  - interpreter test
    ```py
    tf.__version__
    # '2.1.0'

    import glob2
    interpreter = tf.lite.Interpreter('./model.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    def tf_imread(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = (img - 0.5) * 2
        return tf.expand_dims(img, 0)

    imm = tf_imread('/datasets/faces_emore_112x112_folders/0/1.jpg')
    # imm = tf_imread('./temp_test/faces_emore_test/0/1.jpg')
    interpreter.set_tensor(input_index, imm)
    interpreter.invoke()
    aa = interpreter.get_tensor(output_index)[0]

    def foo(imm):
        interpreter.set_tensor(input_index, imm)
        interpreter.invoke()
        return interpreter.get_tensor(output_index)[0]
    %timeit -n 100 foo(imm)
    # 36.7 ms ± 471 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    mm = tf.keras.models.load_model(glob2.glob('./keras_mobilefacenet_256_basic_*.h5')[0], compile=False)
    bb = mm(imm).numpy()
    assert np.allclose(aa, bb, rtol=1e-3)
    %timeit mm(imm).numpy()
    # 71.6 ms ± 213 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
    ```
  - **On ARM64 board**
    ```sh
    lscpu
    # Architecture:        aarch64

    python --version
    # Python 3.6.9

    sudo apt install python3-pip ipython cython3
    pip install ipython

    git clone https://github.com/noahzhy/tf-aarch64.git
    cd tf-aarch64/
    pip install tensorflow-1.9.0rc0-cp36-cp36m-linux_aarch64.whl
    pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp36-cp36m-linux_aarch64.whl
    ```
    ```py
    import tensorflow as tf
    tf.enable_eager_execution()
    tf.__version__
    # 1.9.0-rc0

    import tflite_runtime
    tflite_runtime.__version__
    # 2.1.0.post1

    import tflite_runtime.interpreter as tflite
    interpreter = tflite.Interpreter('./mobilefacenet_tf2.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    imm = tf.convert_to_tensor(np.ones([1, 112, 112, 3]), dtype=tf.float32)
    interpreter.set_tensor(input_index, imm)
    interpreter.invoke()
    out = interpreter.get_tensor(output_index)[0]

    def foo(imm):
        interpreter.set_tensor(input_index, imm)
        interpreter.invoke()
        return interpreter.get_tensor(output_index)[0]
    %timeit -n 100 foo(imm)
    # 42.4 ms ± 43.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

    %timeit -n 100 foo(imm) # EfficientNetB0
    # 71.2 ms ± 52.5 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    ```
  - **Wapper trained model with `Rescale` / `L2_normalize`**
    ```py
    mm2 = keras.Sequential([
        keras.layers.Input((112, 112, 3)),
        keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1),
        mm,
        keras.layers.Lambda(tf.nn.l2_normalize, name='norm_embedding', arguments={'axis': 1})
    ])
    ```
    ```py
    mm2 = keras.Sequential([
        keras.layers.Input((112, 112, 3), dtype='uint8'),
        keras.layers.Lambda(lambda xx: (xx / 127) - 1),
        # keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1),
        mm,
        # keras.layers.Lambda(tf.nn.l2_normalize, name='norm_embedding', arguments={'axis': 1}),
        keras.layers.Lambda(lambda xx: tf.cast(xx / tf.sqrt(tf.reduce_sum(xx ** 2)) * 255, 'uint8')),
        # keras.layers.Lambda(lambda xx: tf.cast(xx * 255, 'uint8')),
    ])
    ```
    ```py
    inputs = keras.layers.Input([112, 112, 3])
    nn = (inputs - 127.5) / 128
    nn = mm(nn)
    nn = tf.divide(nn, tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.pow(nn, 2), -1)), -1))
    bb = keras.models.Model(inputs, nn)
    ```
  - **Dynamic input shape**
    ```py
    mm3 = keras.Sequential([
        keras.layers.Input((None, None, 3)),
        keras.layers.experimental.preprocessing.Resizing(112 ,112),
        keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset=-1),
        mm,
        keras.layers.Lambda(tf.nn.l2_normalize, name='norm_embedding', arguments={'axis': 1})
    ])

    converter = tf.lite.TFLiteConverter.from_keras_model(mm3)
    tflite_model = converter.convert()
    open('./norm_model_tf2.tflite', 'wb').write(tflite_model)

    interpreter = tf.lite.Interpreter('./norm_model_tf2.tflite')
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    interpreter.resize_tensor_input(input_index, (1, 512, 512, 3))
    interpreter.allocate_tensors()
    interpreter.set_tensor(input_index, tf.ones([1, 512, 112, 3], dtype='float32'))
    interpreter.invoke()
    out = interpreter.get_tensor(output_index)[0]
    ```
  - **Integer-only quantization**
    ```py
    def tf_imread(file_path):
        img = tf.io.read_file(file_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = (img - 0.5) * 2
        return tf.expand_dims(img, 0)

    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(image_names).batch(1).take(100):
            yield [tf_imread(input_value[0])]

    aa = np.load('faces_emore_112x112_folders_shuffle.pkl', allow_pickle=True)
    image_names, image_classes = aa["image_names"], aa["image_classes"]

    mm = tf.keras.models.load_model("checkpoints/keras_se_mobile_facenet_emore_triplet_basic_agedb_30_epoch_100_0.958333.h5", compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(mm)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    tflite_model_quant = converter.convert()
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    input_type = interpreter.get_input_details()[0]['dtype']
    print('input: ', input_type)
    output_type = interpreter.get_output_details()[0]['dtype']
    print('output: ', output_type)

    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, tf.ones([1, 112, 112, 3], dtype=input_type))
    interpreter.invoke()
    interpreter.get_tensor(output_index)[0]
    ```
***

# Weight decay
## MXNet SGD and tfa SGDW
  - [AdamW and Super-convergence is now the fastest way to train neural nets](https://www.fast.ai/2018/07/02/adam-weight-decay/)
  - The behavior of `weight_decay` in `mx.optimizer.SGD` and `tfa.optimizers.SGDW` is different.
  - **MXNet SGD** multiplies `wd` with `lr`.
    ```py
    import mxnet as mx
    help(mx.optimizer.SGD)
    # weight = weight - lr * (rescale_grad * clip(grad, clip_gradient) + wd * weight)
    #        = (1 - lr * wd) * weight - lr * (rescale_grad * clip(grad, clip_gradient))
    ```
    Test with `learning_rate=0.1, weight_decay=5e-4`, weight is actually modified by `5e-5`.
    ```py
    import mxnet as mx
    mm_loss_grad = mx.nd.array([[1., 1], [1, 1]])

    mm = mx.nd.array([[1., 1], [1, 1]])
    mopt = mx.optimizer.SGD(learning_rate=0.1)
    mopt.update(0, mm, mm_loss_grad, None)
    print(mm.asnumpy())  # Basic value is `mm - lr * mm_loss = 0.9`
    # [[0.9 0.9] [0.9 0.9]]

    mm = mx.nd.array([[1., 1], [1, 1]])
    mopt = mx.optimizer.SGD(learning_rate=0.1, wd=5e-4)
    mopt.update(0, mm, mm_loss_grad, None)
    print(mm.asnumpy())  # 0.9 - 0.89995 = 5e-5
    # [[0.89995 0.89995]  [0.89995 0.89995]]
    ```
  - **tfa SGDW** behaves different, it does NOT multiply `wd` with `lr`. With `learning_rate=0.1, weight_decay=5e-4`, weight is actually modified with `5e-4`.
    ```py
    # /opt/anaconda3/lib/python3.7/site-packages/tensorflow_addons/optimizers/weight_decay_optimizers.py
    # 170     def _decay_weights_op(self, var, apply_state=None):
    # 177             return var.assign_sub(coefficients["wd_t"] * var, self._use_locking)
    ```
    ```py
    import tensorflow_addons as tfa
    ww_loss_grad = tf.convert_to_tensor([[1., 1.], [1., 1.]])
    ww = tf.Variable([[1., 1.], [1., 1.]])
    opt = tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=5e-4)
    opt.apply_gradients(zip([ww_loss_grad], [ww]))
    print(ww.numpy()) # 0.9 - 0.8995 = 5e-4
    # [[0.8995 0.8995] [0.8995 0.8995]]
    ```
    So `learning_rate=0.1, weight_decay=5e-4` in `mx.optimizer.SGD` is equal to `learning_rate=0.1, weight_decay=5e-5` in `tfa.optimizers.SGDW`.
  - **weight decay multiplier** If we set `wd_mult=10` in a MXNet layer, `wd` will mutiply by `10` in this layer. This means it will be `weight_decay == 5e-4` in a keras layer.
    ```py
    # https://github.com/apache/incubator-mxnet/blob/e6cea0d867329131fa6052e5f45dc5f626c00d72/python/mxnet/optimizer/optimizer.py#L482
    # 29  class Optimizer(object):
    # 482                lrs[i] *= self.param_dict[index].lr_mult
    ```
## L2 Regularization and Weight Decay
  - [Weight Decay == L2 Regularization?](https://towardsdatascience.com/weight-decay-l2-regularization-90a9e17713cd)
  - [PDF DECOUPLED WEIGHT DECAY REGULARIZATION](https://arxiv.org/pdf/1711.05101.pdf)
  - **Keras l2 regularization**
    ```py
    ww = tf.convert_to_tensor([[1.0, -2.0], [-3.0, 4.0]])

    # loss = l2 * reduce_sum(square(x))
    aa = keras.regularizers.L2(0.2)
    aa(ww)  # tf.reduce_sum(ww ** 2) * 0.2
    # 6.0

    # output = sum(t ** 2) / 2
    tf.nn.l2_loss(ww)
    # 15.0
    tf.nn.l2_loss(ww) * 0.2
    # 3.0
    ```
    Total loss with l2 regularization will be
    ```py
    total_loss = Loss(w) + λ * R(w)
    ```
  - `Keras.optimizers.SGD`
    ```py
    help(keras.optimizers.SGD)
    # w = w - learning_rate * g
    #   = w - learning_rate * g - learning_rate * Grad(l2_loss)
    ```
    So with `keras.regularizers.L2(λ)`, it should be
    ```py
    wd * weight = Grad(l2_loss)
        --> wd * weight = 2 * λ * weight
        --> λ = wd / 2
    ```
    **Test**
    ```py
    ww_loss_grad = tf.convert_to_tensor([[1., 1.], [1., 1.]])
    ww = tf.Variable([[1., 1.], [1., 1.]])
    opt = keras.optimizers.SGD(0.1)
    with tf.GradientTape() as tape:
        # l2_loss = tf.nn.l2_loss(ww) * 5e-4
        l2_loss = keras.regularizers.L2(5e-4 / 2)(ww)  # `tf.nn.l2_loss` divided the loss by 2, `keras.regularizers.L2` not
    l2_grad = tape.gradient(l2_loss, ww).numpy()
    opt.apply_gradients(zip([ww_loss_grad + l2_grad], [ww]))
    print(ww.numpy()) # 0.9 - 0.89995 = 5e-5
    # [[0.89995 0.89995] [0.89995 0.89995]]
    ```
    That means the `L2_regulalizer` will modify the weights value by `l2 * lr == 5e-4 * 0.1 = 5e-5`.
  - If we want the same result as `mx.optimizer.SGD(learning_rate=0.1, wd=5e-4)` and `wd_mult=10` in a MXNet layer, which actually decay this layer's weights with `wd * wd_mult * learning_rate == 5e-4`, and other layers `wd * learning_rate == 5e-5`.
    - Firstlly, the keras optimizer is `tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=5e-5)`.
    - Then add a `keras.regularizers.L2` with `l2 == weight_decay / learning_rate * (wd_mult - 1) / 2` to this layer.
    ```py
    ww_loss_grad = tf.convert_to_tensor([[1., 1.], [1., 1.]])
    ww = tf.Variable([[1., 1.], [1., 1.]])
    opt = tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=5e-5)
    with tf.GradientTape() as tape:
        l2_loss = keras.regularizers.L2(5e-5 / 0.1 * (10 - 1) / 2)(ww)
    l2_grad = tape.gradient(l2_loss, ww).numpy()
    opt.apply_gradients(zip([ww_loss_grad + l2_grad], [ww]))
    print(ww.numpy()) # 0.9 - 0.8995 = 5e-4
    # [[0.8995 0.8995] [0.8995 0.8995]]
    ```
## SGD with momentum
  - **MXNet**
    ```py
    # incubator-mxnet/python/mxnet/optimizer/sgd.py, incubator-mxnet/src/operator/optimizer_op.cc +109
    grad += wd * weight
    momentum_stat = momentum * momentum_stat - lr * grad
    weight += momentum_stat
    ```
  - **Keras SGDW** Using `wd == lr * wd`, `weight` will be the same with `MXNet SGD` in the first update, but `momentum_stat` will be different. Then in the second update, `weight` will also be different.
    ```py
    momentum_stat = momentum * momentum_stat - lr * grad
    weight += momentum_stat - wd * weight
    ```

  - **Keras SGD with l2 regularizer** can behave same as `MXNet SGD`
    ```py
    grad += regularizer_loss
    momentum_stat = momentum * momentum_stat - lr * grad
    weight += momentum_stat
    ```
## Keras Model test
  ```py
  import tensorflow_addons as tfa

  def test_optimizer_with_model(opt, epochs=3, l2=0):
      kernel_regularizer = None if l2 == 0 else keras.regularizers.L2(l2)
      aa = keras.layers.Dense(1, use_bias=False, kernel_initializer='ones', kernel_regularizer=kernel_regularizer)
      aa.build([1])
      mm = keras.Sequential([aa])
      loss = lambda y_true, y_pred: (y_true - y_pred) ** 2 / 2
      mm.compile(optimizer=opt, loss=loss)
      for ii in range(epochs):
          mm.fit([[1.]], [[0.]], epochs=ii+1, initial_epoch=ii, verbose=0)
          print("Epoch", ii, "- [weight]", aa.weights[0].numpy(), "- [losses]:", mm.history.history['loss'][0], end="")
          if len(opt.weights) > 1:
              print(" - [momentum]:", opt.weights[-1].numpy(), end="")
          print()
      return mm, opt

  test_optimizer_with_model(tf.keras.optimizers.SGD(learning_rate=0.1), epochs=3)
  # Epoch 0 - [weight] [[0.9]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.81]] - [losses]: 0.4049999713897705
  # Epoch 2 - [weight] [[0.729]] - [losses]: 0.32804998755455017
  test_optimizer_with_model(tf.keras.optimizers.SGD(learning_rate=0.1), l2=0.01, epochs=3)
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5099999904632568
  # Epoch 1 - [weight] [[0.806404]] - [losses]: 0.411266028881073
  # Epoch 2 - [weight] [[0.7241508]] - [losses]: 0.33164656162261963
  test_optimizer_with_model(tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=0.002), epochs=3)
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.806404]] - [losses]: 0.40320199728012085
  # Epoch 2 - [weight] [[0.72415084]] - [losses]: 0.3251436948776245
  test_optimizer_with_model(tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), epochs=3)
  # Epoch 0 - [weight] [[0.9]] - [losses]: 0.5 - [momentum]: [[-0.1]]
  # Epoch 1 - [weight] [[0.71999997]] - [losses]: 0.4049999713897705 - [momentum]: [[-0.17999999]]
  # Epoch 2 - [weight] [[0.486]] - [losses]: 0.25919997692108154 - [momentum]: [[-0.23399998]]
  test_optimizer_with_model(tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), l2=0.01, epochs=3)
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5099999904632568 - [momentum]: [[-0.102]] ==> 0.102 * 0.1
  # Epoch 1 - [weight] [[0.714604]] - [losses]: 0.411266028881073 - [momentum]: [[-0.183396]] ==> -0.102 * 0.9 - 0.898 * 1.02 * 0.1
  # Epoch 2 - [weight] [[0.47665802]] - [losses]: 0.2604360580444336 - [momentum]: [[-0.237946]]
  # ==> momentum_stat_2 == momentum_stat_1 * momentum - weight_1 * (1 + l2 * 2) * learning_rate
  test_optimizer_with_model(tfa.optimizers.SGDW(learning_rate=0.1, momentum=0.9, weight_decay=0.002), epochs=3)
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5 - [momentum]: [[-0.1]]
  # Epoch 1 - [weight] [[0.71640396]] - [losses]: 0.40320199728012085 - [momentum]: [[-0.1798]]
  # Epoch 2 - [weight] [[0.48151073]] - [losses]: 0.25661730766296387 - [momentum]: [[-0.2334604]]

  test_optimizer_with_model(tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9), l2=0.1, epochs=3)
  # Epoch 0 - [weight] [[0.88]] - [losses]: 0.6000000238418579 - [momentum]: [[-0.12]]
  # Epoch 1 - [weight] [[0.66639996]] - [losses]: 0.4646399915218353 - [momentum]: [[-0.21360001]]
  # Epoch 2 - [weight] [[0.39419195]] - [losses]: 0.266453355550766 - [momentum]: [[-0.272208]]
  ```
## MXNet model test
  - **wd_mult** NOT working if just added in `mx.symbol.Variable`, has to be added by `opt.set_wd_mult`.
  ```py
  import mxnet as mx
  import logging
  logging.getLogger().setLevel(logging.ERROR)

  def test_optimizer_with_mxnet_model(opt, epochs=3, wd_mult=None):
      xx, yy = np.array([[1.]]), np.array([[0.]])
      xx_input, yy_input = mx.nd.array(xx), mx.nd.array(yy)
      dataiter = mx.io.NDArrayIter(xx, yy)

      data = mx.symbol.Variable("data", shape=(1,))
      label = mx.symbol.Variable("softmax_label", shape=(1,))
      # ww = mx.symbol.Variable("ww", shape=(1, 1), wd_mult=wd_mult, init=mx.init.One())
      ww = mx.symbol.Variable("ww", shape=(1, 1), init=mx.init.One())
      nn = mx.sym.FullyConnected(data=data, weight=ww, no_bias=True, num_hidden=1)

      # loss = mx.symbol.SoftmaxOutput(data=nn, label=label, name='softmax')
      loss = mx.symbol.MakeLoss((label - nn) ** 2 / 2)
      # sss = loss.bind(mx.cpu(), {'data': xx_input, 'softmax_label': yy_input, 'ww': y_pred})
      # print(sss.forward()[0].asnumpy().tolist())
      # [[0.5]]
      if wd_mult is not None:
          opt.set_wd_mult({'ww': wd_mult})
      model = mx.mod.Module(context=mx.cpu(), symbol=loss)
      weight_value = mx.nd.ones([1, 1])
      for ii in range(epochs):
          loss_value = loss.bind(mx.cpu(), {'data': xx_input, 'softmax_label': yy_input, 'ww': weight_value}).forward()[0]
          # model.fit(train_data=dataiter, num_epoch=ii+1, begin_epoch=0, optimizer=opt, force_init=True)
          model.fit(train_data=dataiter, num_epoch=ii+1, begin_epoch=ii, optimizer=opt)
          weight_value = model.get_params()[0]['ww']
          # output = model.get_outputs()[0]
          print("Epoch", ii, "- [weight]", weight_value.asnumpy(), "- [losses]:", loss_value.asnumpy()[0, 0])
          # if len(opt.weights) > 1:
          #     print(" - [momentum]:", opt.weights[-1].numpy(), end="")
          # print()

  test_optimizer_with_mxnet_model(mx.optimizer.SGD(learning_rate=0.1, wd=0.02))
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.806404]] - [losses]: 0.403202
  # Epoch 2 - [weight] [[0.7241508]] - [losses]: 0.3251437
  test_optimizer_with_mxnet_model(mx.optimizer.SGD(learning_rate=0.1, wd=0.002))
  # Epoch 0 - [weight] [[0.8998]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.80964005]] - [losses]: 0.40482002
  # Epoch 2 - [weight] [[0.72851413]] - [losses]: 0.3277585
  test_optimizer_with_mxnet_model(mx.optimizer.SGD(learning_rate=0.1, momentum=0.9, wd=0.02))
  # Epoch 0 - [weight] [[0.898]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.714604]] - [losses]: 0.403202
  # Epoch 2 - [weight] [[0.47665802]] - [losses]: 0.25532946
  test_optimizer_with_mxnet_model(mx.optimizer.SGD(learning_rate=0.1, momentum=0.9, wd=0.02), wd_mult=10)
  # Epoch 0 - [weight] [[0.88]] - [losses]: 0.5
  # Epoch 1 - [weight] [[0.66639996]] - [losses]: 0.3872
  # Epoch 2 - [weight] [[0.39419195]] - [losses]: 0.22204445
  # ==> Equals to keras model `l2 == 0.1`
  ```
## Modify model with L2 regularizer
  ```py
  mm = keras.applications.MobileNet()

  regularizers_type = {}
  for layer in mm.layers:
      rrs = [kk for kk in layer.__dict__.keys() if 'regularizer' in kk and not kk.startswith('_')]
      if len(rrs) != 0:
          # print(layer.name, layer.__class__.__name__, rrs)
          if layer.__class__.__name__ not in regularizers_type:
              regularizers_type[layer.__class__.__name__] = rrs
  print(regularizers_type)
  # {'Conv2D': ['kernel_regularizer', 'bias_regularizer'],
  # 'BatchNormalization': ['beta_regularizer', 'gamma_regularizer'],
  # 'PReLU': ['alpha_regularizer'],
  # 'SeparableConv2D': ['kernel_regularizer', 'bias_regularizer', 'depthwise_regularizer', 'pointwise_regularizer'],
  # 'DepthwiseConv2D': ['kernel_regularizer', 'bias_regularizer', 'depthwise_regularizer'],
  # 'Dense': ['kernel_regularizer', 'bias_regularizer']}

  weight_decay = 5e-4
  for layer in mm.layers:
      if isinstance(layer, keras.layers.Dense) or isinstance(layer, keras.layers.Conv2D) or isinstance(layer, keras.layers.DepthwiseConv2D):
          print(">>>> Dense or Conv2D", layer.name, "use_bias:", layer.use_bias)
          layer.kernel_regularizer = keras.regularizers.L2(weight_decay / 2)
          if layer.use_bias:
              layer.bias_regularizer = keras.regularizers.L2(weight_decay / 2)
      if isinstance(layer, keras.layers.SeparableConv2D):
          print(">>>> SeparableConv2D", layer.name, "use_bias:", layer.use_bias)
          layer.pointwise_regularizer = keras.regularizers.L2(weight_decay / 2)
          layer.depthwise_regularizer = keras.regularizers.L2(weight_decay / 2)
          if layer.use_bias:
              layer.bias_regularizer = keras.regularizers.L2(weight_decay / 2)
      if isinstance(layer, keras.layers.BatchNormalization):
          print(">>>> BatchNormalization", layer.name, "scale:", layer.scale, ", center:", layer.center)
          if layer.center:
              layer.beta_regularizer = keras.regularizers.L2(weight_decay / 2)
          if layer.scale:
              layer.gamma_regularizer = keras.regularizers.L2(weight_decay / 2)
      if isinstance(layer, keras.layers.PReLU):
          print(">>>> PReLU", layer.name)
          layer.alpha_regularizer = keras.regularizers.L2(weight_decay / 2)
  ```
## Optimizers with weight decay test
  ```py
  from tensorflow import keras
  import tensorflow_addons as tfa
  import losses, data, evals, myCallbacks, train
  # from tensorflow.keras.callbacks import LearningRateScheduler

  # Dataset
  data_path = '/datasets/faces_emore_112x112_folders'
  train_ds = data.prepare_dataset(data_path, batch_size=256, random_status=3, random_crop=(100, 100, 3))
  classes = train_ds.element_spec[-1].shape[-1]

  # Model
  basic_model = train.buildin_models("MobileNet", dropout=0, emb_shape=256)
  # model_output = keras.layers.Dense(classes, activation="softmax")(basic_model.outputs[0])
  model_output = train.NormDense(classes, name="arcface")(basic_model.outputs[0])
  model = keras.models.Model(basic_model.inputs[0], model_output)

  # Evals and basic callbacks
  save_name = 'keras_mxnet_test_sgdw'
  eval_paths = ['/datasets/faces_emore/lfw.bin', '/datasets/faces_emore/cfp_fp.bin', '/datasets/faces_emore/agedb_30.bin']
  my_evals = [evals.eval_callback(basic_model, ii, batch_size=256, eval_freq=1) for ii in eval_paths]
  my_evals[-1].save_model = save_name
  basic_callbacks = myCallbacks.basic_callbacks(checkpoint=save_name + '.h5', evals=my_evals, lr=0.001)
  basic_callbacks = basic_callbacks[:1] + basic_callbacks[2:]
  callbacks = my_evals + basic_callbacks
  # Compile and fit

  ss = myCallbacks.ConstantDecayScheduler([3, 5, 7], lr_base=0.1)
  optimizer = tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=5e-4, momentum=0.9)

  model.compile(optimizer=optimizer, loss=losses.arcface_loss, metrics=["accuracy"])
  # model.compile(optimizer=optimizer, loss=keras.losses.categorical_crossentropy, metrics=["accuracy"])
  wd_callback = myCallbacks.OptimizerWeightDecay(optimizer.lr.numpy(), optimizer.weight_decay.numpy())
  model.fit(train_ds, epochs=15, callbacks=[ss, wd_callback, *callbacks], verbose=1)

  opt = tfa.optimizers.AdamW(weight_decay=lambda : None)
  opt.weight_decay = lambda : 5e-1 * opt.lr

  mlp.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy())
  ```
  ```py
  class Foo:
      def __init__(self, wd):
          self.wd = wd
      def __call__(self):
          return self.wd
      def set_wd(self, wd):
          self.wd = wd

  class L2_decay_wdm(keras.regularizers.L2):
      def __init__(self, wd_func=None, **kwargs):
          super(L2_decay_wdm, self).__init__(**kwargs)
          self.wd_func = wd_func

      def __call__(self, x):
          self.l2 = self.wd_func()
          # tf.print(", l2 =", self.l2, end='')
          return super(L2_decay_wdm, self).__call__(x)

      def get_config(self):
          self.l2 = 0  # Just a fake value for saving
          config = super(L2_decay_wdm, self).get_config()
          return config
  ```
***

# Training Record
## Loss function test on Mobilenet
  - This tests loss functions on `Mobilenet` for their efficiency, but only one epoch training may not be very valuable.
  - **Initialize training from scratch for 6 epochs**
    ```py
    from tensorflow import keras
    import losses
    import train
    basic_model = train.buildin_models("MobileNet", dropout=0.4, emb_shape=256)
    data_path = '/datasets/faces_emore_112x112_folders'
    eval_paths = ['/datasets/faces_emore/lfw.bin', '/datasets/faces_emore/cfp_fp.bin', '/datasets/faces_emore/agedb_30.bin']
    tt = train.Train(data_path, 'keras_mobilenet_256.h5', eval_paths, basic_model=basic_model, model=None, compile=False, lr_base=0.001, batch_size=128, random_status=3)
    sch = [{"loss": losses.ArcfaceLoss(), "optimizer": None, "epoch": 6}]
    tt.train(sch, 0)
    ```
  - **Train next epoch 7 using different loss functions**
    ```py
    ''' Load saved basic model '''
    import losses
    import train
    data_path = '/datasets/faces_emore_112x112_folders'
    eval_paths = ['/datasets/faces_emore/lfw.bin', '/datasets/faces_emore/cfp_fp.bin', '/datasets/faces_emore/agedb_30.bin']
    tt = train.Train(data_path, 'keras_mobilenet_256_V.h5', eval_paths, basic_model="./checkpoints/keras_mobilenet_256_basic_agedb_30_epoch_6_0.900333.h5", model=None, compile=False, lr_base=0.001, batch_size=128, random_status=3)

    ''' Choose one loss function each time --> train one epoch --> reload'''
    sch = [{"loss": keras.losses.categorical_crossentropy, "optimizer": "adam", "epoch": 1}]
    sch = [{"loss": losses.margin_softmax, "optimizer": "adam", "epoch": 1}]
    sch = [{"loss": losses.scale_softmax, "optimizer": "adam", "epoch": 1}]
    sch = [{"loss": losses.arcface_loss, "optimizer": "adam", "epoch": 1}]
    sch = [{"loss": losses.arcface_loss, "optimizer": "adam", "centerloss": True, "epoch": 1}]
    sch = [{"loss": losses.batch_hard_triplet_loss, "optimizer": "adam", "epoch": 1}]
    sch = [{"loss": losses.batch_all_triplet_loss, "optimizer": "adam", "epoch": 1}]

    !pip install -q --no-deps tensorflow-addons
    import tensorflow_addons as tfa
    sch = [{"loss": tfa.losses.TripletSemiHardLoss(), "optimizer": "adam", "epoch": 1, "type": tt.triplet}]

    ''' Train '''
    tt.train(sch, 6)
    ```
  - **Loss and accuracy**

    | Loss type               | loss    | accuracy | lfw      | lfw thresh | cfp_fp   | cfp_fp thresh | agedb_30 | agedb_30 thresh | total time | per step |
    | ----------------------- | ------- | -------- | -------- | ---------- | -------- | ------------- | -------- | --------------- | ---------- | -------- |
    | **Original Epoch 6**    | 22.6342 | 0.7855   | 0.987833 | 0.307455   | 0.891714 | 0.201755      | 0.900333 | 0.229057        | 5653s      | 124ms    |
    | **Train Epoch 7**       |         |          |          |            |          |               |          |                 |            |          |
    | softmax                 | 1.8196  | 0.6941   | 0.987333 | 0.345970   | 0.895286 | 0.204387      | 0.901667 | 0.265905        | 5677s      | 125ms    |
    | margin_softmax          | 3.8359  | 0.6294   | 0.989000 | 0.317540   | 0.889000 | 0.210142      | 0.897833 | 0.246658        | 5716s      | 126ms    |
    | scale_softmax           | 2.2430  | 0.6779   | 0.987333 | 0.340417   | 0.887857 | 0.204122      | 0.900333 | 0.273266        | 5702s      | 125ms    |
    | arcface_loss            | 22.3337 | 0.7928   | 0.987500 | 0.293580   | 0.886857 | 0.199602      | 0.904833 | 0.247436        | 6133s      | 135ms    |
    | center arcface_loss     | 22.5102 | 0.7924   | 0.987833 | 0.321488   | 0.884000 | 0.200262      | 0.894833 | 0.263254        | 5861s      | 129ms    |
    | batch_hard_triplet_loss | 0.2276  |          | 0.986333 | 0.386425   | 0.910571 | 0.245836      | 0.891333 | 0.354833        | 4622s      | 156ms    |
    | batch_all_triplet_loss  | 0.4749  |          | 0.984333 | 0.417722   | 0.902571 | 0.240187      | 0.837167 | 0.475637        | 4708s      | 159ms    |
    | TripletSemiHardLoss     | 0.0047  |          | 0.957500 | 0.520159   | 0.837857 | 0.441421      | 0.778833 | 0.626684        | 4400s      | 148ms    |
## Mobilefacenet
  - Training script is the last exampled one.
  - **Mobilefacenet Record** Two models are trained, with `batch_size=160` and `batch_size=768` respectively.
    | Loss               | Epochs | First epoch (batch_size=768)                        |
    | ------------------ | ------ | --------------------------------------------------- |
    | Softmax            | 15     | 12744s 2s/step - loss: 4.8241 - accuracy: 0.3282    |
    | Margin Softmax     | 10     | 13041s 2s/step - loss: 0.4096 - accuracy: 0.9323    |
    | Bottleneck Arcface | 4      | 4292s 566ms/step - loss: 21.6166 - accuracy: 0.8569 |
    | Arcface 64         | 35     | 12793s 2s/step - loss: 15.4268 - accuracy: 0.9441   |

  - **se_mobilefacenet Record** Two models are trained, with `label_smoothing=0` and `label_smoothing=0.1` respectively, `batch_size = 640`
    | Loss               | Epochs | First epoch (label_smoothing=0.1)                   | First epoch (label_smoothing=0)                     |
    | ------------------ | ------ | --------------------------------------------------- | --------------------------------------------------- |
    | Softmax            | 15     | 13256s 2s/step - loss: 5.9982 - accuracy: 0.3615    |                                                     |
    | Bottleneck Arcface | 4      | 4111s 452ms/step - loss: 21.7145 - accuracy: 0.8624 | 4104s 451ms/step - loss: 20.7879 - accuracy: 0.8643 |
    | Arcface 64         | 30     | 13043s 1s/step - loss: 16.7003 - accuracy: 0.9491   | 13092s 1s/step - loss: 15.0788 - accuracy: 0.9498   |
    | Triplet (BS 1440)  | 50     |                                                     | 6688s 2s/step - loss: 0.2319                        |

  - **Plot**
    ```py
    import plot
    # plot.hist_plot_split("./checkpoints/keras_mobile_facenet_emore_hist.json", [15, 10, 4, 35], ["Softmax", "Margin Softmax", "Bottleneck Arcface", "Arcface scale=64"])
    customs = ["agedb_30", "cfp_fp"]
    epochs = [15, 10, 4, 35]
    _, axes = plt.subplots(1, 3, figsize=(24, 8))
    axes, _ = plot.hist_plot_split("checkpoints/keras_mobile_facenet_emore_hist.json", epochs, customs=customs, axes=axes, fig_label="Mobilefacenet, BS=768")
    axes, _ = plot.hist_plot_split("checkpoints/keras_mobilefacenet_256_hist_all.json", epochs, customs=customs, axes=axes, fig_label="Mobilefacenet, BS=160")

    axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_VI_hist.json', epochs, customs=customs, axes=axes, fig_label="se, Cosine, BS = 640, LS=0.1")
    axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_VII_nadam_hist.json', epochs, customs=customs, axes=axes, fig_label="se, Cosine, BS = 640, nadam, LS=0.1", init_epoch=3)

    axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_VIII_hist.json', epochs, customs=customs, axes=axes, fig_label="new se_mobilefacenet, Cosine, center, BS = 640, nadam, LS=0.1")
    axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_VIII_PR_hist.json', epochs, customs=customs, axes=axes, fig_label="new se_mobilefacenet, PR, Cosine, center, BS = 640, nadam, LS=0.1")
    axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_X_hist.json', epochs, customs=customs, axes=axes, fig_label="new se_mobilefacenet, Cosine, center, leaky, BS = 640, nadam, LS=0.1")

    axes, pre_1 = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_hist.json', epochs, names=["Softmax", "Margin Softmax"], customs=customs, axes=axes, fig_label="se, BS = 640, LS=0.1")
    axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_II_hist.json', [4, 35], customs=customs, init_epoch=25, pre_item=pre_1, axes=axes, fig_label="se, BS = 640, LS=0.1")
    axes, pre_2 = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_III_hist_E45.json', [4, 35], names=["Bottleneck Arcface", "Arcface scale=64"], customs=customs, init_epoch=25, pre_item=pre_1, axes=axes, fig_label="se, BS = 640, LS=0")
    axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_triplet_III_hist.json', [10, 10, 10, 20], names=["Triplet alpha=0.35", "Triplet alpha=0.3", "Triplet alpha=0.25", "Triplet alpha=0.2"], customs=customs, init_epoch=59, pre_item=pre_2, axes=axes, save="", fig_label="se, BS = 640, triplet")
    ```
    ![](checkpoints/keras_se_mobile_facenet_emore_triplet_III_hist.svg)
## Loss function test on Mobilefacenet epoch 44
  - For `Epoch 44`, trained steps are `15 epochs softmax + 10 epochs margin softmax + 4 epochs arcface bottleneck only + 15 epochs arcface`
  - Run a batch of `optimizer` + `loss` test. Each test run is `10 epochs`.
    ```py
    # This `train.Train` is the `batch_size = 160` one.
    sch = [{"loss": losses.ArcfaceLoss(), "epoch": 10}]  # Same as previous epochs
    sch = [{"loss": losses.Arcface(scale=32.0), "epoch": 10}] # fix lr == 1e-5
    sch = [{"loss": losses.Arcface(scale=32.0), "epoch": 10}] # lr decay, decay_rate = 0.1
    sch = [{"loss": losses.ArcfaceLoss(), "optimizer": keras.optimizers.SGD(0.001, momentum=0.9), "epoch": 10}]

    tt.train(sch, 40) # sub bottleneck only epochs
    ```
    From `Epoch 54`, Pick the best one `Scale=64.0, lr decay, optimizer=nadam`, run optimizer `nadam` / `adam` testing
    ```py
    sch = [{"loss": losses.ArcfaceLoss(), "epoch": 10}]
    sch = [{"loss": losses.ArcfaceLoss(), "optimizer": "adam", "epoch": 10}]
    tt.train(sch, 50) # sub bottleneck only epochs
    ```
  - **Result**
    ```py
    import plot
    axes, _ = plot.hist_plot_split('./checkpoints/keras_mobilefacenet_256_II_hist.json', [10], customs=["lr"], init_epoch=40, axes=None, fig_label="S=32, lr=5e-5, nadam")
    axes, _ = plot.hist_plot_split('./checkpoints/keras_mobilefacenet_256_III_hist.json', [10], customs=["lr"], init_epoch=40, axes=axes, save="", fig_label="S=32, lr decay, nadam")
    ```
    ![](checkpoints/keras_mobilefacenet_256_III_hist.svg)
    ```py
    import plot
    axes, _ = plot.hist_plot_split('./checkpoints/keras_mobilefacenet_256_IV_hist.json', [10], customs=["lr"], init_epoch=40, axes=None, fig_label="S=64, lr decay, SGD")
    axes, pre_1 = plot.hist_plot_split('./checkpoints/keras_mobilefacenet_256_VI_hist.json', [10], customs=["lr"], init_epoch=40, axes=axes, fig_label="S=64, lr decay, nadam")
    axes, _ = plot.hist_plot_split('./checkpoints/keras_mobilefacenet_256_VII_hist.json', [10], customs=["lr"], init_epoch=50, pre_item=pre_1, axes=axes, fig_label="S=64, lr decay, nadam")
    axes, _ = plot.hist_plot_split('./checkpoints/keras_mobilefacenet_256_VIII_hist.json', [10], customs=["lr"], init_epoch=50, pre_item=pre_1, axes=axes, save="", fig_label="S=64, lr decay, adam")
    ```
    ![](checkpoints/keras_mobilefacenet_256_VIII_hist.svg)
## ResNet101V2
  - **Training script** is similar with `Mobilefacenet`, just replace `basic_model` with `ResNet101V2`, and set a new `save_path`
    ```py
    basic_model = train.buildin_models("ResNet101V2", dropout=0.4, emb_shape=512)
    tt = train.Train(data_path, 'keras_resnet101_512.h5', eval_paths, basic_model=basic_model, batch_size=1024)
    ```
  - **Record** Two models are trained, with `batch_size=1024` and `batch_size=896, label_smoothing=0.1` respectively.
    | Loss               | epochs | First epoch (batch_size=896)                        | First epoch (2 GPUs, batch_size=1792)           |
    | ------------------ | ------ | --------------------------------------------------- | ----------------------------------------------- |
    | Softmax            | 25     | 11272s 2s/step - loss: 4.6730 - accuracy: 0.5484    |                                                 |
    | Bottleneck Arcface | 4      | 4053s 624ms/step - loss: 16.5645 - accuracy: 0.9414 |                                                 |
    | Arcface 64         | 35     | 11181s 2s/step - loss: 10.8983 - accuracy: 0.9870   | 6419s 2s/step - loss: 5.8991 - accuracy: 0.9896 |
    | Triplet            | 30     |                                                     | 5758s 2s/step - loss: 0.1562                    |

  - **Plot**
    ```py
    """ Evaluating accuracy is not improving from my end point """
    import plot
    # epochs = [15, 10, 4, 65, 15, 5, 5, 15]
    # history = ['./checkpoints/keras_resnet101_emore_hist.json', './checkpoints/keras_resnet101_emore_basic_hist.json']
    # plot.hist_plot_split("./checkpoints/keras_resnet101_emore_hist.json", [15, 10, 4, 35], ["Softmax", "Margin Softmax", "Bottleneck Arcface", "Arcface scale=64"])
    # axes, _ = plot.hist_plot_split(history, epochs, names=["Softmax", "Margin Softmax", "Bottleneck Arcface", "Arcface scale=64", "Triplet alpha=0.35", "Triplet alpha=0.3", "Triplet alpha=0.25", "Triplet alpha=0.2"], customs=customs, axes=axes, save="", fig_label='Resnet101, BS=896, label_smoothing=0.1')
    # axes, _ = plot.hist_plot_split(history, epochs, customs=customs, fig_label="ResNet101V2, BS=1024")
    customs = ["lfw", "agedb_30", "cfp_fp"]
    history = ['./checkpoints/keras_resnet101_emore_II_hist.json', './checkpoints/keras_resnet101_emore_II_triplet_hist.json']
    epochs = [25, 4, 35, 10, 10, 10, 10, 10]
    axes, _ = plot.hist_plot_split(history, epochs, names=["Softmax", "Bottleneck Arcface", "Arcface scale=64", "Triplet alpha=0.35", "Triplet alpha=0.3", "Triplet alpha=0.25", "Triplet alpha=0.2", "Triplet alpha=0.15"], customs=customs, save="", fig_label='Resnet101, BS=896, label_smoothing=0.1')
    ```
    ![](checkpoints/keras_resnet101_emore_II_triplet_hist.svg)
## EfficientNetB4
  - **Training script**
    ```py
    with tf.distribute.MirroredStrategy().scope():
        basic_model = train.buildin_models('EfficientNetB4', 0.4, 512)
        tt = train.Train(data_path, 'keras_EB4_emore.h5', eval_paths, basic_model=basic_model, batch_size=420, random_status=3)
    ```
  - **Record**
    | Loss               | epochs | First epoch (batch_size=420)                        | First epoch (2 GPUs, batch_size=840)                |
    | ------------------ | ------ | --------------------------------------------------- |--------------------------------------------------- |
    | Softmax            | 25     | 17404s 1s/step - loss: 4.4620 - accuracy: 0.5669    |                                                    |
    | Bottleneck Arcface | 4      | 4364s 629ms/step - loss: 18.1350 - accuracy: 0.9166 |                                                    |
    | Arcface 64         | 35     | 11047s 2s/step - loss: 11.3806 - accuracy: 0.9781   |                                                    |
    | Triplet            | 30     |                                                     |                                                    |

  - **Plot**
    ```py
    """ Comparing EfficientNetB4 and ResNet101 """
    import plot
    customs = ["lfw", "agedb_30", "cfp_fp"]
    epochs = [15, 10, 4, 30]
    axes, _ = plot.hist_plot_split("checkpoints/keras_resnet101_emore_II_hist.json", epochs, customs=customs, axes=None, fig_label='Resnet101, BS=1024, label_smoothing=0.1')
    axes, _ = plot.hist_plot_split("checkpoints/keras_EB4_emore_hist.json", epochs, names=["Softmax", "Margin Softmax", "Bottleneck Arcface", "Arcface scale=64", "Triplet"], customs=customs, axes=axes, save="", fig_label='EB4, BS=840, label_smoothing=0.1')
    ```
    ![](checkpoints/keras_EB4_emore_hist.svg)
## ResNeSt101
  - **Training script** is similar with `Mobilefacenet`, just replace `basic_model` with `ResNest101`, and set a new `save_path`
    ```py
    basic_model = train.buildin_models("ResNeSt101", dropout=0.4, emb_shape=512)
    tt = train.Train(data_path, 'keras_ResNest101_emore.h5', eval_paths, basic_model=basic_model, batch_size=600)
    ```
  - **Record** Two models are trained, with `batch_size=128` and `batch_size=1024` respectively.
    | Loss               | epochs | First epoch (batch_size=600)                     | First epoch (2 GPUs, batch_size=1024)               |
    | ------------------ | ------ | ------------------------------------------------ | --------------------------------------------------- |
    | Softmax            | 25     | 16820s 2s/step - loss: 5.2594 - accuracy: 0.4863 |                                                     |
    | Bottleneck Arcface | 4      |                                                  | 2835s 499ms/step - loss: 14.9653 - accuracy: 0.9517 |
    | Arcface 64         | 65     |                                                  | 9165s 2s/step - loss: 9.4768 - accuracy: 0.9905     |
    | Triplet            | 30     |                                                  | 8217s 2s/step - loss: 0.1169                        |

  - **Plot**
    ```py
    import plot
    customs = ["lfw", "agedb_30", "cfp_fp"]
    epochs = [25, 4, 35, 10, 10, 10, 10, 10]
    history = ['./checkpoints/keras_resnet101_emore_II_hist.json', './checkpoints/keras_resnet101_emore_II_triplet_hist.json']
    axes, _ = plot.hist_plot_split(history, epochs, customs=customs, fig_label='Resnet101, BS=896, label_smoothing=0.1')
    hists = ['./checkpoints/keras_ResNest101_emore_arcface_60_hist.json', './checkpoints/keras_ResNest101_emore_triplet_hist.json']
    axes, _ = plot.hist_plot_split(hists, epochs, names=["Softmax", "Bottleneck Arcface", "Arcface scale=64", "Triplet alpha=0.35", "Triplet alpha=0.3", "Triplet alpha=0.25", "Triplet alpha=0.2", "Triplet alpha=0.15"], customs=customs, axes=axes, save="", fig_label='ResNeSt101, BS=600')
    ```
    ![](checkpoints/keras_ResNest101_emore_triplet_hist.svg)
## Mobilenet_V2 emore BS1024 train test
  ```py
  import plot
  epochs = [25, 4, 35]
  customs = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "arcface_loss"]
  axes = None
  axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_n_center_cos_emore_hist.json", epochs, axes=axes, names=["Softmax", "Bottleneck Arcface", "Arcface scale=64"], customs=customs, fig_label='exp, [soft + center, adam, E25] [arc + center, E35]', eval_split=True)
  # axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_emore_hist.json", epochs, axes=axes, customs=customs, fig_label='exp, [soft, nadam, E25] [arc, nadam, E35]')
  # axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_n_emore_hist.json", epochs, axes=axes, customs=customs, fig_label='exp, [soft, adam, E25] [arc, E35]')
  # axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_cos_emore_hist.json", epochs, axes=axes, customs=customs, fig_label='cos, restarts=5, [soft, nadam, E25] [arc, nadam, E35]')
  # axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_cos_4_emore_hist.json", epochs, axes=axes, customs=customs, fig_label='cos, restarts=4, [soft, adam, E25] [arc, E35]')

  epochs = [60, 4, 40, 20]
  axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_n_center_emore_hist.json", epochs, names=["", "Bottleneck Arcface", "Arcface scale=64", "Triplet"], axes=axes, customs=customs, fig_label='exp, [soft + center, adam, E60] [arc + center, E35]')
  # axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_n_center_ls_emore_hist.json", epochs, axes=axes, customs=customs, fig_label='exp, [soft + center, adam, E60] [arc ls=0.1 + center 64, E35]')

  # axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_n_center_triplet_emore_hist.json", epochs, axes=axes, customs=customs, fig_label='exp, [soft + center, adam, E60] [soft + triplet, E12]')
  # axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_n_center_triplet_ls_emore_hist.json", epochs, axes=axes, customs=customs, fig_label='exp, [soft + center, adam, E60] [soft ls=0.1 + triplet, E12]')
  # axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_n_center_triplet_center_emore_hist.json", epochs, axes=axes, customs=customs, fig_label='exp, [soft + center, adam, E60] [soft + triplet + center, E30]')
  # axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_n_center_triplet_center_ls_emore_hist.json", epochs, axes=axes, customs=customs, fig_label='exp, [soft + center, adam, E60] [soft ls=0.1 + triplet + center, E30]')

  # axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_adamw_2_emore_hist.json", epochs, names=["", "", "", "Triplet"], axes=axes, customs=customs+['center_embedding_loss', 'triplet_embedding_loss'], fig_label='exp, [soft,adamw 1e-5,E25] [C->10,A->5e-5,E25] [C->32,E20] [C->64,E35] [triplet 10,a0.3,E5]')
  # axes, _ = plot.hist_plot_split(["checkpoints/T_keras_mobilenet_basic_adamw_2_emore_hist_E105.json", "checkpoints/T_keras_mobilenet_basic_adamw_2_E105_trip20_0.3_hist.json"], epochs, axes=axes, customs=customs+['center_embedding_loss', 'triplet_embedding_loss'], fig_label='exp, [soft,adamw 5e-5, E105] [triplet 20,a0.3,E5]')

  # axes, _ = plot.hist_plot_split(["checkpoints/T_keras_mobilenet_basic_adamw_2_emore_hist_E105.json", "checkpoints/T_keras_mobilenet_basic_adamw_2_E105_trip32_0.3_hist.json"], epochs, axes=axes, customs=customs+['center_embedding_loss', 'triplet_embedding_loss'], fig_label='exp, [soft ls=0.1 + center, adamw 5e-5, E105] [triplet 32,a0.3,E25]')
  # axes, _ = plot.hist_plot_split(["checkpoints/T_keras_mobilenet_basic_adamw_2_emore_hist_E105.json", "checkpoints/T_keras_mobilenet_basic_adamw_2_E105_trip64_0.2_hist.json"], epochs, axes=axes, customs=customs+['center_embedding_loss', 'triplet_embedding_loss'], fig_label='exp, [soft ls=0.1 + center, adamw 5e-5, E105] [triplet 64,a0.2,E5]')

  # axes, _ = plot.hist_plot_split(["checkpoints/T_keras_mobilenet_basic_adamw_2_emore_hist_E25_bottleneck.json", "checkpoints/T_keras_mobilenet_basic_adamw_E25_arcloss_emore_hist.json"], epochs, axes=axes, customs=customs, fig_label=' exp, [soft ls=0.1 + center, adamw 1e-5, E25] [arc, adamw 5e-5, E35]')

  axes, _ = plot.hist_plot_split(["checkpoints/T_keras_mobilenet_basic_adamw_2_emore_hist_E70.json", "checkpoints/T_keras_mobilenet_basic_adamw_2_E70_arc_emore_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp, [soft ls=0.1 + center, adamw 5e-5, E70] [arc, E35]')
  # axes, _ = plot.hist_plot_split(["checkpoints/T_mobilenetv2_adamw_5e5_arc_E80_hist.json", "checkpoints/T_mobilenetv2_adamw_5e5_E80_arc_trip64_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp, [soft ls=0.1 + center, adamw 5e-5, E80] [arc, trip64, E20]')
  # axes, _ = plot.hist_plot_split(["checkpoints/T_mobilenetv2_adamw_5e5_arc_E80_hist.json", "checkpoints/T_mobilenetv2_adamw_5e5_E80_arc_trip32_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp, [soft ls=0.1 + center, adamw 5e-5, E80] [arc, trip32, E20]')
  axes, _ = plot.hist_plot_split(["checkpoints/T_mobilenetv2_adamw_5e5_arc_E80_hist.json", "checkpoints/T_mobilenetv2_adamw_5e5_E80_arc_trip64_A10_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp, [soft ls=0.1 + center, adamw 5e-5, E80] [arc, trip32, A10]')

  axes, _ = plot.hist_plot_split("checkpoints/T_keras_mobilenet_basic_adamw_3_emore_hist.json", epochs, axes=axes, customs=customs, fig_label='exp, [soft, adamw 5e-5, dr 0.4, E10]')
  axes, _ = plot.hist_plot_split("checkpoints/T_mobilenetv2_adamw_5e5_hist.json", epochs, axes=axes, customs=customs, fig_label='exp, [soft, adamw 5e-5, dr 0, E60] [C->64, E20] [C->128, E20]')

  # axes, _ = plot.hist_plot_split(["checkpoints/T_mobilenetv2_adamw_1e4_hist_E80.json", "checkpoints/T_mobilenetv2_adamw_1e4_E80_arc_trip64_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp, [soft, adamw 1e-4, dr 0, E10] [C->64, E20], [arc, trip64, E20]')
  # axes, _ = plot.hist_plot_split(["checkpoints/T_mobilenetv2_adamw_1e4_hist_E80.json", "checkpoints/T_mobilenetv2_adamw_1e4_E80_arc_trip64_A10_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp, [soft, adamw 1e-4, dr 0, E10] [C->64, E20], [arc, trip32, A10]')

  axes, _ = plot.hist_plot_split(["checkpoints/mobilenet_adamw_BS256_E80_hist.json", "checkpoints/mobilenet_adamw_BS256_E80_arc_tripD_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp,mobilenet,BS256,[soft,adamw 5e-5,dr 0 E80] [arc+trip 64,alpha decay,E40]')

  axes, _ = plot.hist_plot_split(["checkpoints/T_mobilenet_adamw_5e5_BS1024_hist.json", "checkpoints/T_mobilenet_adamw_5e5_arc_trip64_BS1024_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp,mobilenet,BS1024,[soft,adamw 5e-5,dr 0 E80] [arc+trip 64,alpha decay,E40]')
  axes, _ = plot.hist_plot_split(["checkpoints/T_mobilenetv3L_adamw_5e5_BS1024_hist.json", "checkpoints/T_mobilenetv3L_adamw_5e5_arc_trip64_BS1024_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp,mobilenetV3L,BS1024,[soft,adamw 5e-5,dr 0 E80] [arc+trip 64,alpha decay,E40]')

  axes, _ = plot.hist_plot_split(["checkpoints/T_mobilenet_adamw_5e5_BS1024_hist.json", "checkpoints/T_mobilenet_adamw_5e5_arc_trip32_BS1024_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp,mobilenet,BS1024,[soft,adamw 5e-5,dr 0 E80] [arc+trip 32,alpha decay,E40]')
  axes, _ = plot.hist_plot_split(["checkpoints/T_mobilenetv3L_adamw_5e5_BS1024_hist.json", "checkpoints/T_mobilenetv3L_adamw_5e5_arc_trip32_BS1024_hist.json"], epochs, axes=axes, customs=customs, fig_label='exp,mobilenetV3L,BS1024,[soft,adamw 5e-5,dr 0 E80] [arc+trip 32,alpha decay,E40]')
  ```
## Mobilnet emore BS256
  ```py
  import plot
  axes = None
  customs = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  epochs = [10, 10, 10, 10, 10, 10, 10, 10]
  names = ["Softmax + Center = 1", "Softmax + Center = 10", "Softmax + Center = 20", "Softmax + Center = 30", "Softmax + Center = 40", "Softmax + Center = 50", "Softmax + Center = 60", "Softmax + Center = 70"]
  axes, pre = plot.hist_plot_split("checkpoints/keras_mxnet_test_sgdw_hist.json", epochs, names=names, axes=axes, customs=customs, fig_label='exp, mobilenet, [soft ls=0.1 + center, adamw 5e-5, dr 0, E10]', eval_split=True)

  epochs = [2, 10, 10, 10, 10, 50]
  names = ["Arcloss Bottleneck Only", "Arcloss + Triplet 64 alpha 0.35", "Arcloss + Triplet 64 alpha 0.3", "Arcloss + Triplet 64 alpha 0.25", "Arcloss + Triplet 64 alpha 0.2", "Arcloss + Triplet 64 alpha 0.15"]
  axes, _ = plot.hist_plot_split("checkpoints/mobilenet_adamw_BS256_E80_arc_c64_hist.json", epochs, names=names, axes=axes, customs=customs, fig_label='exp, mobilenet, [soft, E80] [arc, E40]', pre_item=pre, init_epoch=80)
  axes, _ = plot.hist_plot_split("checkpoints/mobilenet_adamw_BS256_E80_arc_trip_hist.json", epochs, axes=axes, customs=customs, fig_label='exp,mobilenet,[soft, E80] [arc+trip 32,E20] [arc+trip 64,alpha0.3,E40]', pre_item=pre, init_epoch=80)
  # axes, _ = plot.hist_plot_split("checkpoints/mobilenet_adamw_BS256_E80_arc_trip128_hist.json", epochs, axes=axes, customs=customs, fig_label='exp,mobilenet,[soft, E80] [arc+trip 128,alpha0.3,E40]', pre_item=pre, init_epoch=80)
  axes, _ = plot.hist_plot_split("checkpoints/mobilenet_adamw_BS256_E80_arc_trip64_hist.json", epochs, axes=axes, customs=customs, fig_label='exp,mobilenet,[soft, E80] [arc+trip 64,alpha0.3,E40]', pre_item=pre, init_epoch=80)
  axes, _ = plot.hist_plot_split("checkpoints/mobilenet_adamw_BS256_E80_arc_tripD_hist.json", epochs, axes=axes, customs=customs, fig_label='exp,mobilenet,[soft, E80] [arc+trip 64,alpha decay,E40]', pre_item=pre, init_epoch=80)

  axes, _ = plot.hist_plot_split("checkpoints/mobilenet_adamw_5e5_dr0_BS256_triplet_E20_arc_emore_hist.json", [20, 2, 20, 20, 20, 20], axes=axes, customs=customs, fig_label='exp,mobilenet,[soft+Triplet,E20] [arc+trip,alpha decay,E80]')
  ```
  ```py
  import plot
  axes = None
  customs = ["cfp_fp", "agedb_30", "lfw", "center_embedding_loss", "triplet_embedding_loss", "lr"]
  epochs = [10, 10, 10, 10, 10, 10, 10, 10, 2, 10, 10, 10, 10, 50]
  names_1 = ["Softmax + Center = 1", "Softmax + Center = 10", "Softmax + Center = 20", "Softmax + Center = 30", "Softmax + Center = 40", "Softmax + Center = 50", "Softmax + Center = 60", "Softmax + Center = 70"]
  names_2 = ["Arcloss Bottleneck Only", "Arcloss + Triplet 64 alpha 0.35", "Arcloss + Triplet 64 alpha 0.3", "Arcloss + Triplet 64 alpha 0.25", "Arcloss + Triplet 64 alpha 0.2", "Arcloss + Triplet 64 alpha 0.15"]

  axes, pre = plot.hist_plot_split(["checkpoints/keras_mxnet_test_sgdw_hist.json", "checkpoints/mobilenet_adamw_BS256_E80_arc_tripD_hist.json"], epochs, axes=axes, customs=customs, names=names_1+names_2)
  pp = {"epochs": epochs, "customs": customs, "axes": axes}

  # axes, pre = plot.hist_plot_split("checkpoints/keras_mobilenet_emore_adamw_1e4_soft_arc_tripD_hist.json", **pp)
  # axes, pre = plot.hist_plot_split("checkpoints/keras_mobilenet_emore_nadam_REG_1e3_soft_arc_tripD_hist.json", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/keras_mobilenet_emore_nadam_soft_arc_tripD_hist.json", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/keras_mobilenet_emore_adamw_5e5_soft_center_1e2D_arc_tripD_hist.json", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/keras_mobilenet_emore_adamw_5e5_soft_center_1e1D_arc_tripD_hist.json", **pp)
  ```
## ResNet34 CASIA
  ```py
  axes = None
  customs = ["cfp_fp", "agedb_30", "lfw", "lr", "regular_loss"]
  epochs = [1, 19, 10, 50]
  names = ["Warmup", "Arcfacelose learning rate 0.1", "Arcfacelose learning rate 0.01", "Arcfacelose learning rate 0.001"]

  axes, pre = plot.hist_plot_split("checkpoints/mxnet_r34_wdm1_new.json", epochs, axes=axes, customs=customs, names=names, fig_label="Original MXNet")
  pp = {"epochs": epochs, "customs": customs, "axes": axes}
  # axes, pre = plot.hist_plot_split("checkpoints/MXNET_r34_casia.json", epochs, axes=axes, customs=customs)

  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_baseline_SGD_lr1e1_random0_arcT4_32_E5_BS512_casia_hist.json", fig_label="TF SGD baseline", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wdm1_random0_arcT4_32_E5_BS512_casia_hist.json", fig_label="TF SGDW 5e-4", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_SGDW_1e3_lr1e1_random0_arc_S32_E1_BS512_casia_4_hist.json", fig_label="TF SGDW 1e-3", **pp)

  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_REG_BN_SGD_5e4_lr1e1_random0_arcT4_S32_E1_BS512_casia_3_hist_no_reg.json", fig_label="TF SGD, l2 5e-4", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_SGD_REG_1e3_clone_lr1e1_random0_arc_S32_E1_BS512_casia_4_hist.json", fig_label="TF SGD, l2 1e-3", **pp)

  choose_accuracy([
      "checkpoints/mxnet_r34_wdm1_new.json",
      "checkpoints/MXNET_r34_casia.json",
      "checkpoints/NNNN_resnet34_MXNET_E_baseline_SGD_lr1e1_random0_arcT4_32_E5_BS512_casia_hist.json",
      "checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wdm1_random0_arcT4_32_E5_BS512_casia_hist.json",
      "checkpoints/NNNN_resnet34_MXNET_E_SGDW_1e3_lr1e1_random0_arc_S32_E1_BS512_casia_4_hist.json",
      "checkpoints/NNNN_resnet34_MXNET_E_REG_BN_SGD_5e4_lr1e1_random0_arcT4_S32_E1_BS512_casia_3_hist_no_reg.json",
      "checkpoints/NNNN_resnet34_MXNET_E_SGD_REG_1e3_clone_lr1e1_random0_arc_S32_E1_BS512_casia_4_hist.json",
  ])
  ```
  ```py
  import plot
  axes = None
  customs = ["cfp_fp", "agedb_30", "lfw", "lr"]
  epochs = [1, 19, 10, 50]
  limit_loss_max = 80
  names = ["Warmup", "Arcfacelose learning rate 0.1", "Arcfacelose learning rate 0.01", "Arcfacelose learning rate 0.001"]
  pp = {"epochs": epochs, "customs": customs, "axes": None}
  axes, pre = plot.hist_plot_split("checkpoints/MXNET_r34_casia.json", epochs, axes=axes, customs=customs, names=names, limit_loss_max=limit_loss_max, fig_label='Orignal MXNet Resnet34')
  axes, pre = plot.hist_plot_split("checkpoints/mxnet_r34_wdm1.json", epochs, axes=axes, customs=customs)
  axes, pre = plot.hist_plot_split("checkpoints/mxnet_r34_wdm1_lazy_false.json", epochs, axes=axes, customs=customs)
  axes, pre = plot.hist_plot_split("checkpoints/r34_wdm1_lazy_false_wd0_no_rescale.json", epochs, axes=axes, customs=customs)
  axes, pre = plot.hist_plot_split("checkpoints/r34_wdm1_lazy_false_wd0.json", epochs, axes=axes, customs=customs, fig_label=None)

  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random0_arc32_E1_arcT4_BS512_casia_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label='Resnet34, SGDW, E, weight_decay_mul 10, random 0, arc32T4, arcT4, bs512')
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e5_dr4_lr1e1_wdm1_random0_arcT4_BS512_casia_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wdm1_random0_arcT4_32_E5_BS512_casia_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_baseline_SGD_lr1e1_random0_arcT4_32_E5_BS512_casia_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)


  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e5_dr4_lr1e1_wdm1_random0_arcT4_BS512_casia_3_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e5_dr4_lr1e1_wdm10_random0_arcT4_BS512_casia_4_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)

  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_rand3_sgdw_5e4_dr0.4_wdm10_soft_E1_arc_BS480_casia_3_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label='Resnet34, SGDW, weight_decay_mul 10, E, random 3, soft, arcT4, bs480')

  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e5_dr4_lr1e1_wd10_random0_arcT4_BS512_casia_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)

  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e4_dr4_lr1e1_wdm10_random0_arcT4_BS512_casia_2_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e5_dr4_lr1e1_wdm10_random0_arcT4_BS512_casia_2_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e4_dr4_lr1e1_wdm10_random0_arcT4_BS512_casia_3_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)

  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e4_dr4_lr1e1_wdm10D_random0_arcT4_BS512_casia_3_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e5_dr4_lr1e1_wdm1_random0_arcT4_BS512_casia_3_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e5_dr4_lr1e1_wdm10_random0_arcT4_S16_S32_BS512_casia_5_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e5_dr4_lr1e1_wdm10_random0_arcT4_BS512_casia_4_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label=None)


  axes, pre = plot.hist_plot_split("checkpoints/TF_resnet50_MXNET_E_sgdw_5e4_dr0.4_wdm10_soft_E10_arc_casia_hist.json", epochs, axes=axes, customs=customs, limit_loss_max=limit_loss_max, fig_label='Original TF resnet50, SGDW, weight_decay_mul 10, E')
  ```
  ```py
  import tensorflow_addons as tfa
  import train, losses

  data_basic_path = '/datasets/'
  data_path = data_basic_path + 'faces_casia_112x112_folders'
  eval_paths = [data_basic_path + ii for ii in ['faces_casia/lfw.bin', 'faces_casia/cfp_fp.bin', 'faces_casia/agedb_30.bin']]

  basic_model = train.buildin_models("resnet34", dropout=0.4, emb_shape=512, output_layer='E', bn_momentum=0.9, bn_epsilon=2e-5)
  basic_model = train.add_l2_regularizer_2_model(basic_model, 1e-4, apply_to_batch_normal=True)
  tt = train.Train(data_path, save_path='NNNN_resnet34_MXNET_E_REG_BN_SGD_1e4_lr1e1_random0_arcT4_S32_E1_BS512_casia_2.h5',
      eval_paths=eval_paths, basic_model=basic_model, model=None, lr_base=0.1, lr_decay=0.1, lr_decay_steps=[20, 30],
      batch_size=512, random_status=0, output_wd_multiply=1e-4, weight_decay=1)

  # AA = tfa.optimizers.extend_with_decoupled_weight_decay(tfa.optimizers.LazyAdam)
  # optimizer = AA(learning_rate=0.01, weight_decay=5e-5)
  # optimizer = tfa.optimizers.SGDW(learning_rate=0.1, weight_decay=5e-3, momentum=0.9)
  optimizer = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
  sch = [
      # {"loss": keras.losses.CategoricalCrossentropy(), "epoch": 1, "optimizer": optimizer},
      {"loss": losses.ArcfaceLossT4(scale=32), "epoch": 1, "optimizer": optimizer},
      {"loss": losses.ArcfaceLossT4(scale=64), "epoch": 19},
  ]
  tt.train(sch, 0)
  ```
  ```py
  import plot
  axes = None
  customs = ["cfp_fp", "agedb_30", "lfw", "lr"]
  epochs = [1, 19, 10, 50]
  limit_loss_max = 30
  names = ["Warmup", "Arcfacelose learning rate 0.1", "Arcfacelose learning rate 0.01", "Arcfacelose learning rate 0.001"]

  # axes, pre = plot.hist_plot_split("checkpoints/MXNET_r34_casia.json", epochs, axes=axes, customs=customs, names=names)
  # axes, pre = plot.hist_plot_split("checkpoints/mxnet_r34_wdm1.json", epochs, axes=axes, customs=customs)
  axes, pre = plot.hist_plot_split("checkpoints/mxnet_r34_wdm1_lazy_false.json", epochs, axes=axes, customs=customs, names=names)
  pp = {"epochs": epochs, "customs": customs, "axes": axes}

  # axes, pre = plot.hist_plot_split("checkpoints/r34_wdm1_lazy_false_wd0_no_rescale.json", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/r34_wdm1_lazy_false_wd0.json", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random0_arc32_E1_arcT4_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)

  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e5_dr4_lr1e1_wdm1_random0_arcT4_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wdm1_random0_arcT4_32_E5_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4S_dr4_lr1e1_wdm1_random0_arcT4_32_E5_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_baseline_SGD_lr1e1_random0_arcT4_32_E5_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)

  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_REG_SGD_5e4_lr1e1_random0_arcT4_32_E5_BS512_casia_hist.json", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_REG_BN_SGD_5e4_lr1e1_random0_arcT4_32_E5_BS512_casia_hist.json", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_REG_SGD_5e5_lr1e1_random0_arcT4_32_E1_BS512_casia_hist.json", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_REG_BN_SGD_5e5_lr1e1_random0_arcT4_BS512_casia_hist.json", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_REG_BN_SGD_5e4_lr1e1_random0_arcT4_S32_E1_BS512_casia_hist.json", **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_REG_BN_SGD_1e4_lr1e1_random0_arcT4_S32_E1_BS512_casia_hist.json", **pp)
  ```
  ```py
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e5_dr4_lr1e1_wdm10_random0_arcT4_BS512_casia_4_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e5_dr4_lr1e1_wdm1_random0_arcT4_BS512_casia_3_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e4_dr4_lr1e1_wdm10D_random0_arcT4_BS512_casia_3_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e4_dr4_lr1e1_wdm10_random0_arcT4_BS512_casia_3_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e5_dr4_lr1e1_wdm10_random0_arcT4_BS512_casia_2_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_fix_gamma_sgdw_5e4_dr4_lr1e1_wdm10_random0_arcT4_BS512_casia_2_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e5_dr4_lr1e1_wd10_random0_arcT4_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random0_arc32_E1_arcT4_BS480_casia_2_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random0_arc32_E1_arcT4_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random2_arc32_E1_arcT4_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random2_arc32_E1_arc_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random0_arc32_E1_arc_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random0_soft_E1_arc_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random0_soft_E1_arc_trip_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_adamw_5e5_dr4_lr1e3_random0_soft_E1_arc_trip_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_adamw_5e5_dr4_lr1e3_random0_soft_E1_arc_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4C_dr4_lr1e1_random0_soft_E1_arc_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_random2_soft_E1_arc_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_admw_5e5_E3_sgdw_5e4_dr4_lr1e3_random3_soft_E1_arc_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_random3_soft_E1_arc_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_admw_5e5_dr4_lr1e3_random3_soft_E1_arc_BS512_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_admw_5e5_E5_sgdw5e4_dr4_lr1e2_random3_soft_E1_arc_BS480_casia_4_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_rand3_sgdw_5e4_dr0.4_wdm10_soft_E1_arc_BS480_casia_3_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_rand3_sgdw_5e4_dr0.4_wdm10_soft_E1_arc_trip32_BS480_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_rand3_sgdw_5e4_dr0.4_soft_E1_arc_BS480_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_rand3_sgdw_5e4_dr0.4_wdm10_soft_E1_arc_BS480_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_GDC_sgdw_5e4_dr0.4_wdm10_soft_E1_arc_BS480_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  axes, pre = plot.hist_plot_split("checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr0.4_wdm10_soft_E1_arc_BS480_casia_hist.json", limit_loss_max=limit_loss_max, **pp)
  ```
## Comparing early softmax training
  ```py
  import plot
  customs = ["agedb_30"]
  epochs = [15, 10]
  axes, _ = plot.hist_plot_split("checkpoints/keras_mobilefacenet_256_hist_all.json", epochs, customs=customs, axes=None, fig_label='Mobilefacenet, BS=160')
  axes, _ = plot.hist_plot_split("checkpoints/keras_mobile_facenet_emore_hist.json", epochs, customs=customs, axes=axes, fig_label='Mobilefacenet, BS=768')

  axes, _ = plot.hist_plot_split("checkpoints/keras_se_mobile_facenet_emore_hist.json", epochs, customs=customs, axes=axes, fig_label='se_mobilefacenet, BS=680, label_smoothing=0.1')
  axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_VI_hist.json', epochs, customs=customs, axes=axes, fig_label="se_mobilefacenet, Cosine, BS = 640, LS=0.1")
  axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_VII_nadam_hist.json', epochs, customs=customs, axes=axes, fig_label="se_mobilefacenet, Cosine, nadam, BS = 640, nadam, LS=0.1", init_epoch=3)
  axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_VIII_PR_hist.json', epochs, customs=customs, axes=axes, fig_label="new se_mobilefacenet, Cosine, center, BS = 640, nadam, LS=0.1")
  axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_IX_hist.json', epochs, customs=customs, axes=axes, fig_label="new se_mobilefacenet, Cosine, no center, BS = 640, nadam, LS=0.1")
  axes, _ = plot.hist_plot_split('checkpoints/keras_se_mobile_facenet_emore_X_hist.json', epochs, customs=customs, axes=axes, fig_label="new se_mobilefacenet, Cosine, center, leaky, BS = 640, nadam, LS=0.1")

  axes, _ = plot.hist_plot_split("checkpoints/keras_resnet101_512_II_hist.json", epochs, customs=customs, axes=axes, fig_label='Resnet101, BS=128')
  axes, _ = plot.hist_plot_split("checkpoints/keras_resnet101_emore_hist.json", epochs, customs=customs, axes=axes, fig_label='Resnet101, BS=1024')
  axes, _ = plot.hist_plot_split("checkpoints/keras_resnet101_emore_II_hist.json", epochs, customs=customs, axes=axes, fig_label='Resnet101, BS=960, label_smoothing=0.1')
  axes, _ = plot.hist_plot_split("checkpoints/keras_ResNest101_emore_hist.json", epochs, customs=customs, axes=axes, fig_label='Resnest101, BS=600, label_smoothing=0.1')
  axes, _ = plot.hist_plot_split("checkpoints/keras_EB4_emore_hist.json", epochs, names=["Softmax", "Margin Softmax"], customs=customs, axes=axes, fig_label='EB4, BS=420, label_smoothing=0.1')

  axes[0].plot((2, 15), (0.3807, 0.3807), 'k:')
  axes[1].plot((2, 15), (0.9206, 0.9206), 'k:')
  axes[0].plot((2, 15), (0.6199, 0.6199), 'k:')
  axes[1].plot((2, 15), (0.8746, 0.8746), 'k:')
  axes[0].figure.savefig('./checkpoints/softmax_compare.svg')
  ```
  ![](checkpoints/softmax_compare.svg)
## MXNet record
  ```sh
  $ CUDA_VISIBLE_DEVICES="1" python -u train_softmax.py --data-dir /datasets/faces_casia --network "r34" --loss-type 4 --prefix "./model/mxnet_r34_wdm1_casia" --per-batch-size 512 --lr-steps "19180,28770" --margin-s 64.0 --margin-m 0.5 --ckpt 1 --emb-size 512 --fc7-wd-mult 1.0 --wd 0.0005 --verbose 959 --end-epoch 38400 --ce-loss

  Called with argument: Namespace(batch_size=512, beta=1000.0, beta_freeze=0, beta_min=5.0, bn_mom=0.9, ce_loss=True, ckpt=1, color=0, ctx_num=1, cutoff=0, data_dir='/datasets/faces_casia', easy_margin=0, emb_size=512, end_epoch=38400, fc7_lr_mult=1.0, fc7_no_bias=False, fc7_wd_mult=1.0, gamma=0.12, image_channel=3, image_h=112, image_size='112,112', image_w=112, images_filter=0, loss_type=4, lr=0.1, lr_steps='19180,28770', margin=4, margin_a=1.0, margin_b=0.0, margin_m=0.5, margin_s=64.0, max_steps=0, mom=0.9, network='r34', num_classes=10572, num_layers=34, per_batch_size=512, power=1.0, prefix='./model/mxnet_r34_wdm1_casia', pretrained='', rand_mirror=1, rescale_threshold=0, scale=0.9993, target='lfw,cfp_fp,agedb_30', use_deformable=0, verbose=959, version_act='prelu', version_input=1, version_multiplier=1.0, version_output='E', version_se=0, version_unit=3, wd=0.0005)
  ```
  ```py
  Called with argument: Namespace(batch_size=512, beta=1000.0, beta_freeze=0, beta_min=5.0, bn_mom=0.9, ce_loss=True,
  ckpt=1, color=0, ctx_num=1, cutoff=0, data_dir='/datasets/faces_casia', easy_margin=0, emb_size=512, end_epoch=38400,
  fc7_lr_mult=1.0, fc7_no_bias=False, fc7_wd_mult=1.0, gamma=0.12, image_channel=3, image_h=112, image_size='112,112',
  image_w=112, images_filter=0, loss_type=4, lr=0.1, lr_steps='19180,28770', margin=4, margin_a=1.0, margin_b=0.0,
  margin_m=0.5, margin_s=64.0, max_steps=0, mom=0.9, network='r34', num_classes=10572, num_layers=34, per_batch_size=512,
  power=1.0, prefix='./model/mxnet_r34_wdm1_lazy_false_casia', pretrained='', rand_mirror=1, rescale_threshold=0, scale=0.9993,
  target='lfw,cfp_fp,agedb_30', use_deformable=0, verbose=959, version_act='prelu', version_input=1, version_multiplier=1.0,
  version_output='E', version_se=0, version_unit=3, wd=0.0005)
  ```
  ```py
  from train_softmax import *
  sys.argv.extend('--data-dir /datasets/faces_casia --network "r34" --loss-type 4 --prefix "./model/mxnet_r34_wdm1_lazy_false_wd0_casia" --per-batch-size 512 --lr-steps "19180,28770" --margin-s 64.0 --margin-m 0.5 --ckpt 1 --emb-size 512 --fc7-wd-mult 1.0 --wd 5e-4 --verbose 959 --end-epoch 38400 --ce-loss'.replace('"', '').split(' '))
  args = parse_args()
  ```
  ```py
  CUDA_VISIBLE_DEVICES='0' python train.py --network r34 --dataset casia --loss 'arcface' --per-batch-size 512 --lr-steps '19180,28770' --verbose 959

  ```
***

# Label smoothing
  - **Train schedulers**
    ```py
    basic_model = train.buildin_models("MobileNet", dropout=0.4, emb_shape=256)
    tt = train.Train(..., random_status=0)
    sch = [{"loss": keras.losses.CategoricalCrossentropy(label_smoothing=0), "optimizer": "nadam", "epoch": 3}]
    sch = [{"loss": keras.losses.CategoricalCrossentropy(label_smoothing=0.1), "optimizer": "nadam", "epoch": 3}]
    tt.train(sch, 0)

    sch = [{"loss": losses.ArcfaceLoss(label_smoothing=0), "epoch": 5}]
    tt.train(sch, 3)

    sch = [{"loss": losses.ArcfaceLoss(label_smoothing=0), "epoch": 3}]
    sch = [{"loss": losses.ArcfaceLoss(label_smoothing=0.1), "epoch": 3}]
    tt.train(sch, 8)

    tt = train.Train(..., random_status=3)
    sch = [{"loss": losses.ArcfaceLoss(label_smoothing=0), "epoch": 3}]
    tt.train(sch, 8)
    ```
  - **Result**
    ```py
    import plot
    axes, _ = plot.hist_plot_split('checkpoints/keras_mobilenet_256_hist.json', [3], init_epoch=0, axes=None, fig_label="LS=0, Softmax")
    axes, pre_1 = plot.hist_plot_split('checkpoints/keras_mobilenet_ls_0.1_256_hist.json', [3, 5], names=["Softmax", "Arcface"], init_epoch=0, axes=axes, fig_label="LS=0.1, Softmax")
    axes, _ = plot.hist_plot_split('checkpoints/keras_mobilenet_arcface_ls_0_256_hist.json', [3], init_epoch=8, pre_item=pre_1, axes=axes, fig_label="LS=0, Arcface")
    axes, _ = plot.hist_plot_split('checkpoints/keras_mobilenet_arcface_ls_0.1_256_hist.json', [3], init_epoch=8, pre_item=pre_1, axes=axes, fig_label="LS=0.1, Arcface")
    axes, _ = plot.hist_plot_split('checkpoints/keras_mobilenet_arcface_randaug_256_hist.json', [3], init_epoch=8, pre_item=pre_1, axes=axes, fig_label="Random=3, LS=0, Arcface")
    axes, _ = plot.hist_plot_split('checkpoints/keras_mobilenet_arcface_randaug_ls0.1_256_hist.json', [5], names=["Arcface"], init_epoch=8, pre_item=pre_1, axes=axes, fig_label="Random=3, LS=0.1, Arcface")
    axes[2].legend(fontsize=8, loc='lower center')
    axes[0].figure.savefig('./checkpoints/label_smoothing.svg')
    ```
    ![](checkpoints/label_smoothing.svg)
  - **CASIA**
    ```py
    import plot
    axes = None
    customs = ["cfp_fp", "agedb_30", "lfw", "lr"]
    epochs = [5, 5, 10, 10, 40]
    names = ["ArcFace Scale 16, learning rate 0.1", "ArcFace Scale 32, learning rate 0.1", "ArcFace Scale 64, learning rate 0.1", "ArcFace Scale 64, learning rate 0.01", "ArcFace Scale 64, learning rate 0.001"]
    axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_bs400_hist.json", epochs, axes=axes, customs=customs, names=names, fig_label="Mobilenet, emb256, dr0, bs400, base")
    axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_nesterov_emb256_bs400_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb256, dr0, bs400, nesterov_emb256")

    axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_ls1_emb256_bs400_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb256, dr0, bs400, ls 0.1")
    axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_nesterov_ls1_emb256_bs400_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb256, dr0, bs400, nesterov, ls 0.1")

    aa = [
        "checkpoints/TT_mobilenet_base_bs400_hist.json",
        "checkpoints/TT_mobilenet_base_emb512_dr4_bs400_hist.json",
        "checkpoints/TT_mobilenet_base_nesterov_emb256_bs400_hist.json",
        "checkpoints/TT_mobilenet_base_nesterov_emb512_dr4_bs400_hist.json",
    ]

    choose_accuracy(aa)
    ```
***

# Nesterov
  ```py
  import plot
  axes = None
  customs = ["cfp_fp", "agedb_30", "lfw", "lr"]
  epochs = [5, 5, 10, 10, 40]
  names = ["ArcFace Scale 16, learning rate 0.1", "ArcFace Scale 32, learning rate 0.1", "ArcFace Scale 64, learning rate 0.1", "ArcFace Scale 64, learning rate 0.01", "ArcFace Scale 64, learning rate 0.001"]
  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_bs400_hist.json", epochs, axes=axes, customs=customs, names=names, fig_label='Mobilnet, CASIA, baseline, topk1, wdm1')
  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_emb512_dr4_bs400_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb512, dr0.4, bs400, base")

  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_nesterov_emb256_bs400_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb256, dr0, bs400, nesterov_emb256")
  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_nesterov_emb512_dr4_bs400_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb512, dr0.4, bs400, nesterov_emb512")

  aa = [
      "checkpoints/TT_mobilenet_base_bs400_hist.json",
      "checkpoints/TT_mobilenet_base_emb512_dr4_bs400_hist.json",
      "checkpoints/TT_mobilenet_base_nesterov_emb256_bs400_hist.json",
      "checkpoints/TT_mobilenet_base_nesterov_emb512_dr4_bs400_hist.json",
  ]

  choose_accuracy(aa)
  ```

  | emb | Dropout | nesterov | Max lfw       | Max cfp_fp    | Max agedb_30  |
  | --- | ------- | -------- | ------------- | ------------- | ------------- |
  | 256 | 0       | False    | 0.9822,38     | 0.8694,44     | 0.8695,36     |
  | 512 | 0.4     | False    | **0.9837**,43 | 0.8491,47     | 0.8745,40     |
  | 256 | 0       | True     | 0.9830,30     | **0.8739**,40 | 0.8772,34     |
  | 512 | 0.4     | True     | 0.9828,40     | 0.8673,42     | **0.8810**,31 |
***

# Sub center
  - **MXNet**
    ```sh
    CUDA_VISIBLE_DEVICES='1' python train_parall.py --network r50 --per-batch-size 512
    INFO:root:Iter[20] Batch [8540] Speed: 301.72 samples/sec
    {fc7_acc} 236000 0.80078125
    CELOSS,236000,1.311261
    [lfw][236000]Accuracy-Flip: 0.99817+-0.00273
    [cfp_fp][236000]Accuracy-Flip: 0.97557+-0.00525
    [agedb_30][236000]Accuracy-Flip: 0.98167+-0.00707

    CUDA_VISIBLE_DEVICES='1' python drop.py --data /datasets/faces_emore --model models/r50-arcface-emore/model,1 --threshold 75 --k 3 --output /datasets/faces_emore_topk3_1
    ```
  - **Result and plot**
    ```py
    import plot
    axes = None
    customs = ["cfp_fp", "agedb_30", "lfw", "lr"]
    epochs = [5, 5, 10, 10, 40]
    names = ["ArcFace Scale 16, learning rate 0.1", "ArcFace Scale 32, learning rate 0.1", "ArcFace Scale 64, learning rate 0.1", "ArcFace Scale 64, learning rate 0.01", "ArcFace Scale 64, learning rate 0.001"]
    axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_bs400_hist.json", epochs, axes=axes, customs=customs, names=names, fig_label='Mobilnet, CASIA, baseline, topk1, wdm1')
    axes, pre = plot.hist_plot_split("checkpoints/TT_mobilenet_topk_bs400_hist.json", epochs, axes=axes, customs=customs, fig_label='topk3, wdm1')
    axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_topk1_bs400_hist.json", epochs, axes=axes, customs=customs, fig_label='topk3->1, wdm1')

    axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_topk1_BTNO_bs400_hist.json", epochs, axes=axes, customs=customs, pre_item=pre, fig_label='topk3->1, wdm1, bottleneckOnly')
    axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_topk1_BTNO_init_E40_bs400_hist.json", epochs, axes=axes, customs=customs, pre_item=pre, fig_label='topk3->1, wdm1, bottleneckOnly, init_epoch40')

    import json
    aa = ["checkpoints/TT_mobilenet_base_bs400_hist.json",
        "checkpoints/TT_mobilenet_topk_bs400_hist.json",
        "checkpoints/TT_mobilenet_topk1_bs400_hist.json",
        "checkpoints/TT_mobilenet_topk1_BTNO_bs400_hist.json",
        "checkpoints/TT_mobilenet_topk1_BTNO_init_E40_bs400_hist.json",
    ]

    choose_accuracy(aa)
    ```

    | Scenario                                    | Max lfw    | Max cfp_fp | Max agedb_30 |
    | ------------------------------------------- | ---------- | ---------- | ------------ |
    | Baseline, topk 1                            | 0.9822     | 0.8694     | 0.8695       |
    | TopK 3                                      | 0.9838     | **0.9044** | 0.8743       |
    | TopK 3->1                                   | 0.9838     | 0.8960     | 0.8768       |
    | TopK 3->1, bottleneckOnly, initial_epoch=0  | **0.9878** | 0.8920     | **0.8857**   |
    | TopK 3->1, bottleneckOnly, initial_epoch=40 | 0.9835     | **0.9030** | 0.8763       |
***

# Distillation
## MNIST example
  - [Github keras-team/keras-io knowledge_distillation.py](https://github.com/keras-team/keras-io/blob/master/examples/vision/knowledge_distillation.py)
  ```py
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras import layers
  import numpy as np

  # Create the teacher
  teacher = keras.Sequential(
      [
          layers.Conv2D(256, (3, 3), strides=(2, 2), padding="same"),
          layers.LeakyReLU(alpha=0.2),
          layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
          layers.Conv2D(512, (3, 3), strides=(2, 2), padding="same"),
          layers.Flatten(),
          layers.Dense(10),
      ],
      name="teacher",
  )

  # Create the student
  student = keras.Sequential(
      [
          layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same"),
          layers.LeakyReLU(alpha=0.2),
          layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="same"),
          layers.Conv2D(32, (3, 3), strides=(2, 2), padding="same"),
          layers.Flatten(),
          layers.Dense(10),
      ],
      name="student",
  )

  # Prepare the train and test dataset.
  batch_size = 64
  # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  # x_train, x_test = np.reshape(x_train, (-1, 28, 28, 1)), np.reshape(x_test, (-1, 28, 28, 1))
  (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

  # Normalize data
  x_train = x_train.astype("float32") / 255.0
  x_test = x_test.astype("float32") / 255.0

  # Train teacher as usual
  teacher.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  # Train and evaluate teacher on data.
  teacher.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
  teacher.evaluate(x_test, y_test)

  def create_distiller_model(teacher, student, clone=True):
      if clone:
          teacher_copy = keras.models.clone_model(teacher)
          student_copy = keras.models.clone_model(student)
      else:
          teacher_copy, student_copy = teacher, student

      teacher_copy.trainable = False
      student_copy.trainable = True
      inputs = teacher_copy.inputs[0]
      student_output = student_copy(inputs)
      teacher_output = teacher_copy(inputs)
      mm = keras.models.Model(inputs, keras.layers.Concatenate()([student_output, teacher_output]))
      return student_copy, mm

  class DistillerLoss(keras.losses.Loss):
      def __init__(self, student_loss_fn, distillation_loss_fn, alpha=0.1, temperature=10, **kwargs):
          super(DistillerLoss, self).__init__(**kwargs)
          self.student_loss_fn, self.distillation_loss_fn = student_loss_fn, distillation_loss_fn
          self.alpha, self.temperature = alpha, temperature

      def call(self, y_true, y_pred):
          student_output, teacher_output = tf.split(y_pred, 2, axis=-1)
          student_loss = self.student_loss_fn(y_true, student_output)
          distillation_loss = self.distillation_loss_fn(
              tf.nn.softmax(teacher_output / self.temperature, axis=1),
              tf.nn.softmax(student_output / self.temperature, axis=1),
          )
          loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
          return loss

  def distiller_accuracy(y_true, y_pred):
      student_output, _ = tf.split(y_pred, 2, axis=-1)
      return keras.metrics.sparse_categorical_accuracy(y_true, student_output)

  distiller_loss = DistillerLoss(
      student_loss_fn=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      distillation_loss_fn=keras.losses.KLDivergence(),
      alpha=0.1,
      # temperature=100,
      temperature=10,
  )

  student_copy, mm = create_distiller_model(teacher, student)
  mm.compile(optimizer=keras.optimizers.Adam(), loss=distiller_loss, metrics=[distiller_accuracy])
  mm.summary()
  mm.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))

  mm.evaluate(x_test, y_test)
  student_copy.compile(metrics=["accuracy"])
  student_copy.evaluate(x_test, y_test)

  # Train student scratch
  student_scratch = keras.models.clone_model(student)
  student_scratch.compile(
      optimizer=keras.optimizers.Adam(),
      loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[keras.metrics.SparseCategoricalAccuracy()],
  )

  student_scratch.fit(x_train, y_train, epochs=15, validation_data=(x_test, y_test))
  student_scratch.evaluate(x_test, y_test)
  ```
## Embedding
  ```py
  from data import pre_process_folder
  from sklearn.preprocessing import normalize
  from tqdm import tqdm

  def tf_imread(file_path):
      img = tf.io.read_file(file_path)
      img = tf.image.decode_jpeg(img, channels=3) # [0, 255]
      img = tf.image.convert_image_dtype(img, tf.float32) # [0, 1]
      return (img * 2) - 1  # [-1, 1]

  data_path = '/datasets/faces_casia_112x112_folders'
  # model = "checkpoints/TT_mobilenet_topk1_BTNO_bs400_basic_agedb_30_epoch_32_0.885667.h5"
  model = "checkpoints/NNNN_resnet34_MXNET_E_sgdw_5e4_dr4_lr1e1_wd10_random0_arc32_E1_arcT4_BS512_casia_basic_agedb_30_epoch_37_0.946667.h5"
  batch_size = 256
  limit = -1
  dataset_pickle_file_dest= None

  image_names, image_classes, _, dataset_pickle_file_src = pre_process_folder(data_path)
  if limit != -1:
      image_names, image_classes = image_names[:limit], image_classes[:limit]
  AUTOTUNE = tf.data.experimental.AUTOTUNE
  ds = tf.data.Dataset.from_tensor_slices((image_names, image_classes))
  ds = ds.batch(batch_size)
  total = int(np.ceil(len(image_names) // batch_size)) + 1

  bb = tf.keras.models.load_model(model, compile=False)
  imms, labels, embs = [], [], []
  for imm, label in tqdm(ds, "Embedding", total=total):
      imgs = tf.stack([tf_imread(ii) for ii in imm])
      emb = bb(imgs)
      # emb = normalize(bb(img).numpy(), axis=1)
      # emb = normalize(bb.predict(img), axis=1)
      imms.extend(imm.numpy())
      labels.extend(label.numpy())
      embs.extend(normalize(emb.numpy(), axis=1))
  # imms, labels, embs = np.array(imms), np.array(labels), np.array(embs)

  if dataset_pickle_file_dest is None:
      src_name = os.path.splitext(os.path.basename(dataset_pickle_file_src))[0]
      dataset_pickle_file_dest = src_name + "_label_embs_normed_{}.pkl".format(embs[0].shape[0])
  with open(dataset_pickle_file_dest, "wb") as ff:
      pickle.dump({"image_names": imms, "image_classes": labels, "embeddings": embs}, ff)
  ```
  ```py
  def tf_imread(file_path):
      img = tf.io.read_file(file_path)
      img = tf.image.decode_jpeg(img, channels=3) # [0, 255]
      img = tf.image.convert_image_dtype(img, tf.float32) # [0, 1]
      return img

  data_path = "faces_casia_112x112_folders_shuffle_label_embs.pkl"
  batch_size = 64
  aa = np.load(data_path, allow_pickle=True)
  image_names, image_classes, embeddings = aa['image_names'], aa['image_classes'], aa['embeddings']
  classes = np.max(image_classes) + 1
  print(">>>> Image length: %d, Image class length: %d, embeddings: %s" % (len(image_names), len(image_classes), np.shape(embeddings)))
  # >>>> Image length: 490623, Image class length: 490623, embeddings: (490623, 256)

  AUTOTUNE = tf.data.experimental.AUTOTUNE
  dss = tf.data.Dataset.from_tensor_slices((image_names, image_classes, embeddings))
  ds = dss.map(lambda imm, label, emb: (tf_imread(imm), (tf.one_hot(label, depth=classes, dtype=tf.int32), emb)), num_parallel_calls=AUTOTUNE)

  ds = ds.batch(batch_size)  # Use batch --> map has slightly effect on dataset reading time, but harm the randomness
  ds = ds.map(lambda xx, yy: ((xx * 2) - 1, yy))
  ds = ds.prefetch(buffer_size=AUTOTUNE)

  xx = tf.keras.applications.MobileNetV2(include_top=False, weights=None)
  xx.trainable = True
  inputs = keras.layers.Input(shape=(112, 112, 3))
  nn = xx(inputs)
  nn = keras.layers.GlobalAveragePooling2D()(nn)
  nn = keras.layers.BatchNormalization()(nn)
  # nn = layers.Dropout(0)(nn)
  embedding = keras.layers.Dense(256, name="embeddings")(nn)
  logits = keras.layers.Dense(classes, activation='softmax', name="logits")(embedding)

  model = keras.models.Model(inputs, [logits, embedding])

  def distiller_loss(true_emb_normed, pred_emb):
      pred_emb_normed = tf.nn.l2_normalize(pred_emb, axis=-1)
      loss = tf.reduce_sum(tf.square(true_emb_normed - pred_emb_normed), axis=-1)
      return loss

  model.compile(optimizer='adam', loss=[keras.losses.categorical_crossentropy, distiller_loss], loss_weights=[1, 7])
  # model.compile(optimizer='adam', loss=[keras.losses.sparse_categorical_crossentropy, keras.losses.mse], metrics=['accuracy', 'mae'])
  model.summary()
  model.fit(ds)
  ```
## Result
  ```py
  import plot
  axes = None
  customs = ["cfp_fp", "agedb_30", "lfw", "lr", "embedding_loss"]
  epochs = [5, 5, 10, 10, 40]
  names = ["ArcFace Scale 16, learning rate 0.1", "ArcFace Scale 32, learning rate 0.1", "ArcFace Scale 64, learning rate 0.1", "ArcFace Scale 64, learning rate 0.01", "ArcFace Scale 64, learning rate 0.001"]
  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_emb512_dr0_bs400_hist.json", epochs, axes=axes, customs=customs, names=names, fig_label="Mobilenet, emb512, dr0, bs400, base")
  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_base_emb512_dr4_bs400_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb512, dr0.4, bs400, base")

  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_distill_emb512_dr0_bs400_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb512, dr0, bs400, Teacher r34")

  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_distill_emb512_dr0_bs400_r100_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb512, dr0, bs400, Teacher r100")
  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_distill_emb512_dr4_bs400_2_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb512, dr0.4, bs400, Teacher r100")

  # axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_distill_32_emb512_dr4_arcT4_bs400_r100_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb512, dr0.4, distill 32, bs400, arcT4, Teacher r100")

  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_distill_64_emb512_dr4_adamw_lr1e3_arcT4_bs400_r100_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb512, dr0.4, distill 64, bs400, arcT4, adamw, Teacher r100")
  axes, _ = plot.hist_plot_split("checkpoints/TT_mobilenet_distill_64_emb512_dr4_arcT4_bs400_r100_hist.json", epochs, axes=axes, customs=customs, fig_label="Mobilenet, emb512, dr0.4, distill 64, bs400, arcT4, Teacher r100")

  aa = [
      "checkpoints/TT_mobilenet_base_emb512_dr0_bs400_hist.json",
      "checkpoints/TT_mobilenet_base_emb512_dr4_bs400_hist.json",
      "checkpoints/TT_mobilenet_distill_emb512_dr0_bs400_hist.json",
      "checkpoints/TT_mobilenet_distill_emb512_dr0_bs400_r100_hist.json",
      "checkpoints/TT_mobilenet_distill_emb512_dr4_bs400_2_hist.json",
      "checkpoints/TT_mobilenet_distill_32_emb512_dr4_arcT4_bs400_r100_hist.json",
      "checkpoints/TT_mobilenet_distill_64_emb512_dr4_adamw_lr1e3_arcT4_bs400_r100_hist.json",
      "checkpoints/TT_mobilenet_distill_64_emb512_dr4_arcT4_bs400_r100_hist.json",
  ]
  choose_accuracy(aa)
  ```

  | Teacher | Dropout | Optimizer | distill | Max lfw    | Max cfp_fp | Max agedb_30 |
  | ------- | ------- | --------- | ------- | ---------- | ---------- | ------------ |
  | None    | 0       | SGDW      | 0       | 0.9838     | 0.8730     | 0.8697       |
  | None    | 0.4     | SGDW      | 0       | 0.9837     | 0.8491     | 0.8745       |
  | r34     | 0       | SGDW      | 7       | 0.9890     | 0.9099     | 0.9058       |
  | r100    | 0       | SGDW      | 7       | 0.9900     | 0.9111     | 0.9068       |
  | r100    | 0.4     | SGDW      | 7       | 0.9905     | 0.9170     | 0.9112       |
  | r100    | 0.4     | SGDW      | 64      | **0.9938** | 0.9333     | **0.9435**   |
  | r100    | 0.4     | AdamW     | 64      | 0.9920     | **0.9346** | 0.9387       |
***

# IJB
  ```py
  $ time CUDA_VISIBLE_DEVICES='1' python IJB_evals.py -m '/media/SD/tdtest/IJB_release/pretrained_models/MS1MV2-ResNet100-Arcface/model,0' -L -d /media/SD/tdtest/IJB_release -B -b 64 -F
  >>>> loading mxnet model: /media/SD/tdtest/IJB_release/pretrained_models/MS1MV2-ResNet100-Arcface/model 0 [gpu(0)]
  [09:17:15] src/nnvm/legacy_json_util.cc:209: Loading symbol saved by previous version v1.2.0. Attempting to upgrade...
  [09:17:15] src/nnvm/legacy_json_util.cc:217: Symbol successfully upgraded!
  >>>> Loading templates and medias...
  templates: (227630,), medias: (227630,), unique templates: (12115,)
  >>>> Loading pairs...
  p1: (8010270,), unique p1: (1845,)
  p2: (8010270,), unique p2: (10270,)
  label: (8010270,), label value counts: {0: 8000000, 1: 10270}
  >>>> Loading images...
  img_names: (227630,), landmarks: (227630, 5, 2), face_scores: (227630,)
  face_scores value counts: {0.1: 2515, 0.2: 0, 0.3: 62, 0.4: 94, 0.5: 136, 0.6: 197, 0.7: 291, 0.8: 538, 0.9: 223797}
  >>>> Saving backup to: /media/SD/IJB_release/IJBB_backup.npz ...

  Embedding: 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 3557/3557 [17:00<00:00,  3.48it/s]
  >>>> N1D1F1 True True True
  Extract template feature: 100%|███████████████████████████████████████████████████████████████████████████| 12115/12115 [00:02<00:00, 4948.45it/s]
  Verification: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [00:39<00:00,  2.04it/s]
  >>>> N1D1F0 True True False
  Extract template feature: 100%|███████████████████████████████████████████████████████████████████████████| 12115/12115 [00:02<00:00, 5010.94it/s]
  Verification: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [00:38<00:00,  2.12it/s]
  >>>> N1D0F1 True False True
  Extract template feature: 100%|███████████████████████████████████████████████████████████████████████████| 12115/12115 [00:02<00:00, 5018.59it/s]
  Verification: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [00:38<00:00,  2.12it/s]
  >>>> N1D0F0 True False False
  Extract template feature: 100%|███████████████████████████████████████████████████████████████████████████| 12115/12115 [00:02<00:00, 4994.06it/s]
  Verification: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [00:38<00:00,  2.13it/s]
  >>>> N0D1F1 False True True
  Extract template feature: 100%|███████████████████████████████████████████████████████████████████████████| 12115/12115 [00:02<00:00, 4984.76it/s]
  Verification: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [00:38<00:00,  2.12it/s]
  >>>> N0D1F0 False True False
  Extract template feature: 100%|███████████████████████████████████████████████████████████████████████████| 12115/12115 [00:02<00:00, 4997.04it/s]
  Verification: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [00:38<00:00,  2.12it/s]
  >>>> N0D0F1 False False True
  Extract template feature: 100%|███████████████████████████████████████████████████████████████████████████| 12115/12115 [00:02<00:00, 4992.75it/s]
  Verification: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [00:38<00:00,  2.12it/s]
  >>>> N0D0F0 False False False
  Extract template feature: 100%|███████████████████████████████████████████████████████████████████████████| 12115/12115 [00:02<00:00, 5007.00it/s]
  Verification: 100%|███████████████████████████████████████████████████████████████████████████████████████████████| 81/81 [00:38<00:00,  2.12it/s]
  |                                      |    1e-06 |    1e-05 |   0.0001 |    0.001 |     0.01 |      0.1 |
  |:-------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|
  | MS1MV2-ResNet100-Arcface_IJBB_N1D1F1 | 0.408861 | 0.899513 | 0.946349 | 0.964167 | 0.976144 | 0.98666  |
  | MS1MV2-ResNet100-Arcface_IJBB_N1D1F0 | 0.389192 | 0.898442 | 0.94557  | 0.96261  | 0.975268 | 0.986076 |
  | MS1MV2-ResNet100-Arcface_IJBB_N1D0F1 | 0.402142 | 0.893184 | 0.943622 | 0.963096 | 0.975755 | 0.986173 |
  | MS1MV2-ResNet100-Arcface_IJBB_N1D0F0 | 0.382765 | 0.893281 | 0.942454 | 0.961538 | 0.975073 | 0.985589 |
  | MS1MV2-ResNet100-Arcface_IJBB_N0D1F1 | 0.42814  | 0.908179 | 0.948978 | 0.964654 | 0.976728 | 0.986563 |
  | MS1MV2-ResNet100-Arcface_IJBB_N0D1F0 | 0.392989 | 0.903895 | 0.947614 | 0.962999 | 0.975755 | 0.986076 |
  | MS1MV2-ResNet100-Arcface_IJBB_N0D0F1 | 0.425998 | 0.907011 | 0.947809 | 0.96446  | 0.976436 | 0.986563 |
  | MS1MV2-ResNet100-Arcface_IJBB_N0D0F0 | 0.389971 | 0.904187 | 0.946738 | 0.962025 | 0.976144 | 0.985979 |

  real    23m36.871s
  user    68m56.171s
  sys     138m6.504s
  ```
***
