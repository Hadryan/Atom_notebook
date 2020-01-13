# ___2018 - 10 - 22 Tensorflow Examples___
***

## How to Retrain an Image Classifier for New Categories
  - [How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)
  - https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/
  ```py
  import tensorflow_hub as hub

  hub_module = 'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'
  module = hub.Module(hub_module)
  ```
  **测试**
  ```py
  height, width = hub.get_expected_image_size(module)

  image_file = './datasets/flower_photos/daisy/100080576_f52e8ee070_n.jpg'
  images = tf.gfile.FastGFile(image_file, 'rb').read()
  images = tf.image.decode_jpeg(images)

  sess = tf.InteractiveSession()
  images.eval().shape

  imm = tf.image.resize_images(images, (height, width))
  imm = tf.expand_dims(imm, 0)  # A batch of images with shape [batch_size, height, width, 3].
  plt.imshow(imm[0].eval().astype('int'))

  tf.global_variables_initializer().run()
  features = module(imm).eval()  # Features with shape [batch_size, num_features].
  print(features.shape)
  # (1, 2048)
  ```
  ```py
  def jpeg_decoder_layer(module_spec):
      height, width = hub.get_expected_image_size(module_spec)
      input_depth = hub.get_num_image_channels(module_spec)
      jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
      imm = tf.image.decode_jpeg(jpeg_data, channels=input_depth)

      imm = tf.image.convert_image_dtype(imm, dtype=tf.float32)
      imm = tf.expand_dims(imm, 0)
      imm = tf.image.resize_images(images, (height, width))

      return jpeg_data, imm
  ```
  **测试**
  ```py
  jj, ii = jpeg_decoder_layer(module)
  tt = sess.run(ii, {jj: tf.gfile.FastGFile(image_file, 'rb').read()})
  print(tt.shape)
  # (299, 299, 3)
  ```
  ```py
  CLASS_COUNT = 5
  def add_classifier_op(class_count, bottleneck_module, is_training, learning_rate=0.01):
      height, width = hub.get_expected_image_size(bottleneck_module)
      resized_input_tensor = tf.placeholder(tf.float32, [None, height, width, 3])
      bottleneck_tensor = bottleneck_module(resized_input_tensor)
      batch_size, bottleneck_out = bottleneck_tensor.get_shape().as_list()  # None, 2048

      # Add a fully connected layer and a softmax layer
      with tf.name_scope('input'):
          bottleneck_input = tf.placeholder_with_default(bottleneck_tensor, shape=[batch_size, bottleneck_out], name='BottleneckInputPlaceholder')
          target_label = tf.placeholder(tf.int64, [batch_size], name='GroundTruthInput')

      with tf.name_scope('final_retrain_ops'):
          with tf.name_scope('weights'):
              init_value = tf.truncated_normal([bottleneck_out, class_count], stddev=0.001)
              weights = tf.Variable(init_value, name='final_weights')
          with tf.name_scope('biases'):
              biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
          with tf.name_scope('dense'):
              logits = tf.matmul(bottleneck_input, weights) + biases

      final_tensor = tf.nn.softmax(logits, name='final_result')

      # The tf.contrib.quantize functions rewrite the graph in place for
      # quantization. The imported model graph has already been rewritten, so upon
      # calling these rewrites, only the newly added final layer will be
      # transformed.
      if is_training:
          tf.contrib.quantize.create_training_graph()
      else:
          tf.contrib.quantize.create_eval_graph()

      # If this is an eval graph, we don't need to add loss ops or an optimizer.
      if not is_training:
          return None, None, bottleneck_input, target_label, final_tensor

      with tf.name_scope('cross_entropy'):
          cross_entropy_mean = tf.losses.sparse_softmax_cross_entropy(labels=target_label, logits=logits)

      with tf.name_scope('train'):
          optimizer = tf.train.GradientDescentOptimizer(learning_rate)
          train_step = optimizer.minimize(cross_entropy_mean)

      return (train_step, cross_entropy_mean, bottleneck_input, target_label, final_tensor)
  ```
  ```py
  flower_url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
  train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(flower_url), origin=flower_url)

  def load_image_train_test(data_path, test_rate=10):
      rr = {}
      for sub_dir_name in tf.gfile.ListDirectory(data_path):
          sub_dir = os.path.join(data_path, sub_dir_name)
          print(sub_dir)
          if not tf.gfile.IsDirectory(sub_dir):
              continue

          item_num = len(tf.gfile.ListDirectory(sub_dir))

          train_dd = []
          test_dd = []
          for item_name in tf.gfile.ListDirectory(sub_dir):
              hash_name_hashed = hashlib.sha1(tf.compat.as_bytes(item_name)).hexdigest()
              percentage_hash = int(hash_name_hashed, 16) % (item_num + 1) * (100 / item_num)
              if percentage_hash < 10:
                  test_dd.append(os.path.join(sub_dir, item_name))
              else:
                  train_dd.append(os.path.join(sub_dir, item_name))
          rr[sub_dir_name] = {'train': train_dd, 'test': test_dd}

      return rr
  ```
## Advanced Convolutional Neural Networks
  - [Advanced Convolutional Neural Networks](https://www.tensorflow.org/tutorials/images/deep_cnn)
  - [tensorflow/models/tutorials/image/cifar10/](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10/)
  - [tensorflow/models/tutorials/image/cifar10_estimator/](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator)
***

# Sequences
## Recurrent Neural Networks
  - [Recurrent Neural Networks](https://www.tensorflow.org/tutorials/sequences/recurrent)
  - [tensorflow/models/tutorials/rnn/ptb/](https://github.com/tensorflow/models/tree/master/tutorials/rnn/ptb)
## Recurrent Neural Networks for Drawing Classification
  - [Recurrent Neural Networks for Drawing Classification](https://www.tensorflow.org/tutorials/sequences/recurrent_quickdraw)
  - [tensorflow/models/tutorials/rnn/quickdraw/](https://github.com/tensorflow/models/tree/master/tutorials/rnn/quickdraw)
## Simple Audio Recognition
  - [Simple Audio Recognition](https://www.tensorflow.org/tutorials/sequences/audio_recognition)
  - [tensorflow/tensorflow/examples/speech_commands/](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands)
## Neural Machine Translation seq2seq Tutorial
  - [tensorflow/nmt](https://github.com/tensorflow/nmt)
***

# 数据表示 data representation
## Vector Representations of Words
## Improving Linear Models Using Explicit Kernel Methods
## Large-scale Linear Models with TensorFlow
***

# Non ML
## Mandelbrot set
## Partial differential equations
***

# GOO
  - [TensorFlow Hub](https://www.tensorflow.org/hub/)
  - [基于字符的LSTM+CRF中文实体抽取](https://github.com/jakeywu/chinese_ner)
  - [Matplotlib tutorial](http://www.labri.fr/perso/nrougier/teaching/matplotlib/)
  - [TensorFlow 实战电影个性化推荐](https://blog.csdn.net/chengcheng1394/article/details/78820529)
  - [TensorRec](https://github.com/jfkirk/tensorrec)
  - [Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course/ml-intro)

  ![](images/opt1.gif)
***

- **先验概率 Prior probability** 是在获得某些信息或者依据前，对 p 的不确定性进行猜测，在结果发生前确定原因的概率分布
  ```py
  P(因) P(θ)
  ```
- **似然函数 likelihood function** 根据原因来估计结果的概率分布，似然是关于参数的函数，在参数给定的条件下，对于观察到的 x 的值的条件分布
  ```py
  P(果∣因) P(x∣θ)
  ```
- **后验概率 Posterior probability** 根据结果估计猜原因的概率分布，是在相关证据或者背景给定并纳入考虑之后的条件概率
  ```py
  P(因∣果) P(θ∣x)
  后验概率 P(θ∣x) = P(x∣θ) * P(θ) / P(x) = 似然估计 ∗ 先验概率 / evidence
  ```
***

# Dog Species Classifier
## Tensorflow 1.14
  ```py
  import tensorflow as tf
  from tensorflow import keras
  # config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5, allow_growth=True))
  config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
  sess = tf.Session(config=config)
  keras.backend.set_session(sess)

  from tensorflow.python.keras import layers
  from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
  from PIL import ImageFile
  ImageFile.LOAD_TRUNCATED_IMAGES = True

  train_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
      width_shift_range=0.2, height_shift_range=0.2, brightness_range=(0.1, 2),
      shear_range=0.15, horizontal_flip=True, fill_mode="nearest")

  train_img_gen = train_data_gen.flow_from_directory('./dogImages/train/', target_size=(512, 512), batch_size=4, seed=1)
  val_data_gen = ImageDataGenerator(rescale=1./255)
  val_img_gen = val_data_gen.flow_from_directory('./dogImages/valid/', target_size=(512, 512), seed=1)

  img_shape = (512, 512, 3)
  xx = keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet')
  # xx = keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  xx.trainable = True
  model = tf.keras.Sequential([
      layers.Input(shape=img_shape),
      xx,
      layers.Conv2D(512, 1, strides=1, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)),
      # layers.MaxPooling2D(2),
      layers.Dropout(0.5),
      # layers.AveragePooling2D(pool_size=512, strides=512, padding='same'),
      layers.GlobalAveragePooling2D(),
      layers.Flatten(),
      layers.Dense(133, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.00001)),        
  ])
  model.summary()

  callbacks = [
      keras.callbacks.TensorBoard(log_dir='./logs'),
      keras.callbacks.ModelCheckpoint("./keras_checkpoints", monitor='val_loss', save_best_only=True),
      keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
  ]
  model.compile(optimizer=keras.optimizers.Adadelta(0.1), loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit_generator(train_img_gen, validation_data=val_img_gen, epochs=50, callbacks=callbacks, verbose=1, workers=10)

  import glob2
  from skimage.io import imread
  from skimage.transform import resize

  model = tf.keras.models.load_model('keras_checkpoints')
  index_2_name = {vv: kk for kk, vv in train_img_gen.class_indices.items()}
  aa = resize(imread('./dogImages/1806687557.jpg'), (512, 512))
  pp = model.predict(np.expand_dims(aa, 0))
  print(index_2_name[pp.argmax()])
  # 029.Border_collie

  imm = glob2.glob('./dogImages/test/*/*')
  xx = np.array([resize(imread(ii), (512, 512)) for ii in imm])
  yy = np.array([int(os.path.basename(os.path.dirname(ii)).split('.')[0]) -1 for ii in imm])
  pp = model.predict(xx)
  tt = np.argmax(pp, 1)
  print((tt == yy).sum() / yy.shape[0])
  # 0.8588516746411483

  top_3_err = [(np.sort(ii)[-3:], ii.argmax(), imm[id]) for id, (ii, jj) in enumerate(zip(pp, yy)) if jj not in ii.argsort()[-3:]]
  print(1 - len(top_3_err) / yy.shape[0])
  # 0.965311004784689
  ```
## Tensorflow 2.0
  ```py
  import tensorflow as tf
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)
  # tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])

  from tensorflow import keras
  from tensorflow.python.keras import layers
  from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
  from PIL import ImageFile

  ImageFile.LOAD_TRUNCATED_IMAGES = True

  img_shape = (224, 224, 3)
  train_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.15,
      width_shift_range=0.2, height_shift_range=0.2, brightness_range=(0.1, 2),
      shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
  train_img_gen = train_data_gen.flow_from_directory('./dogImages/train/', target_size=img_shape[:2], batch_size=4, seed=1)
  val_data_gen = ImageDataGenerator(rescale=1./255)
  val_img_gen = val_data_gen.flow_from_directory('./dogImages/valid/', target_size=img_shape[:2], batch_size=4, seed=1)                           

  xx = keras.applications.ResNet50V2(include_top=False, weights='imagenet')

  xx.trainable = True
  model = tf.keras.Sequential([
      layers.Input(shape=img_shape),
      xx,
      layers.Conv2D(512, 1, strides=1, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)),
      # layers.MaxPooling2D(2),
      layers.Dropout(0.5),
      # layers.AveragePooling2D(pool_size=512, strides=512, padding='same'),
      layers.GlobalAveragePooling2D(),
      layers.Flatten(),
      layers.Dense(133, activation="softmax", kernel_regularizer=keras.regularizers.l2(0.00001)),                                                                                                                 
  ])
  model.summary()
  callbacks = [
      keras.callbacks.TensorBoard(log_dir='./logs'),
      keras.callbacks.ModelCheckpoint("./keras_checkpoints", monitor='val_loss', save_best_only=True),
      keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
  ]
  model.compile(optimizer=keras.optimizers.Adadelta(0.1), loss='categorical_crossentropy', metrics=['accuracy'])
  model.fit_generator(train_img_gen, validation_data=val_img_gen, epochs=50, callbacks=callbacks, verbose=1, workers=10)
  ```
  ```sh
  toco --saved_model_dir ./keras_checkpoints --output_file foo.tflite
  ```
  ```py
  from tensorflow_model_optimization.sparsity import keras as sparsity
  batch_size = 4
  end_step = np.ceil(train_img_gen.classes.shape[0] / batch_size).astype(np.int32) * 55
  pruning_params = {
      "pruning_schedule": sparsity.PolynomialDecay(
          initial_sparsity=0.5,
          final_sparsity=0.9,
          begin_step=2000,
          end_step=end_step,
          frequency=100)
  }

  pruned_model = tf.keras.Sequential([
      layers.Input(shape=img_shape),
      # sparsity.prune_low_magnitude(keras.applications.ResNet50V2(include_top=False, weights='imagenet'), **pruning_params),
      keras.applications.ResNet50V2(include_top=False, weights='imagenet'),
      sparsity.prune_low_magnitude(layers.Conv2D(512, 1, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.00001)), **pruning_params),
      layers.Dropout(0.5),
      layers.GlobalAveragePooling2D(),
      layers.Flatten(),
      sparsity.prune_low_magnitude(layers.Dense(133, activation='softmax', kernel_regularizer=keras.regularizers.l2(0.00001)), **pruning_params)
  ])
  ```
  ```py
  import glob2
  from skimage.io import imread
  from skimage.transform import resize
  imm = glob2.glob('./dogImages/test/*/*')
  xx = np.array([resize(imread(ii), (224, 224)) for ii in imm])
  ixx = tf.convert_to_tensor(xx, dtype='float32')
  # ixx = tf.convert_to_tensor(xx, dtype=tf.uint8)
  idd = tf.data.Dataset.from_tensor_slices((ixx)).batch(1)

  def representative_data_gen():
      for ii in idd.take(100):
          yield [ii]
  converter = tf.lite.TFLiteConverter.from_saved_model('./keras_checkpoints/')
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.representative_dataset = representative_data_gen
  tflite_quant_all_model = converter.convert()
  ```
***
