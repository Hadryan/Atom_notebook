# ___2019 - 09 - 19 Model Optimization___

# 链接
  - [TensorFlow model optimization](https://www.tensorflow.org/model_optimization/guide?hl=zh_cn)
  - [针对常见移动和边缘用例的优化模型](https://www.tensorflow.org/lite/models?hl=zh_cn)
  - [论文 - Learning both Weights and Connections for Efficient Neural Networks](https://xmfbit.github.io/2018/03/14/paper-network-prune-hansong/)
  - [Github Network Slimming (Pytorch)](https://github.com/Eric-mingjie/network-slimming)
  - [Distiller Documentation](https://nervanasystems.github.io/distiller/index.html)
  - [NPU使用示例](http://ai.nationalchip.com/docs/gx8010/npukai-fa-zhi-nan/shi-li.html)
  - [Github pruning_with_keras](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/g3doc/guide/pruning/pruning_with_keras.ipynb)
  - [](https://www.tensorflow.org/lite/models/segmentation/overview)
  - [优化机器学习模型](https://www.tensorflow.org/model_optimization)
  - [TensorFlow C++ and Python Image Recognition Demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image)
  - [Issue with converting Keras .h5 files to .tflite files](https://github.com/tensorflow/tensorflow/issues/20878)
  - [TensorFlow Lite converter](https://www.tensorflow.org/lite/convert)
  - [Github Tencent/ncnn](https://github.com/Tencent/ncnn)

过度参数化（over-parameterization）是深度神经网络的一个普遍属性，这会导致高计算成本和高内存占用。作为一种补救措施，网络剪枝（network pruning）已被证实是一种有效的改进技术，可以在计算预算有限的情况下提高深度网络的效率。

网络剪枝的过程一般包括三个阶段：1）训练一个大型，过度参数化的模型，2）根据特定标准修剪训练好的大模型，以及3）微调（fine-tune）剪枝后的模型以重新获得丢失的性能。

使用剪枝的方法，将模型中不重要的权重设置为0，将原来的dense model转变为sparse model，达到压缩的目的

而本文的作者check了六种SOA的工作，发现：在剪枝算法得到的模型上进行finetune，只比相同结构，但是使用random初始化权重的网络performance好了一点点，甚至有的时候还不如。作者的结论是：
  训练一个over parameter的model对最终得到一个efficient的小模型不是必要的
  为了得到剪枝后的小模型，求取大模型中的important参数其实并不打紧
  剪枝得到的结构，相比求得的weight，更重要。所以不如将剪枝算法看做是网络结构搜索的一种特例。
作者立了两个论点来打：
  要先训练一个over-parameter的大模型，然后在其基础上剪枝。因为大模型有更强大的表达能力。
  剪枝之后的网络结构和权重都很重要，是剪枝模型finetune的基础。


***

# 模型量化 Post-training quantization
## 量化 Optimization techniques
  - 模型优化常用方法
    - **剪枝 pruning** 减少模型参数数量，简化模型
    - **量化 quantization** 降低表示精度
    - 将原始模型拓扑更新为参数更少 / 执行更快的结构，如张量分解与净化 tensor decomposition methods and distillation
  - **量化 Quantization** 将模型参数替换为低精度的表示，如使用 8-bit 整数替换 32-bit 浮点数，对于提升特定硬件的执行效率，低精度是必须的
  - **稀疏与剪枝 Sparsity and pruning** 将某些张量置零，即将层间的连接剪枝，使模型更稀疏化
## 模型的训练后量化 post training quantization
  - **权重量化 Quantizing weights** 在训练好的模型上经过量化，可以减少 CPU 以及硬件的传输 / 计算 / 资源占用以及模型大小，但会略微牺牲准确度，量化可以应用在 float 类型的模型上，在 tflite 转化期间，将参数类型转化为 int
  - 一般可以再 GPU 运算时使用 16-bit float，CPU 计算时使用 8-bit int
  - **模型量化示例** 前向过程的大部分计算使用 int 替换 float
    ```py
    import tensorflow as tf
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()    
    ```
  - **权重和激活的全整数量化 Full integer quantization of weights and activations** 需要提供一个小的表示数据集 representative data set
    ```py
    import tensorflow as tf

    def representative_dataset_gen():
      for _ in range(num_calibration_steps):
        # Get sample input data as a numpy array in a method of your choosing.
        yield [input]

    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    tflite_quant_model = converter.convert()    
    ```
    模型的输入输出依然可以是浮点值
## MNIST 示例
  - 训练 keras MNIST 模型[Keras MNIST](https://github.com/leondgarse/Atom_notebook/blob/master/public/2018/09-06_tensorflow_tutotials.md#keras-mnist)
  - **模型保存**
    ```py
    tf.saved_model.save(model, './models')
    !tree models/
    # models/
    # ├── assets
    # ├── saved_model.pb
    # └── variables
    #     ├── variables.data-00000-of-00002
    #     ├── variables.data-00001-of-00002
    #     └── variables.index

    !du -hd0 models/
    # 4.8M	models/
    ```
  - **模型转化**
    ```py
    import tensorflow as tf
    # Convert to a tflite file
    converter = tf.lite.TFLiteConverter.from_saved_model('./models/')
    tflite_model = converter.convert()
    open("converted_model.tflite", "wb").write(tflite_model)

    # To quantize the model on export, set the optimizations flag to optimize for size
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    tflite_quant_model = converter.convert()
    open("converted_model_quant.tflite", "wb").write(tflite_quant_model)
    ```
    模型大小大约可以压缩为 1 / 4
    ```py
    !ls -lh converted_model.tflite converted_model_quant.tflite
    # -rw-r--r-- 1 leondgarse leondgarse 401K 十月 28 16:27 converted_model_quant.tflite
    # -rw-r--r-- 1 leondgarse leondgarse 1.6M 十月 28 16:26 converted_model.tflite
    ```
  - **模型加载运行** 使用的数据 batch 必须是 1
    ```py
    # Run the TFLite models
    interpreter = tf.lite.Interpreter(model_path='./converted_model_quant.tflite')
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]
    interpreter.set_tensor(input_index, tf.convert_to_tensor(np.ones([1, 28, 28], dtype='float32')))
    interpreter.invoke()
    pp = interpreter.get_tensor(output_index)

    # x_test
    interpreter.set_tensor(input_index, tf.convert_to_tensor(x_test[:1], dtype='float32'))
    interpreter.invoke()
    pp = interpreter.get_tensor(output_index)
    print(pp.argmax(1), y_test[:1])
    # [7] [7]

    # Dataset
    images, labels = tf.cast(x_test, tf.float32), y_test
    mnist_ds = tf.data.Dataset.from_tensor_slices((images, labels)).batch(1).shuffle(1)
    for img, label in mnist_ds.take(1):
        break

    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_index)
    print(predictions.argmax(1), label.numpy())
    # [7] [7]

    plt.imshow(img[0])
    template = "True:{true}, predicted:{predict}"
    plt.title(template.format(true=str(label[0].numpy()), predict=str(predictions.argmax())))
    ```
  - **模型验证 Evaluate**
    ```py
    def eval_model(interpreter, mnist_ds):
        total_seen = 0
        num_correct = 0

        for img, label in mnist_ds:
            total_seen += 1
            interpreter.set_tensor(input_index, img)
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_index)
            if predictions.argmax(1) == label.numpy():
                num_correct += 1

            if total_seen % 500 == 0:
                print("Accuracy after %i images: %f" % (total_seen, float(num_correct) / float(total_seen)))

        return float(num_correct) / float(total_seen)
    print(eval_model(interpreter, mnist_ds))
    # 0.9794
    interpreter_no_quant = tf.lite.Interpreter(model_path='./converted_model.tflite')
    interpreter_no_quant.allocate_tensors()
    print(eval_model(interpreter_no_quant, mnist_ds))
    # 0.9797
    ```
## Optimizing an existing model
  - resnet-v2 是带有预激活层 pre-activation layers 的 resnet，可以将训练好的 frozen graph resnet-v2-101 量化为 tflite 的 flatbuffer 格式
    ```py
    archive_path = tf.keras.utils.get_file("resnet_v2_101.tgz", "https://storage.googleapis.com/download.tensorflow.org/models/tflite_11_05_08/resnet_v2_101.tgz", extract=True)
    archive_path = pathlib.Path(archive_path)
    archive_dir = str(archive_path.parent)
    print(archive_path, archive_dir)                                                                
    # /home/leondgarse/.keras/datasets/resnet_v2_101.tgz /home/leondgarse/.keras/datasets

    !ls -lh {archive_dir}/resnet_*                                                                  
    # -rwxrwxrwx 1 root root 512M 四月 15  2017 /home/leondgarse/.keras/datasets/resnet_v2_101_299.ckpt
    # -rwxrwxrwx 1 root root 2.0M 九月  6  2018 /home/leondgarse/.keras/datasets/resnet_v2_101_299_eval.pbtxt
    # -rwxrwxrwx 1 root root 171M 九月  6  2018 /home/leondgarse/.keras/datasets/resnet_v2_101_299_frozen.pb
    # -rwxrwxrwx 1 root root   49 九月  6  2018 /home/leondgarse/.keras/datasets/resnet_v2_101_299_info.txt
    # -rwxrwxrwx 1 root root 171M 九月  6  2018 /home/leondgarse/.keras/datasets/resnet_v2_101_299.tflite
    # -rwxrwxrwx 1 root root 794M 十月 14 14:45 /home/leondgarse/.keras/datasets/resnet_v2_101.tgz
    ```
    `info.txt` 文件包含输入输出的名称，也可以使用 TensorBoard 可视化查看
    ```py
    ! cat {archive_dir}/resnet_v2_101_299_info.txt
    # Model: resnet_v2_101
    # Input: input
    # Output: output
    ```
  - **模型转化** 使用 `tf.compat.v1.lite.TFLiteConverter.from_frozen_graph` [ ??? ]
    ```py
    graph_def_file = pathlib.Path(archive_path).parent/"resnet_v2_101_299_frozen.pb"
    input_arrays = ["input"]
    output_arrays = ["output"]
    converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
      str(graph_def_file), input_arrays, output_arrays, input_shapes={"input":[1,299,299,3]})
    converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
    resnet_tflite_file = graph_def_file.parent/"resnet_v2_101_quantized.tflite"
    resnet_tflite_file.write_bytes(converter.convert())

    !ls -lh {str(resnet_tflite_file)}                                                               
    # -rwxrwxrwx 1 root root 43M 十月 28 17:52 /home/leondgarse/.keras/datasets/resnet_v2_101_quantized.tflite
    ```
***

# Post-training integer quantization
  The optimized model top-1 accuracy is 76.8, the same as the floating point model.

  Now, in order to create quantized values with an accurate dynamic range of activations, you need to provide a representative dataset:

  ```py
  mnist_train, _ = tf.keras.datasets.mnist.load_data()
  images = tf.cast(mnist_train[0], tf.float32)/255.0
  mnist_ds = tf.data.Dataset.from_tensor_slices((images)).batch(1)
  def representative_data_gen():
    for input_value in mnist_ds.take(100):
      yield [input_value]

  converter.representative_dataset = representative_data_gen
  ```
  Finally, convert the model to TensorFlow Lite format:
  ```py
  tflite_model_quant = converter.convert()
  tflite_model_quant_file = tflite_models_dir/"mnist_model_quant.tflite"
  tflite_model_quant_file.write_bytes(tflite_model_quant)
  ```
  Note how the resulting file is approximately 1/4 the size:

  ```py
  !ls -lh {tflite_models_dir}
  ```
  Your model should now be fully quantized. However, if you convert a model that includes any operations that TensorFlow Lite cannot quantize, those ops are left in floating point. This allows for conversion to complete so you have a smaller and more efficient model, but the model won't be compatible with some ML accelerators that require full integer quantization. Also, by default, the converted model still use float input and outputs, which also is not compatible with some accelerators.

  So to ensure that the converted model is fully quantized (make the converter throw an error if it encounters an operation it cannot quantize), and to use integers for the model's input and output, you need to convert the model again using these additional configurations:

  ```py
  converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
  converter.inference_input_type = tf.uint8
  converter.inference_output_type = tf.uint8

  tflite_model_quant = converter.convert()
  tflite_model_quant_file = tflite_models_dir/"mnist_model_quant_io.tflite"
  tflite_model_quant_file.write_bytes(tflite_model_quant)
  ```
  In this example, the resulting model size remains the same because all operations successfully quantized to begin with. However, this new model now uses quantized input and output, making it compatible with more accelerators, such as the Coral Edge TPU.

  In the following sections, notice that we are now handling two TensorFlow Lite models: tflite_model_file is the converted model that still uses floating-point parameters, and tflite_model_quant_file is the same model converted with full integer quantization, including uint8 input and output.
## Post-training float16 quantization
  To instead quantize the model to float16 on export, first set the optimizations flag to use default optimizations. Then specify that float16 is the supported type on the target platform:
  ```py
  tf.logging.set_verbosity(tf.logging.INFO)
  converter.optimizations = [tf.lite.Optimize.DEFAULT]
  converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
  ```
  Finally, convert the model like usual. Note, by default the converted model will still use float input and outputs for invocation convenience.
  ```py
  tflite_fp16_model = converter.convert()
  tflite_model_fp16_file = tflite_models_dir/"mnist_model_quant_f16.tflite"
  tflite_model_fp16_file.write_bytes(tflite_fp16_model)
  ```
  Note how the resulting file is approximately 1/2 the size.
  ```py
  !ls -lh {tflite_models_dir}
  ```
***

# Magnitude-based weight pruning with Keras
## Train a MNIST model without pruning
  ```py
  %load_ext tensorboard
  import tensorboard
  import tensorflow as tf
  import tempfile
  import zipfile
  import os
  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)

  (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
  x_train = np.expand_dims(x_train, -1).astype('float32') / 255
  x_test = np.expand_dims(x_test, -1).astype('float32') / 255
  y_train = tf.keras.utils.to_categorical(y_train, 10)
  y_test = tf.keras.utils.to_categorical(y_test, 10)
  print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
  # (60000, 28, 28, 1) (60000, 10) (10000, 28, 28, 1) (10000, 10)

  l = tf.keras.layers
  model = tf.keras.Sequential([
      l.Conv2D(32, 5, padding='same', activation='relu', input_shape=(28, 28, 1)),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.BatchNormalization(),
      l.Conv2D(64, 5, padding='same', activation='relu'),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      l.Dense(1024, activation='relu'),
      l.Dropout(0.4),
      l.Dense(10, activation='softmax')
  ])
  model.summary()
  # Model: "sequential"
  # _________________________________________________________________
  # Layer (type)                 Output Shape              Param #   
  # =================================================================
  # conv2d (Conv2D)              (None, 28, 28, 32)        832       
  # _________________________________________________________________
  # max_pooling2d (MaxPooling2D) (None, 14, 14, 32)        0         
  # _________________________________________________________________
  # batch_normalization (BatchNo (None, 14, 14, 32)        128       
  # _________________________________________________________________
  # conv2d_1 (Conv2D)            (None, 14, 14, 64)        51264     
  # _________________________________________________________________
  # max_pooling2d_1 (MaxPooling2 (None, 7, 7, 64)          0         
  # _________________________________________________________________
  # flatten (Flatten)            (None, 3136)              0         
  # _________________________________________________________________
  # dense (Dense)                (None, 1024)              3212288   
  # _________________________________________________________________
  # dropout (Dropout)            (None, 1024)              0         
  # _________________________________________________________________
  # dense_1 (Dense)              (None, 10)                10250     
  # =================================================================
  # Total params: 3,274,762
  # Trainable params: 3,274,698
  # Non-trainable params: 64
  # _________________________________________________________________

  logdir = tempfile.mkdtemp()
  print('Writing training logs to ' + logdir)
  # Writing training logs to /tmp/tmp03ewcd_c
  %tensorboard --logdir={logdir}

  batch_size = 32
  callbacks = [tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=0)]
  model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
  model.fit(x_train, y_train, batch_size=batch_size, epochs=10, verbose=1, callbacks=callbacks, validation_data=(x_test, y_test))

  score = model.evaluate(x_test, y_test, verbose=0)
  print('Test loss: %f, Test accuracy: %f' % (score[0], score[1]))
  # Test loss: 0.058006, Test accuracy: 0.990000

  tempfile.mkstemp('.h5')
  _, keras_file = tempfile.mkstemp('.h5')
  print('Saving model to: ', keras_file)
  # Saving model to:  /tmp/tmpw_728hos.h5
  tf.keras.models.save_model(model, keras_file, include_optimizer=False)
  !ls -lh {keras_file}
  # -rw------- 1 leondgarse leondgarse 13M 十月 15 10:59 /tmp/tmpw_728hos.h5
  ```
## Train a pruned MNIST
  We provide a prune_low_magnitude() API to train models with removed connections. The Keras-based API can be applied at the level of individual layers, or the entire model. We will show you the usage of both in the following sections.

  At a high level, the technique works by iteratively removing (i.e. zeroing out) connections between layers, given an schedule and a target sparsity.

  For example, a typical configuration will target a 75% sparsity, by pruning connections every 100 steps (aka epochs), starting from step 2,000. For more details on the possible configurations, please refer to the github documentation.

  Build a pruned model layer by layer￼
  In this example, we show how to use the API at the level of layers, and build a pruned MNIST solver model.

  In this case, the prune_low_magnitude() receives as parameter the Keras layer whose weights we want pruned.

  This function requires a pruning params which configures the pruning algorithm during training. Please refer to our github page for detailed documentation. The parameter used here means:

  Sparsity. PolynomialDecay is used across the whole training process. We start at the sparsity level 50% and gradually train the model to reach 90% sparsity. X% sparsity means that X% of the weight tensor is going to be pruned away.
  Schedule. Connections are pruned starting from step 2000 to the end of training, and runs every 100 steps. The reasoning behind this is that we want to train the model without pruning for a few epochs to reach a certain accuracy, to aid convergence. Furthermore, we give the model some time to recover after each pruning step, so pruning does not happen on every step. We set the pruning frequency to 100.

  To demonstrate how to save and restore a pruned keras model, in the following example we first train the model for 10 epochs, save it to disk, and finally restore and continue training for 2 epochs. With gradual sparsity, four important parameters are begin_sparsity, final_sparsity, begin_step and end_step. The first three are straight forward. Let's calculate the end step given the number of train example, batch size, and the total epochs to train.
  ```py
  from tensorflow_model_optimization.sparsity import keras as sparsity
  np.ceil(x_train.shape[0] / 32).astype(np.int32) * 12
  pruning_params = {
      "pruning_schedule": sparsity.PolynomialDecay(initial_sparsity=0.5, final_sparsity=0.9, begin_step=2000, end_step=22500, frequency=100)
  }

  pruned_model = tf.keras.Sequential([
      sparsity.prune_low_magnitude(l.Conv2D(32, 5, padding='same', activation='relu'), input_shape=(28, 28, 1), **pruning_params),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.BatchNormalization(),
      sparsity.prune_low_magnitude(l.Conv2D(64, 5, padding='same', activation='relu'), **pruning_params),
      l.MaxPooling2D((2, 2), (2, 2), padding='same'),
      l.Flatten(),
      sparsity.prune_low_magnitude(l.Dense(1024, activation='relu'), **pruning_params),
      l.Dropout(0.4),
      sparsity.prune_low_magnitude(l.Dense(10, activation='softmax'), **pruning_params)
  ])
  pruned_model.summary()
  logdir = tempfile.mkdtemp()
  print('Writing training logs to ' + logdir)
  %tensorboard --logdir={logdir}
  pruned_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
  callbacks = [
      sparsity.UpdatePruningStep(),
      sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
  ]
  pruned_model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, callbacks=callbacks, validation_data=(x_test, y_test))
  ```
  ```py
  _, checkpoint_file = tempfile.mkstemp('.h5')
  print('Saving pruned model to: ', checkpoint_file)
  tf.keras.models.save_model(pruned_model, checkpoint_file, include_optimizer=True)
  !ls -lh {checkpoint_file}
  with sparsity.prune_scope():
      restored_model = tf.keras.models.load_model(checkpoint_file)
  restored_model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=1, callbacks=callbacks, validation_data=(x_test, y_test))
  pruned_model.summary()
  final_model = sparsity.strip_pruning(pruned_model)
  final_model.summary()
  _, pruned_keras_file = tempfile.mkstemp('.h5')
  print('Saving pruned model to: ', pruned_keras_file)

  # No need to save the optimizer with the graph for serving.
  tf.keras.models.save_model(final_model, pruned_keras_file, include_optimizer=False)
  !ls -lh {pruned_keras_file}
  _, zip1 = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zip1, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)
  print("Size of the unpruned model before compression: %.2f Mb" %
        (os.path.getsize(keras_file) / float(2**20)))
  print("Size of the unpruned model after compression: %.2f Mb" %
        (os.path.getsize(zip1) / float(2**20)))

  _, zip2 = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zip2, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(pruned_keras_file)
  print("Size of the pruned model before compression: %.2f Mb" %
        (os.path.getsize(pruned_keras_file) / float(2**20)))
  print("Size of the pruned model after compression: %.2f Mb" %
        (os.path.getsize(zip2) / float(2**20)))
  ```
## Prune a whole model
  The prune_low_magnitude function can also be applied to the entire Keras model.

  In this case, the algorithm will be applied to all layers that are ameanable to weight pruning (that the API knows about). Layers that the API knows are not ameanable to weight pruning will be ignored, and unknown layers to the API will cause an error.

  If your model has layers that the API does not know how to prune their weights, but are perfectly fine to leave "un-pruned", then just apply the API in a per-layer basis.

  Regarding pruning configuration, the same settings apply to all prunable layers in the model.

  Also noteworthy is that pruning doesn't preserve the optimizer associated with the original model. As a result, it is necessary to re-compile the pruned model with a new optimizer.

  Before we move forward with the example, lets address the common use case where you may already have a serialized pre-trained Keras model, which you would like to apply weight pruning on. We will take the original MNIST model trained previously to show how this works. In this case, you start by loading the model into memory like this:
  ```py
  loaded_model = tf.keras.models.load_model(keras_file)
  batch_size = 32
  end_step = np.ceil(x_train.shape[0] / batch_size).astype(np.int32) * 4
  new_pruning_params = {
      "pruning_schedule": sparsity.PolynomialDecay(
          initial_sparsity=0.5,
          final_sparsity=0.9,
          begin_step=0,
          end_step=end_step,
          frequency=100
      )
  }
  new_pruned_model = sparsity.prune_low_magnitude(loaded_model, **new_pruning_params)
  new_pruned_model.summary()
  new_pruned_model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

  logdir = tempfile.mkdtemp()
  print('Writing training logs to ' + logdir)
  %tensorboard --logdir={logdir}

  callbacks = [
      sparsity.UpdatePruningStep(),
      sparsity.PruningSummaries(log_dir=logdir, profile_batch=0)
  ]
  new_pruned_model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=4,
            verbose=1,
            callbacks=callbacks,
            validation_data=(x_test, y_test))
  score = new_pruned_model.evaluate(x_test, y_test, verbose=0)
  print('Test loss: %f, Test accuracy: %f' % (score[0], score[1]))

  final_model = sparsity.strip_pruning(pruned_model)
  final_model.summary()

  _, new_pruned_keras_file = tempfile.mkstemp('.h5')
  print('Saving pruned model to: ', new_pruned_keras_file)
  tf.keras.models.save_model(final_model, new_pruned_keras_file, include_optimizer=False)

  _, zip3 = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zip3, 'w', compression=zipfile.ZIP_DEFLATED) as f:
      f.write(new_pruned_keras_file)
  print("Size of the pruned model before compression: %.2f Mb"
        % (os.path.getsize(new_pruned_keras_file) / float(2**20)))
  print("Size of the pruned model after compression: %.2f Mb"
        % (os.path.getsize(zip3) / float(2**20)))
  ```
## Convert to TensorFlow Lite
  ```py
  converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
  tflite_model = converter.convert()
  tflite_model_file = '/tmp/sparse_mnist.tflite'
  with open(tflite_model_file, 'wb') as f:
      f.write(tflite_model)
  ! ls -lh {tflite_model_file}

  _, zip_tflite = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zip_tflite, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(tflite_model_file)
  print("Size of the tflite model before compression: %.2f Mb"
        % (os.path.getsize(tflite_model_file) / float(2**20)))
  print("Size of the tflite model after compression: %.2f Mb"
        % (os.path.getsize(zip_tflite) / float(2**20)))

  interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
  interpreter.allocate_tensors()
  interpreter.get_input_details()[0]
  input_index = interpreter.get_input_details()[0]['index']
  output_index = interpreter.get_output_details[0]['index']
  output_index = interpreter.get_output_details()[0]['index']

  def eval_model(interpreter, x_test, y_test):
      total_seen = 0
      num_correct = 0

      for img, label in zip(x_test, y_test):
          inp = img.reshape((1, 28, 28, 1))
          total_seen += 1
          interpreter.set_tensor(input_index, inp)
          interpreter.invoke()
          predictions = interpreter.get_tensor(output_index)
          if np.argmax(predictions) == np.argmax(label):
            num_correct += 1

          if total_seen % 1000 == 0:
              print("Accuracy after %i images: %f" % (total_seen, float(num_correct) / float(total_seen)))

      return float(num_correct) / float(total_seen)

  print(eval_model(interpreter, x_test, y_test))
  ```
  ```py
  converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
  converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

  tflite_quant_model = converter.convert()

  tflite_quant_model_file = '/tmp/sparse_mnist_quant.tflite'
  with open(tflite_quant_model_file, 'wb') as f:
      f.write(tflite_quant_model)

  _, zip_tflite = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zip_tflite, 'w', compression=zipfile.ZIP_DEFLATED) as f:
      f.write(tflite_quant_model_file)
  print("Size of the tflite model before compression: %.2f Mb"
        % (os.path.getsize(tflite_quant_model_file) / float(2**20)))
  print("Size of the tflite model after compression: %.2f Mb"
        % (os.path.getsize(zip_tflite) / float(2**20)))

  interpreter = tf.lite.Interpreter(model_path=str(tflite_quant_model_file))
  interpreter.allocate_tensors()
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  print(eval_model(interpreter, x_test, y_test))
  ```
***

# Train sparse TensorFlow models with Keras
Yes, the current TensorFlow Lite op kernels are optimized for ARM processor (using NEON instruction set). If SSE is available, it will try to use NEON_2_SSE to adapt NEON calls to SSE, so it should be still running with some sort of SIMD. However we didn't put much effort to optimize this code path.

Regarding number of threads. There is a SetNumThreads function in C++ API, but it's not exposed in Python API (yet). When it's not set, the underlying implementation may try to probe number of available cores. If you build the code by yourself, you can try to change the value and see if it affects the result.

Hope these helps.
```sh
toco
--input_file=mobilenet_v1_1.0_224/teste/sfrozen_inference_graph.pb
--input_format=TENSORFLOW_GRAPHDEF
--output_file=/tmp/mobilenet_v1_1.0_224.tflite
--input_shape=-1,-1,-1,3
--input_array=image_tensor
--output_array=detection_boxes,detection_scores,detection_classes,detection_nums \

bazel run -c opt tensorflow/contrib/lite/toco:toco --
--input_file=$OUTPUT_DIR/tflite_graph.pb
--output_file=$OUTPUT_DIR/detect.tflite
--input_shapes=1,300,300,3
--input_arrays=normalized_input_image_tensor
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'
--inference_type=QUANTIZED_UINT8
--mean_values=128
--std_values=128
--change_concat_input_ranges=false
--allow_custom_ops
```
```sh
toco --graph_def_file=mtcnn.pb --output_file=mtcnn.tflite --input_shape=-1,-1,3:None:3:None --input_array=input,min_size,thresholds,factor --output_array=prob,landmarks,box
```
```py
model_path = './mtcnn.pb'
graph = tf.Graph()
with graph.as_default():
  with open(model_path, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef.FromString(f.read())
      tf.import_graph_def(graph_def, name='')
config = tf.compat.v1.ConfigProto(
  gpu_options = tf.compat.v1.GPUOptions(allow_growth=True),
  allow_soft_placement=True,
  intra_op_parallelism_threads=4,
  inter_op_parallelism_threads=4)
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(graph=graph, config=config)

feeds = {
  'input': graph.get_tensor_by_name('input:0'),
  'min_size': graph.get_tensor_by_name('min_size:0'),
  'thresholds': graph.get_tensor_by_name('thresholds:0'),
  'factor': graph.get_tensor_by_name('factor:0'),
}
fetches = {
  'prob': graph.get_tensor_by_name('prob:0'),
  'landmarks': graph.get_tensor_by_name('landmarks:0'),
  'box': graph.get_tensor_by_name('box:0'),
}
tf.compat.v1.saved_model.simple_save(sess,
  "./1",
  inputs=feeds,
  outputs=fetches)
```
```py
import tensorflow as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = './saved'
builder = tf.compat.v1.saved_model.builder.SavedModelBuilder(export_dir)

model_path = './mtcnn.pb'
graph = tf.Graph()
with graph.as_default():
  with open(model_path, 'rb') as f:
      graph_def = tf.compat.v1.GraphDef.FromString(f.read())

sigs = {}

with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    graph = tf.compat.v1.get_default_graph()
    feeds = {
      'input': graph.get_tensor_by_name('input:0'),
      'thresholds': graph.get_tensor_by_name('thresholds:0'),
    }
    fetches = {
      'prob': graph.get_tensor_by_name('prob:0'),
      'landmarks': graph.get_tensor_by_name('landmarks:0'),
      'box': graph.get_tensor_by_name('box:0'),
    }

    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.compat.v1.saved_model.signature_def_utils.predict_signature_def(
            feeds, fetches)

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()
```
