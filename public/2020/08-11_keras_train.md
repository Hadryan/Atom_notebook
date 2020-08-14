# Auto Tuner
## Keras Tuner
  - [Keras Tuner 简介](https://www.tensorflow.org/tutorials/keras/keras_tuner)
    ```py
    import tensorflow as tf
    from tensorflow import keras

    !pip install -q -U keras-tuner
    import kerastuner as kt

    (img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()
    # Normalize pixel values between 0 and 1
    img_train = img_train.astype('float32') / 255.0
    img_test = img_test.astype('float32') / 255.0

    def model_builder(hp):
        model = keras.Sequential()
        model.add(keras.layers.Flatten(input_shape=(28, 28)))

        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units = hp.Int('units', min_value = 32, max_value = 512, step = 32)
        model.add(keras.layers.Dense(units = hp_units, activation = 'relu'))
        model.add(keras.layers.Dense(10))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values = [1e-2, 1e-3, 1e-4])

        model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                      loss = keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                      metrics = ['accuracy'])

        return model

    tuner = kt.Hyperband(model_builder,
                         objective = 'val_accuracy',
                         max_epochs = 10,
                         factor = 3,
                         directory = 'my_dir',
                         project_name = 'intro_to_kt')

    tuner.search(img_train, label_train, epochs = 10, validation_data = (img_test, label_test), callbacks = [ClearTrainingOutput()])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

    print(f"""
    The hyperparameter search is complete. The optimal number of units in the first densely-connected
    layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
    is {best_hps.get('learning_rate')}.
    """)

    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)
    model.fit(img_train, label_train, epochs = 10, validation_data = (img_test, label_test))
    ```
  - **Tune on cifar10**
    ```py
    import tensorflow as tf
    from tensorflow import keras
    import matplotlib.pyplot as plt
    import kerastuner as kt
    import tensorflow_addons as tfa

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels_oh = tf.one_hot(tf.squeeze(train_labels), depth=10, dtype='uint8')
    test_labels_oh = tf.one_hot(tf.squeeze(test_labels), depth=10, dtype='uint8')
    print(train_images.shape, test_images.shape, train_labels_oh.shape, test_labels_oh.shape)

    def create_model(hp):
        hp_wd = hp.Choice("weight_decay", values=[0.0, 1e-5, 5e-5, 1e-4])
        hp_ls = hp.Choice("label_smoothing", values=[0.0, 0.1])
        hp_dropout = hp.Choice("dropout_rate", values=[0.0, 0.4])

        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dropout(rate=hp_dropout))
        model.add(keras.layers.Dense(10))

        model.compile(
            optimizer=tfa.optimizers.AdamW(weight_decay=hp_wd),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=hp_ls),
            metrics = ['accuracy'])

        return model

    tuner = kt.Hyperband(create_model,
                         objective='val_accuracy',
                         max_epochs=50,
                         factor=6,
                         directory='my_dir',
                         project_name='intro_to_kt')

    tuner.search(train_images, train_labels_oh, epochs=50, validation_data=(test_images, test_labels_oh))

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("best parameters: weight_decay = {}, label_smoothing = {}, dropout_rate = {}".format(best_hps.get('weight_decay'), best_hps.get('label_smoothing'), best_hps.get('dropout_rate')))

    # Build the model with the optimal hyperparameters and train it on the data
    model = tuner.hypermodel.build(best_hps)
    model.fit(train_images, train_labels_oh, epochs = 50, validation_data = (test_images, test_labels_oh))
    ```
## TensorBoard HParams
  - [Hyperparameter Tuning with the HParams Dashboard](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams)
    ```py
    # Load the TensorBoard notebook extension
    %load_ext tensorboard

    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp

    fashion_mnist = tf.keras.datasets.fashion_mnist

    (x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([16, 32]))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.2))
    HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
        hp.hparams_config(
            hparams=[HP_NUM_UNITS, HP_DROPOUT, HP_OPTIMIZER],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

    def train_test_model(hparams):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation=tf.nn.relu),
            tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax),
        ])
        model.compile(
            optimizer=hparams[HP_OPTIMIZER],
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
        )

        # model.fit(
        #   ...,
        #   callbacks=[
        #       tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        #       hp.KerasCallback(logdir, hparams),  # log hparams
        #   ],
        # )
        model.fit(x_train, y_train, epochs=1) # Run with 1 epoch to speed things up for demo purposes
        _, accuracy = model.evaluate(x_test, y_test)
        return accuracy

    def run(run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = train_test_model(hparams)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

    session_num = 0

    for num_units in HP_NUM_UNITS.domain.values:
        for dropout_rate in (HP_DROPOUT.domain.min_value, HP_DROPOUT.domain.max_value):
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_NUM_UNITS: num_units,
                    HP_DROPOUT: dropout_rate,
                    HP_OPTIMIZER: optimizer,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning/' + run_name, hparams)
                session_num += 1

    %tensorboard --logdir logs/hparam_tuning
    ```
  - **Tune on cifar10**
    ```py
    %load_ext tensorboard

    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp
    import tensorflow_addons as tfa

    (train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0
    train_labels_oh = tf.one_hot(tf.squeeze(train_labels), depth=10, dtype='uint8')
    test_labels_oh = tf.one_hot(tf.squeeze(test_labels), depth=10, dtype='uint8')
    print(train_images.shape, test_images.shape, train_labels_oh.shape, test_labels_oh.shape)

    HP_WD = hp.HParam("weight_decay", hp.Discrete([0.0, 1e-5, 5e-5, 1e-4]))
    HP_LS = hp.HParam("label_smoothing", hp.Discrete([0.0, 0.1]))
    HP_DR = hp.HParam("dropout_rate", hp.Discrete([0.0, 0.4]))
    METRIC_ACCURACY = 'accuracy'

    with tf.summary.create_file_writer('logs/hparam_tuning_cifar10').as_default():
        hp.hparams_config(
            hparams=[HP_WD, HP_LS, HP_DR],
            metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
        )

    def create_model(dropout=1):
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D((2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(64, activation='relu'))
        if dropout > 0 and dropout < 1:
            model.add(keras.layers.Dropout(dropout))
        model.add(keras.layers.Dense(10))
        return model

    def train_test_model(hparams, epochs=1):
        model = create_model(hparams[HP_DR])
        model.compile(
            optimizer=tfa.optimizers.AdamW(weight_decay=hparams[HP_WD]),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=hparams[HP_LS], from_logits=True),
            metrics=['accuracy'],
        )

        # model.fit(
        #   ...,
        #   callbacks=[
        #       tf.keras.callbacks.TensorBoard(logdir),  # log metrics
        #       hp.KerasCallback(logdir, hparams),  # log hparams
        #   ],
        # )
        hist = model.fit(train_images, train_labels_oh, epochs=epochs, validation_data=(test_images, test_labels_oh)) # Run with 1 epoch to speed things up for demo purposes
        return max(hist.history["accuracy"])

    def run(run_dir, hparams):
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            accuracy = train_test_model(hparams, epochs=20)
            tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

    session_num = 0
    for dr in HP_DR.domain.values:
        for label_smoothing in HP_LS.domain.values:
            for wd in HP_WD.domain.values:
                hparams = {
                    HP_WD: wd,
                    HP_LS: label_smoothing,
                    HP_DR: dr,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run('logs/hparam_tuning_cifar10/' + run_name, hparams)
                session_num += 1

    %tensorboard --logdir logs/hparam_tuning_cifar10
    ```
***

# TfLite
## TFLite Model Benchmark Tool
  - [TFLite Model Benchmark Tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
    ```sh
    cd ~/workspace/tensorflow.arm32
    ./configure
    bazel build -c opt --config=android_arm tensorflow/lite/tools/benchmark:benchmark_model

    adb push bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model /data/local/tmp
    adb shell chmod +x /data/local/tmp/benchmark_model

    cd ~/workspace/examples/lite/examples/image_classification/android/app/src/main/assets
    adb push mobilenet_v1_1.0_224_quant.tflite /data/local/tmp
    adb push mobilenet_v1_1.0_224.tflite /data/local/tmp
    adb push efficientnet-lite0-int8.tflite /data/local/tmp
    adb push efficientnet-lite0-fp32.tflite /data/local/tmp
    ```
  - **参数**
    - **--graph** 字符串，TFLite 模型路径
    - **--enable_op_profiling** true / false，是否测试每个步骤的执行时间: bool (default=false) Whether to enable per-operator profiling measurement.
    - **--nnum_threads** 整数值，线程数量
    - **--use_gpu** true / false，是否使用 GPU
    - **--use_nnapi** true / false，是否使用 nnapi
    - **--use_xnnpack** true / false，是否使用 xnnpack
    - **--use_coreml** true / false，是否使用 coreml
  - **Int8 模型 nnapi 测试**
    ```cpp
    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/mobilenet_v1_1.0_224_quant.tflite --num_threads=1
    INFO: Initialized TensorFlow Lite runtime.
    The input model file size (MB): 4.27635
    Inference timings in us: Init: 5388, First inference: 101726, Warmup (avg): 92755.2, Inference (avg): 90865.9

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/mobilenet_v1_1.0_224_quant.tflite --num_threads=4
    INFO: Initialized TensorFlow Lite runtime.
    The input model file size (MB): 4.27635
    Inference timings in us: Init: 5220, First inference: 50829, Warmup (avg): 29745.6, Inference (avg): 27044.7

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/mobilenet_v1_1.0_224_quant.tflite --num_threads=1 --use_nnapi=true
    Explicitly applied NNAPI delegate, and the model graph will be completely executed by the delegate.
    The input model file size (MB): 4.27635
    Inference timings in us: Init: 25558, First inference: 9992420, Warmup (avg): 9.99242e+06, Inference (avg): 8459.69

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/mobilenet_v1_1.0_224_quant.tflite --num_threads=4 --use_nnapi=true
    Explicitly applied NNAPI delegate, and the model graph will be completely executed by the delegate.
    The input model file size (MB): 4.27635
    Inference timings in us: Init: 135723, First inference: 10013451, Warmup (avg): 1.00135e+07, Inference (avg): 8413.35

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/efficientnet-lite0-int8.tflite --num_threads=1
    INFO: Initialized TensorFlow Lite runtime.
    The input model file size (MB): 5.42276
    Inference timings in us: Init: 16296, First inference: 111237, Warmup (avg): 100603, Inference (avg): 98068.9

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/efficientnet-lite0-int8.tflite --num_threads=4
    INFO: Initialized TensorFlow Lite runtime.
    The input model file size (MB): 5.42276
    Inference timings in us: Init: 13910, First inference: 52150, Warmup (avg): 30097.1, Inference (avg): 28823.8

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/efficientnet-lite0-int8.tflite --num_threads=1 --use_nnapi=true
    Explicitly applied NNAPI delegate, and the model graph will be partially executed by the delegate w/ 11 delegate kernels.
    The input model file size (MB): 5.42276
    Inference timings in us: Init: 30724, First inference: 226753, Warmup (avg): 171396, Inference (avg): 143630

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/efficientnet-lite0-int8.tflite --num_threads=4 --use_nnapi=true
    Explicitly applied NNAPI delegate, and the model graph will be partially executed by the delegate w/ 11 delegate kernels.
    The input model file size (MB): 5.42276
    Inference timings in us: Init: 32209, First inference: 207213, Warmup (avg): 75055, Inference (avg): 53974.5
    ```
  - **Float32 模型 xnnpack 测试**
    ```cpp
    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/mobilenet_v1_1.0_224.tflite --num_threads=1
    INFO: Initialized TensorFlow Lite runtime.
    The input model file size (MB): 16.9008
    Inference timings in us: Init: 2491, First inference: 183222, Warmup (avg): 170631, Inference (avg): 163455

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/mobilenet_v1_1.0_224.tflite --num_threads=4
    INFO: Initialized TensorFlow Lite runtime.
    The input model file size (MB): 16.9008
    Inference timings in us: Init: 2482, First inference: 101750, Warmup (avg): 58520.2, Inference (avg): 52692.1

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/mobilenet_v1_1.0_224.tflite --num_threads=1 --use_xnnpack=true
    Explicitly applied XNNPACK delegate, and the model graph will be partially executed by the delegate w/ 2 delegate kernels.
    The input model file size (MB): 16.9008
    Inference timings in us: Init: 55797, First inference: 167033, Warmup (avg): 160670, Inference (avg): 159191

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/mobilenet_v1_1.0_224.tflite --num_threads=4 --use_xnnpack=true
    Explicitly applied XNNPACK delegate, and the model graph will be partially executed by the delegate w/ 2 delegate kernels.
    The input model file size (MB): 16.9008
    Inference timings in us: Init: 61780, First inference: 75098, Warmup (avg): 51450.6, Inference (avg): 47564.3

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/efficientnet-lite0-fp32.tflite --num_threads=1
    INFO: Initialized TensorFlow Lite runtime.
    The input model file size (MB): 18.5702
    Inference timings in us: Init: 6697, First inference: 169388, Warmup (avg): 148210, Inference (avg): 141517

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/efficientnet-lite0-fp32.tflite --num_threads=4
    INFO: Initialized TensorFlow Lite runtime.
    The input model file size (MB): 18.5702
    Inference timings in us: Init: 4137, First inference: 84115, Warmup (avg): 52832.1, Inference (avg): 52848.9

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/efficientnet-lite0-fp32.tflite --num_threads=1 --use_xnnpack=true
    Explicitly applied XNNPACK delegate, and the model graph will be completely executed by the delegate.
    The input model file size (MB): 18.5702
    Inference timings in us: Init: 53629, First inference: 120858, Warmup (avg): 114820, Inference (avg): 112744

    $ adb shell /data/local/tmp/benchmark_model --graph=/data/local/tmp/efficientnet-lite0-fp32.tflite --num_threads=4 --use_xnnpack=true
    Explicitly applied XNNPACK delegate, and the model graph will be completely executed by the delegate.
    The input model file size (MB): 18.5702
    Inference timings in us: Init: 52265, First inference: 45786, Warmup (avg): 42789.7, Inference (avg): 40117.3
    ```
***
