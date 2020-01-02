# ___2017 - 10 - 22 Kaggle___

- [Titanic Data Science Solutions](https://www.kaggle.com/startupsci/titanic-data-science-solutions/notebook)
- [Kaggle competitions gettingStarted](https://www.kaggle.com/competitions?sortBy=grouped&group=general&page=1&pageSize=20&category=gettingStarted)
- [Kaggle Facial Keypoints](https://www.kaggle.com/c/facial-keypoints-detection)
- [Introduction to Ensembling/Stacking in Python](https://www.kaggle.com/arthurtok/introduction-to-ensembling-stacking-in-python)

# Kaggle Facial Keypoints
## Colab
  - [Colab kaggle_facial_keypoints_detection.ipynb](https://colab.research.google.com/drive/1lLHxtXTnbzW5M-mlk59av6rKtqqh_fwW)
## Fillna in data and conv model
  ```py
  cd ~/workspace/datasets/facial-keypoints-detection/

  train_data = pd.read_csv('./training.csv')

  aa = train_data['Image'][0]
  bb = np.array(aa.split(' '), dtype='int').reshape(96, 96, 1)
  plt.imshow(bb[:, :, 0], cmap='gray')

  train_data.isnull().any().value_counts()
  train_data.isnull().any(1).value_counts()
  train_data.fillna(method='ffill', inplace=True)

  imags = []
  for imm in train_data.Image:
      img = [int(ii) if len(ii.strip()) != 0 else 0 for ii in imm.strip().split(' ')]
      imags.append(img)
  train_x = np.array(imags).reshape(-1, 96, 96, 1) / 255
  train_y = train_data.drop('Image', axis=1).to_numpy() / 96
  np.fromstring(x, dtype=int, sep=' ').reshape((96,96))

  test_data = pd.read_csv('./test.csv')
  images = [np.fromstring(ii, dtype=int, sep=' ') for ii in test_data.Image]
  test_x = np.array(images).reshape(-1, 96, 96, 1) / 255

  np.savez('train_test', train_x=train_x, train_y=train_y, test_x=test_x)
  ```
  ```py
  fig, axis = plt.subplots(2, 5)
  axis = axis.flatten()
  for ax, imm, ipp in zip(axis, train_x, train_y):
      ax.imshow(imm[:, :, 0], cmap='gray')
      ax.scatter(ipp[0::2] * 96, ipp[1::2] * 96)
      ax.set_axis_off()

  gpus = tf.config.experimental.list_physical_devices('GPU')
  tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  tf.config.experimental.set_memory_growth(gpus[0], True)
  aa = np.load('train_test.npz')
  train_x, train_y, test_x = aa['train_x'], aa['train_y'], aa['test_x']
  print(train_x.shape, train_y.shape)
  # (7049, 96, 96, 1) (7049, 30)

  from tensorflow import keras
  from tensorflow.keras.layers import LeakyReLU
  from tensorflow.keras.models import Sequential, Model
  from tensorflow.keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D

  def Conv_model(input_shape=(96,96,1), num_classes=30):
      model = Sequential()

      model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=input_shape))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())
      model.add(MaxPool2D(pool_size=(2, 2)))

      model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())
      model.add(MaxPool2D(pool_size=(2, 2)))

      model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())
      model.add(MaxPool2D(pool_size=(2, 2)))

      model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
      # model.add(BatchNormalization())
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())
      model.add(MaxPool2D(pool_size=(2, 2)))

      model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())
      model.add(MaxPool2D(pool_size=(2, 2)))

      model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())

      model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
      model.add(LeakyReLU(alpha = 0.1))
      model.add(BatchNormalization())


      model.add(Flatten())
      model.add(Dense(512,activation='relu'))
      model.add(Dropout(0.1))
      model.add(Dense(num_classes))

      return model

  model = Conv_model(input_shape=(96,96,1), num_classes=30)
  model.summary()

  model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
  hist = model.fit(train_x, train_y, epochs=150, batch_size=256, validation_split=0.2)
  # Epoch 50/50
  # 5639/5639 [==============================] - 3s 534us/sample - loss: 0.0055 - mae: 0.0551 - val_loss: 0.0015 - val_mae: 0.0275
  # Epoch 200/200
  # 5639/5639 [==============================] - 3s 550us/sample - loss: 0.0017 - mae: 0.0290 - val_loss: 6.4047e-04 - val_mae: 0.0148
  # Epoch 250/250
  # 5639/5639 [==============================] - 3s 550us/sample - loss: 6.3462e-04 - mae: 0.0157 - val_loss: 4.1988e-04 - val_mae: 0.0107
  tf.saved_model.save(model, './1')

  loaded = tf.saved_model.load('./1')
  interf = loaded.signatures['serving_default']
  pred = interf(tf.convert_to_tensor(test_x, dtype=tf.float32))
  pp = pred['dense_1'].numpy()

  fig, axis = plt.subplots(5, 5)
  axis = axis.flatten()
  for ax, imm, ipp in zip(axis, test_x[1000:], pp[1000:]):
      ax.imshow(imm[:, :, 0], cmap='gray')
      ax.scatter(ipp[0::2] * 96, ipp[1::2] * 96)
      ax.set_axis_off()
  ```
## Train two mini xception model with separated data
  ```py
  csv_data = pd.read_csv('./training.csv')
  all_image = csv_data.Image
  all_data = csv_data.drop('Image', axis=1)

  aa = all_data.isnull().apply(pd.value_counts)
  vv = aa.iloc[0] > 7000
  vv = all_data.count() > 7000
  # print(aa.iloc[0])

  integrity_columns = vv[vv == True].index
  integrity_data = all_data[integrity_columns]
  unintegrity_columns = vv[vv == False].index
  unintegrity_data = all_data[unintegrity_columns]
  print(integrity_data.shape, unintegrity_data.shape)
  # (7049, 8) (7049, 22)

  integrity_data_select = integrity_data.notnull().all(1)
  integrity_data = integrity_data[integrity_data_select].to_numpy() / 96
  unintegrity_data_select = unintegrity_data.notnull().all(1)
  unintegrity_data = unintegrity_data[unintegrity_data_select].to_numpy() / 96
  print(integrity_data.shape, unintegrity_data.shape)
  # (7000, 8) (2155, 22)

  image_data = [np.fromstring(ii, dtype=int, sep=' ') for ii in all_image]
  image_data = np.array(image_data).reshape(-1, 96, 96, 1) / 255
  integrity_image_data = image_data[integrity_data_select]
  unintegrity_image_data = image_data[unintegrity_data_select]
  print(integrity_image_data.shape, unintegrity_image_data.shape)
  # (7000, 96, 96, 1) (2155, 96, 96, 1)

  aa = integrity_image_data[:, :, ::-1, :]
  integrity_image_data_2 = np.concatenate([integrity_image_data, aa])
  bb = np.array([np.abs([1, 0] * 4 - ii) for ii in integrity_data])
  integrity_data_2 = np.concatenate([integrity_data, bb])
  print(integrity_image_data_2.shape, integrity_data_2.shape)
  # (14000, 96, 96, 1) (14000, 8)
  aa = unintegrity_image_data[:, :, ::-1, :]
  unintegrity_image_data_2 = np.concatenate([unintegrity_image_data, aa])
  bb = np.array([np.abs([1, 0] * 11 - ii) for ii in unintegrity_data])
  unintegrity_data_2 = np.concatenate([unintegrity_data, bb])
  print(unintegrity_image_data_2.shape, unintegrity_data_2.shape)
  # (4310, 96, 96, 1) (4310, 22)
  ```
  ```py
  ## model
  import tensorflow as tf
  from tensorflow.keras import layers
  from tensorflow.keras.models import Model
  from tensorflow.keras.layers import Input, Activation, Dropout, Conv2D, Dense, BatchNormalization, GlobalAveragePooling2D, MaxPooling2D, SeparableConv2D, concatenate
  from tensorflow.keras.regularizers import l2
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

  def mini_XCEPTION(input_shape, num_classes, l2_regularization=0.01):
      regularization = l2(l2_regularization)

      # base
      img_input = Input(input_shape)
      x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(img_input)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = Conv2D(8, (3, 3), strides=(1, 1), kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)

      # module 1
      residual = Conv2D(16, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
      residual = BatchNormalization()(residual)

      x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = SeparableConv2D(16, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
      x = layers.add([x, residual])

      # module 2
      residual = Conv2D(32, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
      residual = BatchNormalization()(residual)

      x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = SeparableConv2D(32, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
      x = layers.add([x, residual])

      # module 3
      residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
      residual = BatchNormalization()(residual)

      x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = SeparableConv2D(64, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
      x = layers.add([x, residual])

      # module 4
      residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
      residual = BatchNormalization()(residual)

      x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)
      x = Activation('relu')(x)
      x = SeparableConv2D(128, (3, 3), padding='same', kernel_regularizer=regularization, use_bias=False)(x)
      x = BatchNormalization()(x)

      x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
      x = layers.add([x, residual])

      x = Conv2D(num_classes, (3, 3), kernel_regularizer=regularization, padding='same')(x)
      x = GlobalAveragePooling2D()(x)
      x = Dropout(0.3)(x)
      output = Dense(num_classes)(x)

      model = Model(img_input, output)
      return model

  #training the model
  model = mini_XCEPTION((96, 96, 1), 30)
  model.compile(optimizer='adam', loss='mse', metrics=["mae",'accuracy'])
  model.summary()

  # callbacks
  early_stop = EarlyStopping('val_loss', patience=50)
  reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(50/4), verbose=1)
  model_checkpoint = ModelCheckpoint("./keras_checkpoints", 'val_loss', verbose=1, save_best_only=True)
  callbacks = [model_checkpoint, early_stop, reduce_lr]
  hist = model.fit(train_x, train_y, batch_size=256, epochs=200, verbose=1, callbacks=callbacks, validation_split=0.2)
  # 5639/5639 [==============================] - 9s 2ms/sample - loss: 0.0014 - mae: 0.0263 - accuracy: 0.5779 - val_loss: 9.2967e-04 - val_mae: 0.0209 - val_accuracy: 0.6149

  modela = mini_XCEPTION((96, 96, 1), 8)
  modela.compile(optimizer='adam', loss='mse', metrics=["mae",'accuracy'])
  modela.summary()
  hist = modela.fit(integrity_image_data, integrity_data, batch_size=256, epochs=200, verbose=1, callbacks=callbacks, validation_split=0.2)

  modelb = mini_XCEPTION((96, 96, 1), 22)
  modelb.compile(optimizer='adam', loss='mse', metrics=["mae",'accuracy'])
  modelb.summary()
  hist = modelb.fit(unintegrity_image_data, unintegrity_data, batch_size=256, epochs=200, verbose=1, callbacks=callbacks, validation_split=0.2)
  ```
## Fill missing data by model predict
  ```py
  train_select = label_data.notnull().all(1)
  test_select = label_data.isnull().any(1)
  feature_data_train = feature_data[train_select]
  feature_data_test = feature_data[test_select]
  label_data_train = label_data[train_select]
  label_data_test = label_data[test_select]
  print(feature_data_train.shape, feature_data_test.shape, label_data_train.shape, label_data_test.shape)
  # (2140, 8) (4860, 8) (2140, 22) (4860, 22)

  data_model = tf.keras.models.Sequential([
      layers.Input(shape=[8,]),
      layers.Dense(32),
      layers.Dense(64),
      layers.Dropout(0.1),
      layers.Dense(22)
  ])
  data_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
  data_model.fit(feature_data_train.to_numpy() / 96, label_data_train.to_numpy() / 96, epochs=150, verbose=1, validation_split=0.2)
  pred = data_model.predict(feature_data_train.to_numpy() / 96) * 96
  yy = label_data_train.to_numpy()
  print(((pred - yy) ** 2).sum(0))
  # [ 3621.08980817  1323.51842721  4242.13195221  1211.73041032
  #   3259.46735638  1228.29699784  4461.2025572   1299.45073064
  #   7967.08317693  9896.2162609   9108.37986825 11682.81402873
  #   8811.50593809 10063.70865708  8610.68334281 10994.46973902
  #  10112.65304633 15877.52864348 10719.43984155 16190.33853022
  #   1266.60829962 25825.21071116]
  print(((pred - yy) ** 2).max(0))
  # [ 29.89434935  10.22904536  39.20663097  15.79585616  25.61755941
  #   10.01605354  41.52014917  17.64380907  43.83821083 151.01494419
  #  414.56626675 133.13552419  44.85152737 143.92310498 408.80228546
  #   92.91965074  71.18659086 253.96232386 135.8481257  180.55287795
  #   16.11288025 335.5595574 ]

  train_image = train_image[select_data.notnull().all(1)]
  image_string_train = train_image[train_select]
  image_string_test = train_image[test_select]

  image_data_train = [np.fromstring(ii, dtype=int, sep=' ') for ii in image_string_train]
  image_data_train = np.array(image_data_train).reshape(-1, 96, 96, 1) / 255
  image_data_test = [np.fromstring(ii, dtype=int, sep=' ') for ii in image_string_test]
  image_data_test = np.array(image_data_test).reshape(-1, 96, 96, 1) / 255
  print(image_data_train.shape, image_data_test.shape)
  # (2140, 96, 96, 1) (4860, 96, 96, 1)

  mouth_sub_data = train_data[train_data.columns[-10:]]
  mouth_train = mouth_sub_data[mouth_sub_data.notnull().all(1)]
  mouth_test = mouth_sub_data[mouth_sub_data.isnull().any(1)]

  mouth_train_colx_x = mouth_train[['nose_tip_x', 'mouth_center_bottom_lip_x']]
  mouth_train_colx_y = mouth_train[['mouth_left_corner_x', 'mouth_right_corner_x', 'mouth_center_top_lip_x']]

  model_mouth_x = tf.keras.models.Sequential([
      layers.Input(shape=[2,]),
      layers.Dense(10),
      layers.Dense(10),
      layers.Dense(3)])
  model_mouth_x.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
  model_mouth_x.fit(mouth_train_x_x.to_numpy() / 96, mouth_train_x_y.to_numpy() / 96, epochs=150, verbose=1, validation_split=0.2)
  pred = model_mouth_x.predict(mouth_train_x_x.to_numpy() / 96) * 96
  print(((pred - yy) ** 2).sum(0))
  # [17116.85021261 14906.93838227   988.74043935]
  print(((pred - yy) ** 2).max(0))
  # [220.70537516 158.71248198  17.21451939]

  mouth_test_colx_x = mouth_test[['nose_tip_x', 'mouth_center_bottom_lip_x']]
  mouth_test_colx_y = mouth_test[['mouth_left_corner_x', 'mouth_right_corner_x', 'mouth_center_top_lip_x']]
  ```
  ```py
  df.describe().loc['count'].plot.bar()
  from sklearn.pipeline import make_pipeline
  from sklearn.preprocessing import MinMaxScaler

  output_pipe = make_pipeline(
      MinMaxScaler(feature_range=(-1, 1))
  )

  y_train = output_pipe.fit_transform(y)
  xy_predictions = output_pipe.inverse_transform(predictions).reshape(15, 2)
  ```
***
