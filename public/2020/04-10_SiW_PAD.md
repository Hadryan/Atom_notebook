# ___2020 - 04 - 10 SiW PAD___
***

- [Github ResNeSt](https://github.com/zhanghang1989/ResNeSt)
- [Github bleakie/MaskInsightface](https://github.com/bleakie/MaskInsightface)
- [Group Convolution分组卷积，以及Depthwise Convolution和Global Depthwise Convolution](https://cloud.tencent.com/developer/article/1394912)
- [深度学习中的卷积方式](https://zhuanlan.zhihu.com/p/75972500)
***

# 数据整理
## 预处理
  ```sh
  ffmpeg -i 006-2-3-4-1.mov -f image2 "foo/video-frame%05d.png"

  # Find .face files which has lines with only 3 elements.
  find ./Train -name *.face -exec grep -l '  ' {} \;

  find ./Train -name *.mov | wc -l
  # 2417
  find ./Test -name *.mov | wc -l
  # 2061

  # In most case it's missing a `48`
  find ./Train -name *.face -exec sed -i 's/  / 48 /' {} \;
  ```
  ```sh
  # find 'mov' file ends with '.m.mov'
  find Test/ -name '*.m.mov' | wc -l
  # 16

  # Rename
  find Test/ -name '*.m.mov' -exec rename 's/.m.mov/.mov/' {} \;
  ```
## 提取图片
  ```py
  #!/usr/bin/env python3

  import os
  import sys
  import argparse
  import glob2
  import cv2
  import numpy as np
  from skimage.io import imread, imsave
  from skimage.transform import SimilarityTransform, resize
  from tqdm import tqdm
  import imageio


  def face_align_landmarks(img, landmarks, image_size=(112, 112)):
      ret = []
      for landmark in landmarks:
          src = np.array(
              [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]],
              dtype=np.float32,
          )

          if image_size[0] != 112:
              src *= image_size[0] / 112
              src[:, 0] += 8.0

          dst = landmark.astype(np.float32)
          tform = SimilarityTransform()
          tform.estimate(dst, src)
          M = tform.params[0:2, :]
          ret.append(cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0))

      return np.array(ret)


  def get_video_faces(mov_path, detector):
      frame_path = os.path.relpath(mov_path)[:-4]
      # save_orign = os.path.join(data_path, "orign")
      save_orign = os.path.join("orign_frame", frame_path)
      save_mtcnn = os.path.join("detect_frame", frame_path)
      if os.path.exists(save_mtcnn):
          print(">>>> Already processed, skip :", frame_path)
          return
      os.makedirs(save_mtcnn, exist_ok=True)
      if os.path.exists(save_orign):
          print(">>>> Already processed, skip :", frame_path)
          return
      os.makedirs(save_orign, exist_ok=True)
      # save_resize = os.path.join(data_path, "resize")
      # os.makedirs(save_resize, exist_ok=True)
      # save_mtcnn = os.path.join(data_path, "detect")

      face_file = mov_path[:-4] + ".face"
      with open(face_file, "r") as ff:
          aa = ff.readlines()
      face_locs = [[int(jj) for jj in ii.strip().split(" ")] for ii in aa]

      vid = imageio.get_reader(mov_path, "ffmpeg")
      for id, (imm, loc) in tqdm(enumerate(zip(vid, face_locs)), "Processing " + mov_path, total=len(face_locs)):
          imm_orign = imm[loc[1] : loc[3], loc[0] : loc[2]]
          img_name = str(id) + ".png"
          if imm_orign.shape[0] != 0 and imm_orign.shape[1] != 0:
              imsave(os.path.join(save_orign, img_name), imm_orign)
              # imm_resize = resize(imm_orign, (112, 112))
              # imm_resize = (imm_resize * 255).astype(np.uint8)
              # imsave(os.path.join(save_resize, os.path.basename(img)), imm_resize)

              # ret = detector.detect_faces(imm)
              _, ccs, points = detector.detect_faces(imm)
              if points is None or len(points) == 0 or ccs[0] < 0.8:
                  print("No face found, image:", img_name)
              else:
                  # points = np.array([list(ii["keypoints"].values()) for ii in ret])
                  points = np.array([ii.reshape(2, 5)[::-1].T for ii in points])
                  nimgs = face_align_landmarks(imm, points)
                  # imsave(os.path.join(save_mtcnn, str(id) + str(ccs[0]) + '.png'), nimgs[0])
                  imsave(os.path.join(save_mtcnn, img_name), nimgs[0])


  if __name__ == "__main__":
      parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
      parser.add_argument("-R", "--reg", type=str, default="./*/*/*/*.mov", help="Regular string points to mov files.")
      parser.add_argument("-S", "--start", type=int, default=0, help="Begining mov index to handle with [Include].")
      parser.add_argument("-E", "--end", type=int, default=-1, help="End mov index to handle with [Exclude].")
      args = parser.parse_known_args(sys.argv[1:])[0]

      # from mtcnn import MTCNN
      # detector = MTCNN(steps_threshold=[0.6, 0.7, 0.7], min_face_size=40)
      sys.path.append("/home/leondgarse/workspace/samba/tdFace-flask")
      from mtcnn_tf.mtcnn import MTCNN

      detector = MTCNN(thresholds=[0.6, 0.7, 0.7], min_size=40)
      movs = glob2.glob(args.reg)
      movs.sort()
      if args.end != -1:
          movs = movs[args.start : args.end]
      else:
          movs = movs[args.start :]

      total = len(movs)
      for id, mov in enumerate(movs):
          print(">>>> %d/%d:" % (id + 1, total))
          get_video_faces(mov, detector)
  ```
  ```sh
  CUDA_VISIBLE_DEVICES='-1' ./extract_faces.py -R 'Train/*/*/*.mov'
  CUDA_VISIBLE_DEVICES='0' ./extract_faces.py -R 'Train/*/*/*.mov'
  CUDA_VISIBLE_DEVICES='1' ./extract_faces.py -R 'Train/*/*/*.mov'

  CUDA_VISIBLE_DEVICES='-1' ./extract_faces.py -R 'Test/*/*/*.mov'
  CUDA_VISIBLE_DEVICES='0' ./extract_faces.py -R 'Test/*/*/*.mov'
  CUDA_VISIBLE_DEVICES='1' ./extract_faces.py -R 'Test/*/*/*.mov'
  ```
## 图片数据分析
  ```py
  # ls detect_frame/Train/live/003/003-1-1-1-1
  import glob2
  image_counts = lambda rr: {os.path.sep.join(os.path.relpath(ii).split(os.path.sep)[1:]): len(os.listdir(ii)) for ii in glob2.glob(rr)}

  ddr = "detect_frame/Test/*/*/*"
  oor = "orign_frame/Test/*/*/*"
  # ddr = "detect_frame/Train/*/*/*"
  # oor = "orign_frame/Train/*/*/*"
  dss = pd.Series(image_counts(ddr), name='detect')
  oss = pd.Series(image_counts(oor), name='original')

  ''' Remove empty directories '''
  tt = pd.concat([dss, oss], axis=1, sort=False).fillna(0)
  tt.sort_values('detect').head(10)
  #                              detect  original  sub
  # Train/spoof/121/121-2-3-2-1       0         0    0
  # Train/spoof/081/081-2-3-2-1       0         0    0
  # Train/spoof/101/101-2-3-2-1       0         0    0
  # Train/spoof/101/101-2-3-3-1       0         0    0
  # Train/spoof/041/041-2-3-2-1       0         0    0
  # Train/spoof/077/077-1-3-3-2       0         0    0
  # Train/spoof/041/041-2-3-3-1       0         0    0
  # Train/spoof/121/121-2-3-3-1       0         0    0
  # Train/spoof/006/006-2-3-4-1     202       202    0
  # Train/spoof/060/060-2-3-1-2     202       209    7

  for ii in tt[tt.detect == 0].index:
      print(os.path.join('./detect_frame/', ii))
      os.rmdir(os.path.join('./detect_frame/', ii))
      os.rmdir(os.path.join('./orign_frame/', ii))

  tt = tt[tt.detect != 0].copy()

  ''' Check face detection results, see how many is missing '''
  tt['sub'] = tt['original'] - tt['detect']
  tt['sub'].describe()
  # count    2409.000000
  # mean        4.427978
  # std        13.497904
  # min         0.000000
  # 25%         0.000000
  # 50%         0.000000
  # 75%         2.000000
  # max       210.000000
  # Name: sub, dtype: float64
  tt.sort_values('sub')[-5:]
  # detect  original  sub
  # Train/spoof/156/156-2-3-2-2     274       394  120
  # Train/spoof/055/055-2-3-4-2     293       430  137
  # Train/spoof/104/104-2-3-4-2     279       426  147
  # Train/spoof/032/032-2-3-4-1     266       437  171
  # Train/spoof/159/159-2-3-2-1     204       414  210
  ```
  ```py
  ''' Folder size estimate '''
  files_size = lambda dd: [os.stat(os.path.join(dd, ii)).st_size for ii in  os.listdir(dd)]
  samples = tt.index[np.random.choice(tt.shape[0], 120, replace=False)]
  aa = [files_size(os.path.join("detect_frame", ii)) for ii in samples]
  mm = np.mean([np.mean(ii) for ii in aa])
  print("~%.2fGB" % (tt['detect'].sum() * mm / 1024 / 1024 / 1024))
  # ~28.99GB
  !du -hd1 detect_frame/Train
  # 15G     detect_frame/Train/live
  # 18G     detect_frame/Train/spoof
  # 33G     detect_frame/Train
  !du -hd1 detect_frame/Test
  # 13G     detect_frame/Test/live
  # 16G     detect_frame/Test/spoof
  # 29G     detect_frame/Test

  aa = [files_size(os.path.join("orign_frame", ii)) for ii in samples]
  mm = np.mean([np.mean(ii) for ii in aa])
  print("~%.2fGB" % (tt['original'].sum() * mm / 1024 / 1024 / 1024))
  # ~180.40GB
  !du -hd1 orign_frame/Train
  # 83G     orign_frame/Train/live
  # 105G    orign_frame/Train/spoof
  # 188G    orign_frame/Train
  !du -hd1 orign_frame/Test
  # 73G     orign_frame/Test/live
  # 92G     orign_frame/Test/spoof
  # 164G    orign_frame/Test
  ```
***

# Basic Training
  ```py
  # data_path = 'detect_frame/Train'
  # ls detect_frame/Train/live/003/003-1-1-1-1/0.png
  import sys
  # sys.path.append('/home/leondgarse/workspace/samba/Keras_insightface')
  sys.path.append('/home/tdtest/workspace/Keras_insightface')
  import data
  import myCallbacks
  image_names_reg = "*/*/*/*.png"
  image_classes_rule = lambda path: 0 if "live" in path else 1
  # image_names, image_classes, classes = data.pre_process_folder('detect_frame/Train', image_names_reg=image_names_reg, image_classes_rule=image_classes_rule)
  train_ds, steps_per_epoch, classes = data.prepare_dataset('detect_frame/Train', image_names_reg=image_names_reg, image_classes_rule=image_classes_rule, batch_size=160, img_shape=(48, 48), random_status=2, random_crop=(48, 48, 3))
  test_ds, validation_steps, _ = data.prepare_dataset('detect_frame/Test', image_names_reg=image_names_reg, image_classes_rule=image_classes_rule, batch_size=160, img_shape=(48, 48), random_status=0, random_crop=(48, 48, 3), is_train=False)

  aa, bb = next(train_ds.as_numpy_iterator())
  plt.imshow(np.vstack([np.hstack(aa[ii * 20 : (ii + 1) * 20]) for ii in range(int(np.ceil(aa.shape[0] / 20)))]))
  plt.axis('off')
  plt.tight_layout()

  with tf.distribute.MirroredStrategy().scope():
      from tensorflow.keras import layers
      model = keras.Sequential([
          layers.Conv2D(512, 1, strides=1, activation='relu'),
          layers.AveragePooling2D(pool_size=1),
          layers.Conv2D(128, 1, strides=1, activation='relu'),
          layers.AveragePooling2D(pool_size=1),
          layers.Conv2D(32, 1, strides=1, activation='relu'),
          layers.Dropout(0.5),
          layers.AveragePooling2D(pool_size=1),
          layers.Flatten(),
          layers.Dense(2, activation=tf.nn.softmax)
      ])

      import train
      bb = train.buildin_models("MobileNet", dropout=1, emb_shape=128, input_shape=(48, 48, 3))
      output = keras.layers.Dense(2, activation=tf.nn.softmax)(bb.outputs[0])
      model = keras.models.Model(bb.inputs[0], output)

      callbacks = [
          keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5),
          keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
          keras.callbacks.ModelCheckpoint("./keras.h5", monitor='val_loss', save_best_only=True),
          myCallbacks.Gently_stop_callback(),
      ]
      model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
      model.fit(
          train_ds,
          epochs=50,
          callbacks=callbacks,
          steps_per_epoch=steps_per_epoch,
          validation_data=test_ds,
          validation_steps=validation_steps,
          use_multiprocessing=True,
          workers=4,
      )
  ```
***

# Android TFlite
  - [Github syaringan357/Android-MobileFaceNet-MTCNN-FaceAntiSpoofing](https://github.com/syaringan357/Android-MobileFaceNet-MTCNN-FaceAntiSpoofing)
  - [Github Leezhen2014/BlurDetection](https://github.com/Leezhen2014/python--/blob/master/BlurDetection.py)
## Java
  ```java
  /**
   * 活体检测
   */
  private void antiSpoofing() {
      if (bitmapCrop1 == null || bitmapCrop2 == null) {
          Toast.makeText(this, "请先检测人脸", Toast.LENGTH_LONG).show();
          return;
      }

      // 活体检测前先判断图片清晰度
      int laplace1 = fas.laplacian(bitmapCrop1);

      String text = "清晰度检测结果left：" + laplace1;
      if (laplace1 < FaceAntiSpoofing.LAPLACIAN_THRESHOLD) {
          text = text + "，" + "False";
          resultTextView.setTextColor(getResources().getColor(android.R.color.holo_red_light));
      } else {
          long start = System.currentTimeMillis();

          // 活体检测
          float score1 = fas.antiSpoofing(bitmapCrop1);

          long end = System.currentTimeMillis();

          text = "活体检测结果left：" + score1;
          if (score1 < FaceAntiSpoofing.THRESHOLD) {
              text = text + "，" + "True";
              resultTextView.setTextColor(getResources().getColor(android.R.color.holo_green_light));
          } else {
              text = text + "，" + "False";
              resultTextView.setTextColor(getResources().getColor(android.R.color.holo_red_light));
          }
          text = text + "。耗时" + (end - start);
      }
      resultTextView.setText(text);

      // 第二张图片活体检测前先判断图片清晰度
      int laplace2 = fas.laplacian(bitmapCrop2);

      String text2 = "清晰度检测结果left：" + laplace2;
      if (laplace2 < FaceAntiSpoofing.LAPLACIAN_THRESHOLD) {
          text2 = text2 + "，" + "False";
          resultTextView2.setTextColor(getResources().getColor(android.R.color.holo_red_light));
      } else {
          // 活体检测
          float score2 = fas.antiSpoofing(bitmapCrop2);
          text2 = "活体检测结果right：" + score2;
          if (score2 < FaceAntiSpoofing.THRESHOLD) {
              text2 = text2 + "，" + "True";
              resultTextView2.setTextColor(getResources().getColor(android.R.color.holo_green_light));
          } else {
              text2 = text2 + "，" + "False";
              resultTextView2.setTextColor(getResources().getColor(android.R.color.holo_red_light));
          }
      }
      resultTextView2.setText(text2);
  }
  ```
  ```java
  public class FaceAntiSpoofing {
      private static final String MODEL_FILE = "FaceAntiSpoofing.tflite";

      public static final int INPUT_IMAGE_SIZE = 256; // 需要feed数据的placeholder的图片宽高
      public static final float THRESHOLD = 0.2f; // 设置一个阙值，大于这个值认为是攻击

      public static final int ROUTE_INDEX = 6; // 训练时观察到的路由索引

      public static final int LAPLACE_THRESHOLD = 50; // 拉普拉斯采样阙值
      public static final int LAPLACIAN_THRESHOLD = 1000; // 图片清晰度判断阙值

      private Interpreter interpreter;

      public FaceAntiSpoofing(AssetManager assetManager) throws IOException {
          Interpreter.Options options = new Interpreter.Options();
          options.setNumThreads(4);
          interpreter = new Interpreter(MyUtil.loadModelFile(assetManager, MODEL_FILE), options);
      }

      /**
       * 活体检测
       * @param bitmap
       * @return 评分
       */
      public float antiSpoofing(Bitmap bitmap) {
          // 将人脸resize为256X256大小的，因为下面需要feed数据的placeholder的形状是(1, 256, 256, 3)
          Bitmap bitmapScale = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);

          float[][][] img = normalizeImage(bitmapScale);
          float[][][][] input = new float[1][][][];
          input[0] = img;
          float[][] clss_pred = new float[1][8];
          float[][] leaf_node_mask = new float[1][8];
          Map<Integer, Object> outputs = new HashMap<>();
          outputs.put(interpreter.getOutputIndex("Identity"), clss_pred);
          outputs.put(interpreter.getOutputIndex("Identity_1"), leaf_node_mask);
          interpreter.runForMultipleInputsOutputs(new Object[]{input}, outputs);

          Log.i("FaceAntiSpoofing", "[" + clss_pred[0][0] + ", " + clss_pred[0][1] + ", "
                  + clss_pred[0][2] + ", " + clss_pred[0][3] + ", " + clss_pred[0][4] + ", "
                  + clss_pred[0][5] + ", " + clss_pred[0][6] + ", " + clss_pred[0][7] + "]");
          Log.i("FaceAntiSpoofing", "[" + leaf_node_mask[0][0] + ", " + leaf_node_mask[0][1] + ", "
                  + leaf_node_mask[0][2] + ", " + leaf_node_mask[0][3] + ", " + leaf_node_mask[0][4] + ", "
                  + leaf_node_mask[0][5] + ", " + leaf_node_mask[0][6] + ", " + leaf_node_mask[0][7] + "]");

          return leaf_score1(clss_pred, leaf_node_mask);
      }

      private float leaf_score1(float[][] clss_pred, float[][] leaf_node_mask) {
          float score = 0;
          for (int i = 0; i < 8; i++) {
              score += Math.abs(clss_pred[0][i]) * leaf_node_mask[0][i];
          }
          return score;
      }

      private float leaf_score2(float[][] clss_pred) {
          return clss_pred[0][ROUTE_INDEX];
      }

      /**
       * 归一化图片到[0, 1]
       * @param bitmap
       * @return
       */
      public static float[][][] normalizeImage(Bitmap bitmap) {
          int h = bitmap.getHeight();
          int w = bitmap.getWidth();
          float[][][] floatValues = new float[h][w][3];

          float imageStd = 255;
          int[] pixels = new int[h * w];
          bitmap.getPixels(pixels, 0, bitmap.getWidth(), 0, 0, w, h);
          for (int i = 0; i < h; i++) { // 注意是先高后宽
              for (int j = 0; j < w; j++) {
                  final int val = pixels[i * w + j];
                  float r = ((val >> 16) & 0xFF) / imageStd;
                  float g = ((val >> 8) & 0xFF) / imageStd;
                  float b = (val & 0xFF) / imageStd;

                  float[] arr = {r, g, b};
                  floatValues[i][j] = arr;
              }
          }
          return floatValues;
      }

      /**
       * 拉普拉斯算法计算清晰度
       * @param bitmap
       * @return 分数
       */
      public int laplacian(Bitmap bitmap) {
          // 将人脸resize为256X256大小的，因为下面需要feed数据的placeholder的形状是(1, 256, 256, 3)
          Bitmap bitmapScale = Bitmap.createScaledBitmap(bitmap, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, true);

          double[][] laplace = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
          int size = laplace.length;
          int[][] img = MyUtil.convertGreyImg(bitmapScale);
          int height = img.length;
          int width = img[0].length;

          int score = 0;
          for (int x = 0; x < height - size + 1; x++){
              for (int y = 0; y < width - size + 1; y++){
                  int result = 0;
                  // 对size*size区域进行卷积操作
                  for (int i = 0; i < size; i++){
                      for (int j = 0; j < size; j++){
                          result += (img[x + i][y + j] & 0xFF) * laplace[i][j];
                      }
                  }
                  if (result > LAPLACE_THRESHOLD) {
                      score++;
                  }
              }
          }
          return score;
      }
  }
  ```
  ```java
  /**
   * 图片转为灰度图
   * @param bitmap
   * @return 灰度图数据
   */
  public static int[][] convertGreyImg(Bitmap bitmap) {
      int w = bitmap.getWidth();
      int h = bitmap.getHeight();

      int[] pixels = new int[h * w];
      bitmap.getPixels(pixels, 0, w, 0, 0, w, h);

      int[][] result = new int[h][w];
      int alpha = 0xFF << 24;
      for(int i = 0; i < h; i++)	{
          for(int j = 0; j < w; j++) {
              int val = pixels[w * i + j];

              int red = ((val >> 16) & 0xFF);
              int green = ((val >> 8) & 0xFF);
              int blue = (val & 0xFF);

              int grey = (int)((float) red * 0.3 + (float)green * 0.59 + (float)blue * 0.11);
              grey = alpha | (grey << 16) | (grey << 8) | grey;
              result[i][j] = grey;
          }
      }
      return result;
  }
  ```
## Python BlurDetection
  ```py
  # -*-coding=UTF-8-*-
  """
  在无参考图下，检测图片质量的方法
  """
  import os
  import cv2

  import numpy as np
  from skimage import filters

  class BlurDetection:
      def __init__(self, strDir):
          print("图片检测对象已经创建...")
          self.strDir = strDir

      def _getAllImg(self, strType='jpg'):
          """
          根据目录读取所有的图片
          :param strType: 图片的类型
          :return:  图片列表
          """
          names = []
          for root, dirs, files in os.walk(self.strDir):  # 此处有bug  如果调试的数据还放在这里，将会递归的遍历所有文件
              for file in files:
                  # if os.path.splitext(file)[1]=='jpg':
                  names.append(str(file))
          return names

      def _imageToMatrix(self, image):
          """
          根据名称读取图片对象转化矩阵
          :param strName:
          :return: 返回矩阵
          """
          imgMat = np.matrix(image)
          return imgMat

      def _blurDetection(self, imgName):

          # step 1 图像的预处理
          img2gray, reImg = self.preImgOps(imgName)
          imgMat=self._imageToMatrix(img2gray)/255.0
          x, y = imgMat.shape
          score = 0
          for i in range(x - 2):
              for j in range(y - 2):
                  score += (imgMat[i + 2, j] - imgMat[i, j]) ** 2
          # step3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
          score=score/10
          newImg = self._drawImgFonts(reImg, str(score))
          newDir = self.strDir + "/_blurDetection_/"
          if not os.path.exists(newDir):
              os.makedirs(newDir)
          newPath = newDir + imgName
          cv2.imwrite(newPath, newImg)  # 保存图片
          cv2.imshow(imgName, newImg)
          cv2.waitKey(0)
          return score

      def _SMDDetection(self, imgName):

          # step 1 图像的预处理
          img2gray, reImg = self.preImgOps(imgName)
          f=self._imageToMatrix(img2gray)/255.0
          x, y = f.shape
          score = 0
          for i in range(x - 1):
              for j in range(y - 1):
                  score += np.abs(f[i+1,j]-f[i,j])+np.abs(f[i,j]-f[i+1,j])
          # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
          score=score/100
          newImg = self._drawImgFonts(reImg, str(score))
          newDir = self.strDir + "/_SMDDetection_/"
          if not os.path.exists(newDir):
              os.makedirs(newDir)
          newPath = newDir + imgName
          cv2.imwrite(newPath, newImg)  # 保存图片
          cv2.imshow(imgName, newImg)
          cv2.waitKey(0)
          return score

      def _SMD2Detection(self, imgName):
          """
          灰度方差乘积
          :param imgName:
          :return:
          """
          # step 1 图像的预处理
          img2gray, reImg = self.preImgOps(imgName)
          f=self._imageToMatrix(img2gray)/255.0
          x, y = f.shape
          score = 0
          for i in range(x - 1):
              for j in range(y - 1):
                  score += np.abs(f[i+1,j]-f[i,j])*np.abs(f[i,j]-f[i,j+1])
          # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
          score=score
          newImg = self._drawImgFonts(reImg, str(score))
          newDir = self.strDir + "/_SMD2Detection_/"
          if not os.path.exists(newDir):
              os.makedirs(newDir)
          newPath = newDir + imgName
          cv2.imwrite(newPath, newImg)  # 保存图片
          cv2.imshow(imgName, newImg)
          cv2.waitKey(0)
          return score
      def _Variance(self, imgName):
          """
                 灰度方差乘积
                 :param imgName:
                 :return:
                 """
          # step 1 图像的预处理
          img2gray, reImg = self.preImgOps(imgName)
          f = self._imageToMatrix(img2gray)

          # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
          score = np.var(f)
          newImg = self._drawImgFonts(reImg, str(score))
          newDir = self.strDir + "/_Variance_/"
          if not os.path.exists(newDir):
              os.makedirs(newDir)
          newPath = newDir + imgName
          cv2.imwrite(newPath, newImg)  # 保存图片
          cv2.imshow(imgName, newImg)
          cv2.waitKey(0)
          return score
      def _Vollath(self,imgName):
          """
                         灰度方差乘积
                         :param imgName:
                         :return:
                         """
          # step 1 图像的预处理
          img2gray, reImg = self.preImgOps(imgName)
          f = self._imageToMatrix(img2gray)
          source=0
          x,y=f.shape
          for i in range(x-1):
              for j in range(y):
                  source+=f[i,j]*f[i+1,j]
          source=source-x*y*np.mean(f)
          # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分

          newImg = self._drawImgFonts(reImg, str(source))
          newDir = self.strDir + "/_Vollath_/"
          if not os.path.exists(newDir):
              os.makedirs(newDir)
          newPath = newDir + imgName
          cv2.imwrite(newPath, newImg)  # 保存图片
          cv2.imshow(imgName, newImg)
          cv2.waitKey(0)
          return source
      def _Tenengrad(self,imgName):
          """
                         灰度方差乘积
                         :param imgName:
                         :return:
                         """
          # step 1 图像的预处理
          img2gray, reImg = self.preImgOps(imgName)
          f = self._imageToMatrix(img2gray)

          tmp = filters.sobel(f)
          source=np.sum(tmp**2)
          source=np.sqrt(source)
          # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分

          newImg = self._drawImgFonts(reImg, str(source))
          newDir = self.strDir + "/_Tenengrad_/"
          if not os.path.exists(newDir):
              os.makedirs(newDir)
          newPath = newDir + imgName
          cv2.imwrite(newPath, newImg)  # 保存图片
          cv2.imshow(imgName, newImg)
          cv2.waitKey(0)
          return source

      def Test_Tenengrad(self):
          imgList = self._getAllImg(self.strDir)
          for i in range(len(imgList)):
              score = self._Tenengrad(imgList[i])
              print(str(imgList[i]) + " is " + str(score))

      def Test_Vollath(self):
          imgList = self._getAllImg(self.strDir)
          for i in range(len(imgList)):
              score = self._Variance(imgList[i])
              print(str(imgList[i]) + " is " + str(score))


      def TestVariance(self):
          imgList = self._getAllImg(self.strDir)
          for i in range(len(imgList)):
              score = self._Variance(imgList[i])
              print(str(imgList[i]) + " is " + str(score))

      def TestSMD2(self):
          imgList = self._getAllImg(self.strDir)

          for i in range(len(imgList)):
              score = self._SMD2Detection(imgList[i])
              print(str(imgList[i]) + " is " + str(score))
          return
      def TestSMD(self):
          imgList = self._getAllImg(self.strDir)

          for i in range(len(imgList)):
              score = self._SMDDetection(imgList[i])
              print(str(imgList[i]) + " is " + str(score))
          return

      def TestBrener(self):
          imgList = self._getAllImg(self.strDir)

          for i in range(len(imgList)):
              score = self._blurDetection(imgList[i])
              print(str(imgList[i]) + " is " + str(score))
          return

      def preImgOps(self, imgName):
          """
          图像的预处理操作
          :param imgName: 图像的而明朝
          :return: 灰度化和resize之后的图片对象
          """
          strPath = self.strDir + imgName

          img = cv2.imread(strPath)  # 读取图片
          cv2.moveWindow("", 1000, 100)
          # cv2.imshow("原始图", img)
          # 预处理操作
          reImg = cv2.resize(img, (800, 900), interpolation=cv2.INTER_CUBIC)  #
          img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY)  # 将图片压缩为单通道的灰度图
          return img2gray, reImg

      def _drawImgFonts(self, img, strContent):
          """
          绘制图像
          :param img: cv下的图片对象
          :param strContent: 书写的图片内容
          :return:
          """
          font = cv2.FONT_HERSHEY_SIMPLEX
          fontSize = 5
          # 照片 添加的文字    /左上角坐标   字体   字体大小   颜色        字体粗细
          cv2.putText(img, strContent, (0, 200), font, fontSize, (0, 255, 0), 6)

          return img

      def _lapulaseDetection(self, imgName):
          """
          :param strdir: 文件所在的目录
          :param name: 文件名称
          :return: 检测模糊后的分数
          """
          # step1: 预处理
          img2gray, reImg = self.preImgOps(imgName)
          # step2: laplacian算子 获取评分
          resLap = cv2.Laplacian(img2gray, cv2.CV_64F)
          score = resLap.var()
          print("Laplacian %s score of given image is %s", str(score))
          # strp3: 绘制图片并保存  不应该写在这里  抽象出来   这是共有的部分
          newImg = self._drawImgFonts(reImg, str(score))
          newDir = self.strDir + "/_lapulaseDetection_/"
          if not os.path.exists(newDir):
            os.makedirs(newDir)
        newPath = newDir + imgName
        # 显示
        cv2.imwrite(newPath, newImg)  # 保存图片
        cv2.imshow(imgName, newImg)
        cv2.waitKey(0)

        # step3: 返回分数
        return score

    def TestDect(self):
        names = self._getAllImg()
        for i in range(len(names)):
            score = self._lapulaseDetection(names[i])
            print(str(names[i]) + " is " + str(score))
        return


  if __name__ == "__main__":
      BlurDetection = BlurDetection(strDir="D:/document/ZKBH/bug/face/")
      BlurDetection.Test_Tenengrad () # TestSMD
  ```
## Test
  - **Blur test**
    ```py
    from scipy import signal
    from skimage.color import rgb2gray
    import cv2

    LAPLACE_THRESHOLD = 50 # 拉普拉斯采样阙值
    LAPLACIAN_THRESHOLD = 1000 # 图片清晰度判断阙值

    Laplace = lambda imm: np.array([[sum([mm * nn for mm, nn in zip(imm[ii: ii + 3, jj: jj + 3].flatten(), [0, 1, 0, 1, -4, 1, 0, 1, 0])]) for jj in range(imm.shape[1] - 3 + 1)] for ii in range(imm.shape[0] - 3 + 1)])
    Laplace_2 = lambda imm: signal.convolve2d(imm, [[0, 1, 0], [1, -4, 1], [0, 1, 0]], mode='valid')
    Laplace_3 = lambda imm: cv2.Laplacian(imm.astype('uint8'), cv2.CV_32F)

    def laplacian_blur(img, laplace):
        lap_img = laplace(rgb2gray(img) * 255)
        score_sum = (lap_img > LAPLACE_THRESHOLD).sum()
        score_var = lap_img.var()
        return score_sum, score_var
    ```
    ```py
    def image_test(imm, det=None):
        if isinstance(imm, str):
            imm = imread(imm)
        print("Original image:")
        print("    Laplace_2 conv score:", laplacian_blur(imm, Laplace_2))
        print("    Laplace_3 cv2 score:", laplacian_blur(imm, Laplace_3))
        plt.imshow(imm)

        if det:
            bb, cc, pp = det.detect_faces(imm)
            bb = bb[0].astype('int')
            iim = imm[bb[0]: bb[2], bb[1]: bb[3]]
            print("Face area:")
            print("    Laplace_2 conv score:", laplacian_blur(iim, Laplace_2))
            print("    Laplace_3 cv2 score:", laplacian_blur(iim, Laplace_3))
            plt.plot([bb[1], bb[3], bb[3], bb[1], bb[1]], [bb[0], bb[0], bb[2], bb[2], bb[0]])
        plt.axis('off')
        plt.tight_layout()

    sys.path.append('/home/leondgarse/workspace/samba/tdFace-flask/')
    from mtcnn_tf.mtcnn import MTCNN
    det = MTCNN()

    image_test('./face_recognition_collection/test_img/blur_1.png', det)                                                                     
    # Original image:
    #     Laplace_2 conv score: (2988, 28.522123750476517)
    #     Laplace_3 cv2 score: (3626, 45.7521)
    # Face area:
    #     Laplace_2 conv score: (1, 6.572161511502175)
    #     Laplace_3 cv2 score: (1, 7.178362)
    image_test('./face_recognition_collection/test_img/blur_2.png', det)                                                                     
    # Original image:
    #     Laplace_2 conv score: (104, 6.858723524141351)
    #     Laplace_3 cv2 score: (110, 7.6073356)
    # Face area:
    #     Laplace_2 conv score: (31, 12.997524875546874)
    #     Laplace_3 cv2 score: (31, 13.875649)
    ```
    ![](images/blur_1.png)
    ![](images/blur_2.png)
  - **Tflite detect**
    ```py
    # /home/leondgarse/workspace/facial_presentation_attack_detection/Android-MobileFaceNet-MTCNN-FaceAntiSpoofing/app/src/main/assets

    from skimage.transform import resize

    def tflite_detect(imm, interp, det=None):
        if isinstance(imm, str):
            imm = imread(imm)
        if det:
            bb, cc, pp = det.detect_faces(imm)
            bb = bb[0].astype('int')
            imm = imm[bb[0]: bb[2], bb[1]: bb[3]]
        imm = resize(imm, (256, 256))

        input_index = interp.get_input_details()[0]["index"]
        class_pred_node = interp.get_output_details()[0]["index"]
        leaf_node_mask_node = interp.get_output_details()[1]["index"]

        interp.set_tensor(input_index, tf.cast(tf.expand_dims(imm, 0), dtype=tf.float32))
        interp.invoke()
        class_pred = interp.get_tensor(class_pred_node)
        leaf_node_mask = interp.get_tensor(leaf_node_mask_node)
        return class_pred, leaf_node_mask, np.dot(np.abs(class_pred[0]), leaf_node_mask[0])

    interp = tf.lite.Interpreter('/home/leondgarse/workspace/facial_presentation_attack_detection/Android-MobileFaceNet-MTCNN-FaceAntiSpoofing/app/src/main/assets/FaceAntiSpoofing.tflite')
    interp.allocate_tensors()

    tflite_detect('./face_recognition_collection/test_img/blur_1.png', interp, det)
    # (array([[ 1.016278  ,  0.00424137,  1.0010722 , -0.00470004,  0.909961  ,
    #           0.29483515,  0.31600803,  0.00585337]], dtype=float32),
    #  array([[0., 0., 0., 0., 1., 0., 0., 0.]], dtype=float32),
    #  0.909961)

    tflite_detect('./face_recognition_collection/test_img/blur_2.png', interp, det)
    # (array([[1.0193512 , 0.879801  , 1.0011172 , 0.04316462, 0.99281096,
    #          0.9843273 , 0.946206  , 0.07745843]], dtype=float32),
    #  array([[0., 0., 0., 0., 0., 0., 0., 1.]], dtype=float32),
    #  0.07745843)
    ```
***

## 曲线拟合
  ```py
  import json
  with open("./checkpoints/keras_resnet101_emore_hist.json", 'r') as ff:
      jj = json.load(ff)
  ss = jj['loss'][29:-5]
  ['%.4f' % ii for ii in jj['loss'][-10:]]
  # ['8.6066', '8.2645', '7.9587', '7.6866', '7.4418', '7.2208']

  zz = np.polyfit(np.arange(1, len(ss)), ss[1:], 3)
  yy = np.poly1d(zz)
  ["%.4f" % ii for ii in yy(np.arange(len(ss) - 5, len(ss) + 10))]
  # ['8.6065', '8.2710', '7.9557', '7.6401', '7.3035', '6.9252', '6.4847', '5.9613']

  ee = 0.105
  pp = ss[:len(ss) - 3].copy()
  for ii in range(len(ss) - 5, len(ss) + 10):
      pp.append(pp[ii - 1] - (pp[ii - 2] - pp[ii - 1]) * (1 - ee))
      print("%.4f" % pp[-1], end=', ')
  # 8.5960, 8.2454, 7.9316, 7.6508, 7.3994, 7.1744, 6.9731, 6.7929
  # ==> (f(x-1) - f(x)) / (f(x-2) - f(x-1)) = (1 - ee)
  #     && f(x) = aa * np.exp(-bb * x) + cc
  # ==> (np.exp(bb) - 1) / (np.exp(2 * bb) - np.exp(bb)) = (1 - ee)
  # ==> (1 - ee) * np.exp(2 * bb) - (2 - ee) * np.exp(bb) + 1 = 0

  from sympy import solve, symbols, Eq
  bb = symbols('bb')
  brr = solve(Eq(np.e ** (2 * bb) * (1 - ee) - (2 - ee) * np.e ** bb + 1, 0), bb)
  print(brr) # [0.0, 0.110931560707281]
  ff = lambda xx: np.e ** (-xx * brr[1])
  ['%.4f' % ((ff(ii - 1) - ff(ii)) / (ff(ii - 2) - ff(ii - 1))) for ii in range(10, 15)]
  # ['0.8950', '0.8950', '0.8950', '0.8950', '0.8950']

  aa, cc = symbols('aa'), symbols('cc')
  rr = solve([Eq(aa * ff(len(ss) - 3) + cc, ss[-3]), Eq(aa * ff(len(ss) - 1) + cc, ss[-1])], [aa, cc])
  func_solve = lambda xx: rr[aa] * ff(xx) + rr[cc]
  ["%.4f" % ii for ii in func_solve(np.arange(len(ss) - 5, len(ss) + 10))]
  # ['8.6061', '8.2645', '7.9587', '7.6850', '7.4401', '7.2209', '7.0247', '6.8491']

  from scipy.optimize import curve_fit

  def func_curv(x, a, b, c):
      return a * np.exp(-b * x) + c
  xx = np.arange(1, 1 + len(ss[1:]))
  popt, pcov = curve_fit(func_curv, xx, ss[1:])
  print(popt) # [6.13053796 0.1813183  6.47103657]
  ["%.4f" % ii for ii in func_curv(np.arange(len(ss) - 5, len(ss) + 10), *popt)]
  # ['8.5936', '8.2590', '7.9701', '7.7208', '7.5057', '7.3200', '7.1598', '7.0215']

  plt.plot(np.arange(len(ss) - 3, len(ss)), ss[-3:], label="Original Curve")
  xx = np.arange(len(ss) - 3, len(ss) + 3)
  plt.plot(xx, pp[-len(xx):], label="Manuel fit")
  plt.plot(xx, func_solve(xx), label="func_solve fit")
  plt.plot(xx, func_curv(xx, *popt), label="func_curv fit")
  plt.legend()
  ```
## Multi GPU
  ```py
  tf.debugging.set_log_device_placement(True)

  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
    inputs = tf.keras.layers.Input(shape=(1,))
    predictions = tf.keras.layers.Dense(1)(inputs)
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))

  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
  model.fit(dataset, epochs=2)
  model.evaluate(dataset)
  ```
  ```py
  mirrored_strategy = tf.distribute.MirroredStrategy()
  # Compute global batch size using number of replicas.
  BATCH_SIZE_PER_REPLICA = 5
  global_batch_size = (BATCH_SIZE_PER_REPLICA *
                       mirrored_strategy.num_replicas_in_sync)
  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100)
  dataset = dataset.batch(global_batch_size)

  LEARNING_RATES_BY_BATCH_SIZE = {5: 0.1, 10: 0.15}
  learning_rate = LEARNING_RATES_BY_BATCH_SIZE[global_batch_size]

  with mirrored_strategy.scope():
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    optimizer = tf.keras.optimizers.SGD()

  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(
      global_batch_size)
  dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)

  @tf.function
  def train_step(dist_inputs):
    def step_fn(inputs):
      features, labels = inputs

      with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        logits = model(features, training=True)
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
      return cross_entropy

    per_example_losses = mirrored_strategy.experimental_run_v2(step_fn, args=(dist_inputs,))
    mean_loss = mirrored_strategy.reduce(
        tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
    return mean_loss

  with mirrored_strategy.scope():
    for inputs in dist_dataset:
      print(train_step(inputs))
  ```
  ```py
  @tf.function
  def step_fn(inputs):
      return ss.experimental_assign_to_logical_device(mm.predict(inputs), 0)

  with ss.scope():
    ss.run(step_fn, args=(np.ones([2, 112, 112, 3]),))
  ```
## NPU
  ```sh
  资料：建议用谷歌浏览器或QQ浏览器浏览可以直接网页翻译！
  文档汇总： https://docs.khadas.com/vim3/index.html
  烧录教程： https://docs.khadas.com/vim3/UpgradeViaUSBCable.html
  硬件资料： https://docs.khadas.com/vim3/HardwareDocs.html
  安卓固件下载： https://docs.khadas.com/vim3/FirmwareAndroid.html
  Ubuntu固件下载： https://docs.khadas.com/vim3/FirmwareUbuntu.html
  第三方操作系统： https://docs.khadas.com/vim3/FirmwareThirdparty.html#AndroidTV

  VIM3释放 NPU资料流程: https://www.khadas.com/npu-toolkit-vim3
  ```
## 阿里云数据整理
  ```py
  import cv2
  import shutil
  import glob2
  from tqdm import tqdm
  from skimage.transform import SimilarityTransform
  from sklearn.preprocessing import normalize

  def face_align_landmarks(img, landmarks, image_size=(112, 112)):
      ret = []
      for landmark in landmarks:
          src = np.array(
              [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.729904, 92.2041]],
              dtype=np.float32,
          )

          if image_size[0] != 112:
              src *= image_size[0] / 112
              src[:, 0] += 8.0

          dst = landmark.astype(np.float32)
          tform = SimilarityTransform()
          tform.estimate(dst, src)
          M = tform.params[0:2, :]
          ret.append(cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0))

      return np.array(ret)

  def extract_face_images(source_reg, dest_path, detector, limit=-1):
      aa = glob2.glob(source_reg)
      dest_single = dest_path
      dest_multi = dest_single + '_multi'
      dest_none = dest_single + '_none'
      os.makedirs(dest_none, exist_ok=True)
      if limit != -1:
          aa = aa[:limit]
      for ii in tqdm(aa):
          imm = imread(ii)
          bbs, pps = detector(imm)
          if len(bbs) == 0:
              shutil.copy(ii, os.path.join(dest_none, '_'.join(ii.split('/')[-2:])))
              continue
          user_name, image_name = ii.split('/')[-2], ii.split('/')[-1]
          if len(bbs) == 1:
              dest_path = os.path.join(dest_single, user_name)
          else:
              dest_path = os.path.join(dest_multi, user_name)

          if not os.path.exists(dest_path):
              os.makedirs(dest_path)
          # if len(bbs) != 1:
          #     shutil.copy(ii, dest_path)

          nns = face_align_landmarks(imm, pps)
          image_name_form = '%s_{}.%s' % tuple(image_name.split('.'))
          for id, nn in enumerate(nns):
              dest_name = os.path.join(dest_path, image_name_form.format(id))
              imsave(dest_name, nn)

  import insightface
  os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
  # retina = insightface.model_zoo.face_detection.retinaface_mnet025_v1()
  retina = insightface.model_zoo.face_detection.retinaface_r50_v1()
  retina.prepare(0)
  detector = lambda imm: retina.detect(imm)

  sys.path.append('/home/leondgarse/workspace/samba/tdFace-flask')
  from face_model.face_model import FaceModel
  det = FaceModel(None)

  def detector(imm):
      bbox, confid, points  = det.get_face_location(imm)
      return bbox, points
  extract_face_images("./face_image/*/*.jpg", 'tdevals', detector)
  extract_face_images("./tdface_Register/*/*.jpg", 'tdface_Register_cropped', detector)
  extract_face_images("./tdface_Register/*/*.jpg", 'tdface_Register_mtcnn', detector)

  ''' Review _multi and _none folder by hand, then do detection again on _none folder using another detector '''
  inn = glob2.glob('tdface_Register_mtcnn_none/*.jpg')
  for ii in tqdm(inn):
      imm = imread(ii)
      bbs, pps = detector(imm)
      if len(bbs) != 0:
          image_name = os.path.basename(ii)
          user_name = image_name.split('_')[0]
          dest_path = os.path.join(os.path.dirname(ii), user_name)
          os.makedirs(dest_path, exist_ok=True)
          nns = face_align_landmarks(imm, pps)
          image_name_form = '%s_{}.%s' % tuple(image_name.split('.'))
          for id, nn in enumerate(nns):
              dest_name = os.path.join(dest_path, image_name_form.format(id))
              imsave(dest_name, nn)
              os.rename(ii, os.path.join(dest_path, image_name))

  print(">>>> 提取特征值")
  model_path = "/home/tdtest/workspace/Keras_insightface/checkpoints/keras_resnet101_emore_II_triplet_basic_agedb_30_epoch_107_0.971000.h5"
  model = tf.keras.models.load_model(model_path, compile=False)
  interp = lambda ii: normalize(model.predict((np.array(ii) - 127.5) / 127))
  register_path = 'tdface_Register_mtcnn'

  backup_file = 'tdface_Register_mtcnn.npy'
  if os.path.exists(backup_file):
      imms = np.load('tdface_Register_mtcnn.npy')
  else:
      imms = glob2.glob(os.path.join(register_path, "*/*.jpg"))
      np.save('tdface_Register_mtcnn.npy', imms)

  batch_size = 64
  steps = int(np.ceil(len(imms) / batch_size))
  embs = []
  for ii in tqdm(range(steps), total=steps):
      ibb = imms[ii * batch_size : (ii + 1) * batch_size]
      embs.append(interp([imread(jj) for jj in ibb]))

  embs = np.concatenate(embs)
  dd, pp = {}, {}
  for ii, ee in zip(imms, embs):
      user = os.path.basename(os.path.dirname(ii))
      dd[user] = np.vstack([dd.get(user, np.array([]).reshape(0, 512)), [ee]])
      pp[user] = np.hstack([pp.get(user, []), ii])
  # dd_bak = dd.copy()
  # pp_bak = pp.copy()
  print("Total: %d" % (len(dd)))

  print(">>>> 合并组间距离过小的成员")
  OUT_THREASH = 0.7
  tt = dd.copy()
  while len(tt) > 0:
      kk, vv = tt.popitem()
      # oo = np.vstack(list(tt.values()))
      for ikk, ivv in tt.items():
          imax = np.dot(vv, ivv.T).max()
          if imax > OUT_THREASH:
              print("First: %s, Second: %s, Max dist: %.4f" % (kk, ikk, imax))
              if kk in dd and ikk in dd:
                  dd[kk] = np.vstack([dd[kk], dd[ikk]])
                  dd.pop(ikk)
                  pp[kk] = np.hstack([pp[kk], pp[ikk]])
                  pp.pop(ikk)
  # print([kk for kk, vv in pp.items() if vv.shape[0] != dd[kk].shape[0]])
  print("Total left: %d" % (len(dd)))

  ''' Similar images between users '''
  src = 'tdface_Register_mtcnn'
  dst = 'tdface_Register_mtcnn_simi'
  with open('tdface_Register_mtcnn.foo', 'r') as ff:
      aa = ff.readlines()
  for id, ii in tqdm(enumerate(aa), total=len(aa)):
      first, second, simi = [jj.split(': ')[1] for jj in ii.strip().split(', ')]
      dest_path = os.path.join(dst, '_'.join([str(id), first, second, simi]))
      os.makedirs(dest_path, exist_ok=True)
      for pp in os.listdir(os.path.join(src, first)):
          src_path = os.path.join(src, first, pp)
          shutil.copy(src_path, os.path.join(dest_path, first + '_' + pp))
      for pp in os.listdir(os.path.join(src, second)):
          src_path = os.path.join(src, second, pp)
          shutil.copy(src_path, os.path.join(dest_path, second + '_' + pp))

  ''' Pos & Neg dists '''
  batch_size = 128
  gg = ImageDataGenerator(rescale=1./255, preprocessing_function=lambda img: (img - 0.5) * 2)
  tt = gg.flow_from_directory('./tdevals', target_size=(112, 112), batch_size=batch_size)
  steps = int(np.ceil(tt.classes.shape[0] / batch_size))
  embs = []
  classes = []
  for _ in tqdm(range(steps), total=steps):
      aa, bb = tt.next()
      emb = interp(aa)
      embs.extend(emb)
      classes.extend(np.argmax(bb, 1))
  embs = np.array(embs)
  classes = np.array(classes)
  class_matrix = np.equal(np.expand_dims(classes, 0), np.expand_dims(classes, 1))
  dists = np.dot(embs, embs.T)
  pos_dists = np.where(class_matrix, dists, np.ones_like(dists))
  neg_dists = np.where(np.logical_not(class_matrix), dists, np.zeros_like(dists))
  (neg_dists.max(1) <= pos_dists.min(1)).sum()
  ```
  ```py
  import glob2
  def dest_test(aa, bb, model, reg_path="./tdface_Register_mtcnn"):
      ees = []
      for ii in [aa, bb]:
          iee = glob2.glob(os.path.join(reg_path, ii, "*.jpg"))
          iee = [imread(jj) for jj in iee]
          ees.append(normalize(model.predict((np.array(iee) / 255. - 0.5) * 2)))
      return np.dot(ees[0], ees[1].T)

  with open('tdface_Register_mtcnn_0.7.foo', 'r') as ff:
      aa = ff.readlines()
  for id, ii in enumerate(aa):
      first, second, simi = [jj.split(': ')[1] for jj in ii.strip().split(', ')]
      dist = dest_test(first, second, model)
      if dist < OUT_THREASH:
          print(("first = %s, second = %s, simi = %s, model_simi = %f" % (first, second, simi, dist)))
  ```
  ```sh
  cp ./face_image/2818/1586472495016.jpg ./face_image/4609/1586475252234.jpg ./face_image/3820/1586472950858.jpg ./face_image/4179/1586520054080.jpg ./face_image/2618/1586471583221.jpg ./face_image/6696/1586529149923.jpg ./face_image/5986/1586504872276.jpg ./face_image/1951/1586489518568.jpg ./face_image/17/1586511238696.jpg ./face_image/17/1586511110105.jpg ./face_image/17/1586511248992.jpg ./face_image/4233/1586482466485.jpg ./face_image/5500/1586493019872.jpg ./face_image/4884/1586474119164.jpg ./face_image/5932/1586471784905.jpg ./face_image/7107/1586575911740.jpg ./face_image/4221/1586512334133.jpg ./face_image/5395/1586578437529.jpg ./face_image/4204/1586506059923.jpg ./face_image/4053/1586477985553.jpg ./face_image/7168/1586579239307.jpg ./face_image/7168/1586489559660.jpg ./face_image/5477/1586512847480.jpg ./face_image/4912/1586489637333.jpg ./face_image/5551/1586502762688.jpg ./face_image/5928/1586579219121.jpg ./face_image/6388/1586513897953.jpg ./face_image/4992/1586471873460.jpg ./face_image/5934/1586492793214.jpg ./face_image/5983/1586490703112.jpg ./face_image/5219/1586492929098.jpg ./face_image/5203/1586487204198.jpg ./face_image/6099/1586490074263.jpg ./face_image/5557/1586490232722.jpg ./face_image/4067/1586491778846.jpg ./face_image/4156/1586512886040.jpg ./face_image/5935/1586492829221.jpg ./face_image/2735/1586513495061.jpg ./face_image/5264/1586557233625.jpg ./face_image/1770/1586470942329.jpg ./face_image/7084/1586514100804.jpg ./face_image/5833/1586497276529.jpg ./face_image/2200/1586577699180.jpg tdevals_none
  ```
## Training Record
  ```sh
  # Mobilefacenet 160
  "loss": [5.0583, 1.6944, 1.2544, 1.0762]
  "accuracy": [0.2964, 0.6755, 0.7532, 0.7862]
  "lfw": [0.9715, 0.98, 0.986, 0.988]
  "cfp_fp": [0.865714, 0.893, 0.898857, 0.91]
  "agedb_30": [0.829167, 0.8775, 0.892667, 0.903167]
  ```
  ```sh
  # Mobilefacenet 768
  "loss": [4.82409207357388, 1.462764699449328, 1.011261948830721, 0.8042656191418587]
  "accuracy": [0.3281610608100891, 0.7102007865905762, 0.7921959757804871, 0.8312187790870667]
  "lfw": [0.9733333333333334, 0.9781666666666666, 0.9833333333333333, 0.983]
  "cfp_fp": [0.8654285714285714, 0.8935714285714286, 0.9002857142857142, 0.9004285714285715]
  "agedb_30": [0.8253333333333334, 0.8801666666666667, 0.8983333333333333, 0.8911666666666667]
  ```
  ```sh
  # Renet100 128
  "loss": [6.3005, 1.6274, 1.0608]
  "accuracy": [0.2196, 0.6881, 0.7901, 0.8293]
  "lfw": [0.97, 0.983, 0.981667, 0.986167]
  "cfp_fp": [0.865571, 0.894714, 0.903571, 0.910714]
  "agedb_30": [0.831833, 0.870167, 0.886333, 0.895833]
  ```
  ```sh
  # Renet100 1024
  "loss": [3.3549567371716877, 0.793737195027363, 0.518424223161905, 0.3872658337998814]
  "accuracy": [0.5023580193519592, 0.8334693908691406, 0.8864460587501526, 0.9125881195068359]
  "lfw": [0.9826666666666667, 0.9861666666666666, 0.9858333333333333, 0.99]
  "cfp_fp": [0.8998571428571429, 0.914, 0.9247142857142857, 0.9264285714285714]
  "agedb_30": [0.8651666666666666, 0.8876666666666667, 0.9013333333333333, 0.899]
  ```
  ```sh
  # EB0 160, Orign Conv + flatten + Dense
  "loss": [4.234781265258789, 1.6501317024230957],
  "accuracy": [0.35960495471954346, 0.6766131520271301],
  "lfw": [0.9816666666666667, 0.987, 0.9826666666666667],
  "cfp_fp": [0.9005714285714286, 0.9105714285714286],
  "agedb_30": [0.848, 0.8835]
  ```
  ```sh
  # EB0 160, GlobalAveragePooling
  "loss": [3.5231969356536865, 1.188686490058899, 0.8824349641799927, 0.7483137249946594],
  "accuracy": [0.45284023880958557, 0.7619950771331787, 0.8200176358222961, 0.8461616039276123],
  "lfw": 0.9826666666666667, 0.9855, 0.9886666666666667, 0.9856666666666667],
  "cfp_fp": [0.9022857142857142, 0.9201428571428572, 0.9212857142857143, 0.922],
  "agedb_30": [0.8531666666666666, 0.8805, 0.8891666666666667, 0.8966666666666666]
  ```
  ```sh
  # EB0 160, GDC
  "loss": [3.9325125217437744, 1.5069712400436401, 1.1851387023925781]
  "accuracy": [0.40046197175979614, 0.7043213248252869, 0.7628725171089172]
  "lfw": [0.9823333333333333, 0.9851666666666666, 0.99]
  "cfp_fp": [0.8962857142857142, 0.9145714285714286, 0.9158571428571428]
  "agedb_30": [0.8548333333333333, 0.8815, 0.8921666666666667]
  ```
  ```sh
  # EB4 840, GDC
  "loss": [2.727688789367676, 0.633741557598114, 0.4151850938796997, 0.3108983337879181]
  "accuracy": [0.5822311043739319, 0.8653680086135864, 0.9080367684364319, 0.9288726449012756]
  "lfw": [0.9876666666666667, 0.9881666666666666, 0.9911666666666666, 0.9913333333333333]
  "cfp_fp": [0.909, 0.918, 0.922, 0.9178571428571428]
  "agedb_30": [0.884, 0.9041666666666667, 0.9115, 0.9076666666666666]
  ```
  ```sh
  # MobileNet, 128, keras.losses.CategoricalCrossentropy, label_smoothing=0
  "loss": [5.324231072016197, 1.9473355543105044, 1.4203207439133088]
  "accuracy": [0.2530779540538788, 0.6315819621086121, 0.7252941727638245]
  "lfw": [0.9741666666666666, 0.9798333333333333, 0.9815]
  "cfp_fp": [0.8307142857142857, 0.8412857142857143, 0.8465714285714285]
  "agedb_30": [0.8403333333333334, 0.8608333333333333, 0.8725]
  ```
  ```sh
  # MobileNet, 128, keras.losses.CategoricalCrossentropy, label_smoothing=0.1
  "loss": [6.335052917359504, 3.5849099863856986, 3.1803239681401547]
  "accuracy": [0.279971718788147, 0.6559497714042664, 0.7314797639846802]
  "lfw": [0.9761666666666666, 0.9821666666666666, 0.9845]
  "cfp_fp": [0.824, 0.8487142857142858, 0.8571428571428571]
  "agedb_30": [0.8406666666666667, 0.8761666666666666, 0.8918333333333334]
  ```
  ```sh
  # MobileNet, 128, Epoch 8, ArcfaceLoss, label_smoothing=0
  "loss": [15.097385468255506, 14.698767092024056, 14.411642811109166]
  "accuracy": [0.92729651927948, 0.9315400719642639, 0.934310257434845]
  "lfw": [0.9895, 0.9906666666666667, 0.99]
  "cfp_fp": [0.8511428571428571, 0.8585714285714285, 0.8574285714285714]
  "agedb_30": [0.916, 0.9146666666666666, 0.9166666666666666]
  ```
***
```py
plt.plot(np.arange(100), [myCallbacks.scheduler(ii, 0.001, 0.1) for ii in np.arange(100)], label="lr 0.001, decay 0.1")

aa = keras.experimental.CosineDecay(0.001, 100)
plt.plot(np.arange(100), [aa(ii).numpy() for ii in np.arange(100)], label="lr 0.001, decay 0.1")
```
```py
from tensorflow.python.keras import backend as K

class CosineLrScheduler(keras.callbacks.Callback):
    def __init__(self, lr_base, decay_steps, alpha=0.0, warmup_iters=0, on_batch=False):
        super(CosineLrScheduler, self).__init__()
        self.lr_base, self.decay_steps, self.warmup_iters = lr_base, decay_steps, warmup_iters
        self.decay_steps -= self.warmup_iters
        self.schedule = keras.experimental.CosineDecay(self.lr_base, self.decay_steps, alpha=alpha)
        if on_batch == True:
            self.on_train_batch_begin = self.__lr_sheduler__
        else:
            self.on_epoch_begin = self.__lr_sheduler__

    def __lr_sheduler__(self, iterNum, logs=None):
        if iterNum < self.warmup_iters:
            lr = self.lr_base
        else:
            lr = self.schedule(iterNum - self.warmup_iters)
        if self.model is not None:
            K.set_value(self.model.optimizer.lr, lr)
        return lr

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
```

## tf dataset cache
  - **dataset.cache** **MUST** be placed **before** data random augment and shuffle
  ```py
  dd = np.arange(30).reshape(3, 10)

  ''' Cache before shuffle and random '''
  ds = tf.data.Dataset.from_tensor_slices(dd)
  # ds = ds.cache()
  ds = ds.shuffle(dd.shape[0])
  ds = ds.map(lambda xx: xx + tf.random.uniform((1,), 1, 10, dtype=tf.int64))

  for ii in range(3):
      print(">>>> Epoch:", ii)
      for jj in ds:
          print(jj)
  # >>>> Epoch: 0
  # tf.Tensor([ 9 10 11 12 13 14 15 16 17 18], shape=(10,), dtype=int64)
  # tf.Tensor([13 14 15 16 17 18 19 20 21 22], shape=(10,), dtype=int64)
  # tf.Tensor([23 24 25 26 27 28 29 30 31 32], shape=(10,), dtype=int64)
  # >>>> Epoch: 1
  # tf.Tensor([11 12 13 14 15 16 17 18 19 20], shape=(10,), dtype=int64)
  # tf.Tensor([21 22 23 24 25 26 27 28 29 30], shape=(10,), dtype=int64)
  # tf.Tensor([ 9 10 11 12 13 14 15 16 17 18], shape=(10,), dtype=int64)
  # >>>> Epoch: 2
  # tf.Tensor([23 24 25 26 27 28 29 30 31 32], shape=(10,), dtype=int64)
  # tf.Tensor([12 13 14 15 16 17 18 19 20 21], shape=(10,), dtype=int64)
  # tf.Tensor([ 3  4  5  6  7  8  9 10 11 12], shape=(10,), dtype=int64)

  ''' Cache before random but after shuffle '''
  ds2 = tf.data.Dataset.from_tensor_slices(dd)
  ds2 = ds2.shuffle(dd.shape[0])
  ds2 = ds2.cache()
  ds2 = ds2.map(lambda xx: xx + tf.random.uniform((1,), 1, 10, dtype=tf.int64))

  for ii in range(3):
      print(">>>> Epoch:", ii)
      for jj in ds2:
          print(jj)
  # >>>> Epoch: 0
  # tf.Tensor([26 27 28 29 30 31 32 33 34 35], shape=(10,), dtype=int64)
  # tf.Tensor([17 18 19 20 21 22 23 24 25 26], shape=(10,), dtype=int64)
  # tf.Tensor([ 6  7  8  9 10 11 12 13 14 15], shape=(10,), dtype=int64)
  # >>>> Epoch: 1
  # tf.Tensor([22 23 24 25 26 27 28 29 30 31], shape=(10,), dtype=int64)
  # tf.Tensor([17 18 19 20 21 22 23 24 25 26], shape=(10,), dtype=int64)
  # tf.Tensor([ 3  4  5  6  7  8  9 10 11 12], shape=(10,), dtype=int64)
  # >>>> Epoch: 2
  # tf.Tensor([21 22 23 24 25 26 27 28 29 30], shape=(10,), dtype=int64)
  # tf.Tensor([15 16 17 18 19 20 21 22 23 24], shape=(10,), dtype=int64)
  # tf.Tensor([ 3  4  5  6  7  8  9 10 11 12], shape=(10,), dtype=int64)

  ''' Cache after random and shuffle '''
  ds3 = tf.data.Dataset.from_tensor_slices(dd)
  ds3 = ds3.shuffle(dd.shape[0])
  ds3 = ds3.map(lambda xx: xx + tf.random.uniform((1,), 1, 10, dtype=tf.int64))
  ds3 = ds3.cache()

  for ii in range(3):
      print(">>>> Epoch:", ii)
      for jj in ds3:
          print(jj)
  # >>>> Epoch: 0
  # tf.Tensor([24 25 26 27 28 29 30 31 32 33], shape=(10,), dtype=int64)
  # tf.Tensor([14 15 16 17 18 19 20 21 22 23], shape=(10,), dtype=int64)
  # tf.Tensor([ 4  5  6  7  8  9 10 11 12 13], shape=(10,), dtype=int64)
  # >>>> Epoch: 1
  # tf.Tensor([24 25 26 27 28 29 30 31 32 33], shape=(10,), dtype=int64)
  # tf.Tensor([14 15 16 17 18 19 20 21 22 23], shape=(10,), dtype=int64)
  # tf.Tensor([ 4  5  6  7  8  9 10 11 12 13], shape=(10,), dtype=int64)
  # >>>> Epoch: 2
  # tf.Tensor([24 25 26 27 28 29 30 31 32 33], shape=(10,), dtype=int64)
  # tf.Tensor([14 15 16 17 18 19 20 21 22 23], shape=(10,), dtype=int64)
  # tf.Tensor([ 4  5  6  7  8  9 10 11 12 13], shape=(10,), dtype=int64)
  ```
