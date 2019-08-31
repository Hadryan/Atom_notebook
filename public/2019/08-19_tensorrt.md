# ___2019 - 08 - 19 TensorRT___

- [JerryJiaGit/facenet_trt](https://github.com/JerryJiaGit/facenet_trt)
- [tensorrt-developer-guide](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [onnx/onnx-tensorrt](https://github.com/onnx/onnx-tensorrt/)
而tensorRT 则是对训练好的模型进行优化。 tensorRT就只是 推理优化器。当你的网络训练完之后，可以将训练模型文件直接丢进tensorRT中，而不再需要依赖深度学习框架（Caffe，TensorFlow等）

可以认为tensorRT是一个只有前向传播的深度学习框架，这个框架可以将 Caffe，TensorFlow的网络模型解析，然后与tensorRT中对应的层进行一一映射，把其他框架的模型统一全部 转换到tensorRT中，然后在tensorRT中可以针对NVIDIA自家GPU实施优化策略，并进行部署加速。

nvcc -V 查看 CUDA 版本

```py
from tensorflow.python.platform import gfile
import tensorflow.contrib.tensorrt as trt

def load_model(mode_file):
    with gfile.FastGFile(mode_file,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        trt_graph = trt.create_inference_graph(input_graph_def=graph_def,
            outputs=['embeddings:0'],
            max_batch_size = 1,
            max_workspace_size_bytes= 500000000, # 500MB mem assgined to TRT
            precision_mode="FP16",  # Precision "FP32","FP16" or "INT8"                                        
            minimum_segment_size=1
            )
        return trt_graph

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
with sess.as_default():
    graph_load = load_model('./test.pb')

sess.close()
tf.reset_default_graph()
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
tf.import_graph_def(graph_load, input_map=None, name='')

images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

feed_dict = {images_placeholder: np.ones([1, 160, 160, 3]), phase_train_placeholder: False}
sess.run(embeddings, feed_dict=feed_dict)

with sess.as_default():
    sess.run(embeddings, feed_dict=feed_dict)

```
```py
self.graph = tf.Graph()
with self.graph.as_default():
    self.sess = tf.Session()
    with self.sess.as_default():
        # Load pre-trained facenet model
        facenet.load_model(model_path)

    # Get input and output tensors
    self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
```
## trt 加载 insightface 模型对比
  ```py
  import numpy as np
  import tensorflow as tf

  from tensorflow.python.platform import gfile
  import tensorflow.contrib.tensorrt as trt
  def load_model(mode_file):
      with tf.gfile.GFile(mode_file,'rb') as f:
          graph_def = tf.GraphDef()
          graph_def.ParseFromString(f.read())
          trt_graph = trt.create_inference_graph(input_graph_def=graph_def,
              outputs=['fc1/add_1:0'],
              max_batch_size = 1,
              max_workspace_size_bytes= 500000000, # 500MB mem assgined to TRT
              precision_mode="INT8",  # Precision "FP32","FP16" or "INT8"                                        
              minimum_segment_size=1
              )


          print("TensorRT INT8 Enabled and Running INT8 Calib")
          input_map = np.random.random_sample((1, 112, 112, 3))
          inc=tf.constant(input_map, dtype=tf.float32)
          dataset=tf.data.Dataset.from_tensors(inc)
          dataset=dataset.repeat()
          iterator=dataset.make_one_shot_iterator()
          next_element=iterator.get_next()
          out=tf.import_graph_def(trt_graph, input_map={"input":next_element, "phase_train": False}, return_elements=[ "embeddings"])
          self.sess.run(out)
          graph_load=trt.calib_graph_to_infer_graph(trt_graph)

          return trt_graph

  gpu_options = tf.GPUOptions(allow_growth=True)
  sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False))
  with sess.as_default():
      graph_load = load_model('../test.pb')
      tf.import_graph_def(graph_load, input_map=None, name='')

      images_placeholder = tf.get_default_graph().get_tensor_by_name("data:0")
      embeddings = tf.get_default_graph().get_tensor_by_name("fc1/add_1:0")

  sess.run(embeddings, feed_dict={images_placeholder: np.ones([1, 112, 112, 3])})
  %timeit sess.run(embeddings, feed_dict={images_placeholder: np.ones([1, 112, 112, 3])})
  # 35.6 ms ± 256 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

  In [22]: %timeit -n 10 sess.run(embeddings, feed_dict={images_placeholder: np.ones([100, 112, 112, 3])})
  10 loops, best of 5: 224 ms per loop

  In [23]: %timeit -n 10 sess.run(embeddings, feed_dict={images_placeholder: np.ones([100, 112, 112, 3])})
  10 loops, best of 5: 226 ms per loop
  ```
  ```py
  import numpy as np
  import tensorflow as tf

  model_path = 'models/'
  graph = tf.Graph()
  with graph.as_default():
      gpu_options = tf.GPUOptions(allow_growth=True)
      config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True, log_device_placement=False)
      sess = tf.Session(config=config)
      with sess.as_default():
          meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], model_path)

          input_x = sess.graph.get_tensor_by_name("data:0")
          embedding = sess.graph.get_tensor_by_name("fc1/add_1:0")

  sess.run(embedding, feed_dict={input_x: np.ones([100, 112, 112, 3])})
  %timeit sess.run(embedding, feed_dict={input_x: np.ones([1, 112, 112, 3])})
  # 39.1 ms ± 268 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

  In [4]: %timeit -n 10 sess.run(embedding, feed_dict={input_x: np.ones([100, 112, 112, 3])})
  10 loops, best of 5: 249 ms per loop

  In [5]: %timeit -n 10 sess.run(embedding, feed_dict={input_x: np.ones([100, 112, 112, 3])})
  10 loops, best of 5: 256 ms per loop
  ```
## MTCNN
  ```py
  import numpy as np
  import tensorflow as tf

  import tensorflow.contrib.tensorrt as trt

  with tf.gfile.GFile(mode_file,'rb') as f:
      graph_def = tf.GraphDef.FromString(f.read())
      trt_graph = trt.create_inference_graph(input_graph_def=graph_def,
          outputs=['prob', 'landmarks', 'box'],
          max_batch_size = 1,
          max_workspace_size_bytes= 500000000, # 500MB mem assgined to TRT
          precision_mode="INT8",  # Precision "FP32","FP16" or "INT8"                                        
          minimum_segment_size=1
          )
      tf.import_graph_def(trt_graph, name='')

  from mtcnn_tf.mtcnn import MTCNN
  det = MTCNN('./mtcnn_tf/mtcnn.pb')
  det.detect_faces(np.ones([560, 560, 3]))

  In [30]: %timeit -n 10 det.detect_faces(np.ones([560, 560, 3]))
  10 loops, best of 5: 23.2 ms per loop
  10 loops, best of 5: 24.4 ms per loop
  10 loops, best of 5: 23.1 ms per loop
  ```
## Uff
```sh
saved_model_cli show --dir ./tf_resnet100 --all
freeze_graph --input_saved_model_dir tf_resnet100 --output_node_names fc1/add_1 --output_graph ./test.pb

convert-to-uff test.pb
```
```py
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
model_file = './test.uff'

with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.UffParser() as parser:
    parser.register_input("data", (112, 112, 3), trt.UffInputOrder.NHWC)
    parser.register_output("fc1/add_1")
    parser.parse(model_file, network)
```
