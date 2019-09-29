
- [论文 - Learning both Weights and Connections for Efficient Neural Networks](https://xmfbit.github.io/2018/03/14/paper-network-prune-hansong/)
- [Github Network Slimming (Pytorch)](https://github.com/Eric-mingjie/network-slimming)
- [Distiller Documentationhttps://nervanasystems.github.io/distiller/index.html)
- [NPU使用示例](http://ai.nationalchip.com/docs/gx8010/npukai-fa-zhi-nan/shi-li.html)
- [Github pruning_with_keras](https://github.com/tensorflow/model-optimization/blob/master/tensorflow_model_optimization/g3doc/guide/pruning/pruning_with_keras.ipynb)
- [](https://www.tensorflow.org/lite/models/segmentation/overview)
- [优化机器学习模型](https://www.tensorflow.org/model_optimization)
- [TensorFlow C++ and Python Image Recognition Demo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image)
- [Issue with converting Keras .h5 files to .tflite files](https://github.com/tensorflow/tensorflow/issues/20878)
- [TensorFlow Lite converter](https://www.tensorflow.org/lite/convert)

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
