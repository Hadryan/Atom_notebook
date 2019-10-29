什么是fine-tuning？
  在实践中，由于数据集不够大，很少有人从头开始训练网络。常见的做法是使用预训练的网络（例如在ImageNet上训练的分类1000类的网络）来重新fine-tuning（也叫微调），或者当做特征提取器。
  以下是常见的两类迁移学习场景：
1 卷积网络当做特征提取器。使用在ImageNet上预训练的网络，去掉最后的全连接层，剩余部分当做特征提取器（例如AlexNet在最后分类器前，是4096维的特征向量）。这样提取的特征叫做CNN codes。得到这样的特征后，可以使用线性分类器（Liner SVM、Softmax等）来分类图像。
2 Fine-tuning卷积网络。替换掉网络的输入层（数据），使用新的数据继续训练。Fine-tune时可以选择fine-tune全部层或部分层。通常，前面的层提取的是图像的通用特征（generic features）（例如边缘检测，色彩检测），这些特征对许多任务都有用。后面的层提取的是与特定类别有关的特征，因此fine-tune时常常只需要Fine-tuning后面的层。
 
预训练模型
在ImageNet上训练一个网络，即使使用多GPU也要花费很长时间。因此人们通常共享他们预训练好的网络，这样有利于其他人再去使用。例如，Caffe有预训练好的网络地址Model Zoo。

  何时以及如何Fine-tune
决定如何使用迁移学习的因素有很多，这是最重要的只有两个：新数据集的大小、以及新数据和原数据集的相似程度。有一点一定记住：网络前几层学到的是通用特征，后面几层学到的是与类别相关的特征。这里有使用的四个场景：
1、新数据集比较小且和原数据集相似。因为新数据集比较小，如果fine-tune可能会过拟合；又因为新旧数据集类似，我们期望他们高层特征类似，可以使用预训练网络当做特征提取器，用提取的特征训练线性分类器。
2、新数据集大且和原数据集相似。因为新数据集足够大，可以fine-tune整个网络。
3、新数据集小且和原数据集不相似。新数据集小，最好不要fine-tune，和原数据集不类似，最好也不使用高层特征。这时可是使用前面层的特征来训练SVM分类器。
4、新数据集大且和原数据集不相似。因为新数据集足够大，可以重新训练。但是实践中fine-tune预训练模型还是有益的。新数据集足够大，可以fine-tine整个网络。
 
实践建议
预训练模型的限制。使用预训练模型，受限于其网络架构。例如，你不能随意从预训练模型取出卷积层。但是因为参数共享，可以输入任意大小图像；卷积层和池化层对输入数据大小没有要求（只要步长stride fit），其输出大小和属于大小相关；全连接层对输入大小没有要求，输出大小固定。
学习率。与重新训练相比，fine-tune要使用更小的学习率。因为训练好的网络模型权重已经平滑，我们不希望太快扭曲（distort）它们（尤其是当随机初始化线性分类器来分类预训练模型提取的特征时）。

```py
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py
--network r100
--loss-type 4
--margin-m 0.5
--prefix ../model-r100
--pretrained '../model-r100, 287' > ../model-r100.log 2>&1 &

I use ArchLoss
Have to discard the weight of fc7 layer, as the number of output is not the same.
@myfighterforever Use deploy/model_slim.py. As is named, this script can slim model. Just in case you need it still.
```
```py
finetune_lr = {k: 0 for k in arg_params}
for key in aux_params:
finetune_lr[key]=0
opt.set_lr_mult(finetune_lr)
#print(aux_params['fc7_weight'])
#print(arg_params.keys())

model.fit(train_dataiter,
    begin_epoch        = begin_epoch,
    num_epoch          = 1,
    eval_data          = val_dataiter,
    eval_metric        = eval_metrics,
    kvstore            = args.kvstore,
    optimizer          = opt,
    #optimizer_params   = optimizer_params,
    initializer        = initializer,
    arg_params         = arg_params,
    aux_params         = aux_params,
    allow_missing      = True,
    batch_end_callback = _batch_callback,
    epoch_end_callback = epoch_cb )
for key in finetune_lr:
    finetune_lr[key]=args.lr
opt.set_lr_mult(finetune_lr)
print('start train the structure')
model.fit(train_dataiter,
          begin_epoch=1,
          num_epoch=100,
          eval_data=val_dataiter,
          eval_metric=eval_metrics,
          kvstore=args.kvstore,
          optimizer=opt,
          # optimizer_params   = optimizer_params,
          initializer=initializer,
          arg_params=arg_params,
          aux_params=aux_params,
          allow_missing=True,
          batch_end_callback=_batch_callback,
          epoch_end_callback=epoch_cb)
like this
```
```py
Delete fc7 weight by calling deploy/model_slim.py
python model_slim.py --model ../models/model-r34-amf/model,0
```
```py
I use the pretrained model https://github.com/deepinsight/insightface/wiki/Model-Zoo#31-lresnet100e-irarcfacems1m-refine-v2 to finetune on the same ms1m-refine-v2 dataset(https://github.com/deepinsight/insightface/wiki/Dataset-Zoo#ms1m-refine-v2recommended)
The accuracy start from 0, is this normal?
The command line is as follows

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r100 --loss-type 4 --margin-m 0.5 --data-dir ../datasets/faces_emore  --prefix ../model-r100 --per-batch-size 64  --pretrained ../pre_trained_models/model-r100-ii/model,0

Try decreasing learning rate and also it is not recommended to fine-tune the same dataset.
```
```py
Hi
I want to finetune your pretrained model (r50) with embedding size 128. Starting point is your train_softmax.py:
CUDA_VISIBLE_DEVICES='0' python -u train_softmax.py --emb-size=128 --pretrained '../models/model-r50-am-lfw/model,0' --network r50 --loss-type 0 --margin-m 0.5 --data-dir ../datasets/faces_ms1m_112x112 --prefix ../models/model-r50-am-lfw/

this results in:
mxnet.base.MXNetError: [08:51:39] src/operator/nn/../tensor/../elemwise_op_common.h:123: Check failed: assign(&dattr, (*vec)[i]) Incompatible attr in node at 0-th output: expected [512], got [128]

I have tried to remove the last fc layer (fc1), then add fc7 as done in your code, then freeze all layers except fc7. Still the dimensions don't match.

Any advise on how to do this? Thanks.

Embedding layer(fc1) is not the last FC layer, but fc7(embedding to number of classes) is. You should train from scratch with different embedding size

train fc1 and fc7 from scratch

I had figured out the same as @zhaowwenzhong and managed to finetune the model by replacing the last two fully connected layers fc1 and fc7 and freezing all others.

@nttstar I want to train from scratch with different embedding size.
1、_weight = mx.symbol.Variable("fc7_weight", shape=(args.num_classes, args.emb_size), lr_mult=1.0, wd_mult=args.fc7_wd_mult)
change to ---->>>_weight = mx.symbol.Variable("fc7_weight_finetune", init = mx.init.Xavier(rnd_type='gaussian', factor_type="out", magnitude=2),shape=(args.num_classes, args.emb_size), lr_mult=1.0, wd_mult=args.fc7_wd_mult)

2、fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7')
change to --->> >fc7 = mx.sym.FullyConnected(data=nembedding, weight = _weight, no_bias = True, num_hidden=args.num_classes, name='fc7_finetune')
```
```py
Appreciate for you goog work! first I train on ms1m dataset until acc is 90% on train dataset. then I wan to finetune on my own data. I have make the *.rec and *.idx. but when I begun to finetune, the program load the pr-trained model, and crash. warning that the number of my dataset classes is not equeal to the ms1m. it seems we should freezen the FC layer.

is your code offer the function of finetuneing?

CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py
--network r100
--loss-type 4
--margin-m 0.5
--prefix ../model-r100
--pretrained '../model-r100, 287' > ../model-r100.log 2>&1 &

I use ArchLoss

Have to discard the weight of fc7 layer, as the number of output is not the same.

@myfighterforever Use deploy/model_slim.py. As is named, this script can slim model.
Just in case you need it still.
```
