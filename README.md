# Docker使用
* docker检查
```sh
$ docker ps -a
```
* docker启动
```sh
docker start -ai ${docker.name}
```

```sh
# (例)
docker start -ai pytorch_2.1.0
```

* docker进入
```sh
docker exec -it pytorch_2.1.0 bash
```

* docker复制文件
```sh
$ docker cp ${docker.name}:filepath ToFilePath
```

```sh
# (例)
$ docker cp fc5bad9f9c64:/workspace/Transformer_ViT/tensorboard /home/huchenyu
```

# 分类模型

| 数据集 | 模型   | 参量|
|  ---  |  ---   | --- |
|cifar100|VGG19|39M|
|cifar100|ResNet50|23M|
|cifar100|Inception-v1|6M|
|cifar100|Inception-v4|41M|
|cifar100|SqueezeNet1.0|0.78M|
|cifar100|DenseNet161|26M|

```sh
$ cd pytorch-cifar100/
```
* 训练过程
```sh
$ python train.py -net ${model.name} -gpu
(例)
# -b 为批大小，若无内存问题，可以使用默认128
$ python train.py -net vgg19 -gpu
$ python train.py -net resnet50 -gpu
$ python train.py -net googlenet -gpu
$ python train.py -net inceptionv4 -gpu -b 64
$ python train.py -net squeezenet -gpu
$ python train.py -net densenet161 -gpu -b 64
$ python train.py -net cascadedfcn -gpu -b 8 -dataset VOCSegmentation
$ python train.py -net voc -gpu -b 8 -dataset VOCDetection
```

* 测试过程
```sh
$ python test.py -net ${model.name} -weights ${path_weights_file}
(例)
$ python test.py -net vgg19 -weights checkpoint/vgg19/Thursday_30_May_2024_05h_09m_34s/vgg19-200-regular.pth
$ python test.py -net resnet50 -weights checkpoint/resnet50/Thursday_30_May_2024_10h_13m_25s/resnet50-200-regular.pth
$ python test.py -net cascadedfcn -weights checkpoint/cascadedfcn/Thursday_27_June_2024_11h_49m_48s/cascadedfcn-10-regular.pth -b 8 -dataset VOCSegmentation
```

# Transformer_ViT

```sh
$ cd Transformer_ViT/
```

* 训练过程
```sh
# 当保存数据集在 dataset\dog_vs_cat_dataset_n100 中
$ python train.py \
--exper_name Transformer_train \
--dataset_dir dataset/dog_vs_cat_dataset_n100

```

* 测试过程
```sh
$ python test.py \
--exper_name Transformer_test \
--dataset_dir ${DATASET_DIR} \
--load_checkpoints_path ${LOAD_CHECKPOINTS_PATH}
```

```sh
# (例)
$ python test.py \
--exper_name Transformer_test \
--dataset_dir dataset/dog_vs_cat_dataset_n100 \
--load_checkpoints_path checkpoints/Transformer_train
```