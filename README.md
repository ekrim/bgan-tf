# bgan-tf
Implementing the Boundary-Seeking GAN in TensorFlow. Credit to the authors of the [original paper](https://arxiv.org/abs/1702.08431v2), who also released their [Theano implementation](https://github.com/rdevon/BGAN).

#### Usage

First, download the celebrity faces (CelebA) dataset, put it in `data/`, and unzip it such that we have `data/img_align_celeba/`.

Then prepare the tfrecords:

```
python make_tfrecords.py
```

Set training params in train.py and run training with:

```
python train.py
```
