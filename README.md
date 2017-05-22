# CitSci2017_TensorflowRetrainer
CitSci2017 Conference: Create Together Day:

Project to use Tensorflow to train a model &amp; use it in wq app

Basic idea is outlined here:

https://www.tensorflow.org/tutorials/image_retraining

## Integration in wq

Here all infos regarding the wq app:

https://wq.io/

https://github.com/wq/wq


## Steps to set-up an AWS Ubuntu 16 instance with GPU, docker and tensorflow.

Some of the sources I have used:

https://petewarden.com/2016/09/27/tensorflow-for-mobile-poets/ 

https://medium.com/@daj/creating-an-image-classifier-on-android-using-tensorflow-part-3-215d61cb5fcd

https://gist.github.com/espdev/a205ec97f650c6695104c3401f7fd9c2

https://alliseesolutions.wordpress.com/2016/09/08/install-gpu-tensorflow-from-sources-w-ubuntu-16-04-and-cuda-8-0/

Set-up a new AWS Ubuntu 16 instance and execute following steps:

```
Part1_install.sh
```

Then there is a manual step to get cuDNN:
```
Part2_install_MANUAL.sh
```

Finally:
```
Part3_install.sh
```

## Code to get data from wq server, re-train googles inception model, and upload to wq server

```
python3 main.py 2 some_description
```

2: is the id of the dataset from the json db (the dataset must be a zip-file of n folders containing images of n classes)

some description: the description of the final model


