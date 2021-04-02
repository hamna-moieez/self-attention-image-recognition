# self-attention-image-recognition
A tensorflow implementation of pair-wise and patch-wise self attention network for image recognition.

## Train
1. Requirements:
+ Python >= 3.6
+ Tensorflow >= 2.0.0
2. To train the SANet on your own dataset, you can put the dataset under the folder **dataset**, and the directory should look like this:
```
|——dataset 
   |——train
      |——class_name
   |——test
      |——class_name
```
3. Change the corresponding parameters in **config.py**.
5. Run **trainer.py** to start training.

### Note
The implementation is supported only for GPU training because of the image format specific limitation of tensorflow.
