# solarGAN

### module need to install:

- torch>=0.4.0
- torchvision
- matplotlib
- numpy
- scipy
- pillow
- urllib3
- scikit-image

Pretrained model can be downloaded by link :

`https://pan.baidu.com/s/1wW_3dlWMcSgkb4zXGW9azQ`

password :`up7m`

The pretrained model should put in the folder 'saved_models'.

Model testing samples are in the folder 'test'

You can test the pretrained model by runnning

```
python model_test.py
```

the result is in the folder test_result. In addtion, the first image is the dirty image, the middle image is the deconved image and the last one is the ground truth.

If you want to train the model on your own dataset, you can split your data into three folder named train, val and test. 

Run example:

```
python solar_deconv.py --dataset 'your dataset name'
```

Our code is based on the https://github.com/eriklindernoren/PyTorch-GAN

Thanks for the author.
