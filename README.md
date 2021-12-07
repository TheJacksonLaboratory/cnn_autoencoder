# cnn_autoencoder
Convolutional neural network Autoencoder.
Implementation based on Cheng *et al.* **Energy Compaction-Based Image Compression Using Convolutional AutoEncoder**, Transactions on Multimedia 14 (8), 2019.

## Training
Training has been tested on MNIST, ImageNet, and a local histology dataset.

### Configuration files
There are a set of configuration files that can be used to set up the training parameters.
Training paramaters can be reviewed by using the following command.

```
python ./src/train.py -h
```

## Testing
The trained model can be tested using the *test_mode.py* script, or using the modules *src/compress.py*, and/or *src/decompress.py* by separate.
