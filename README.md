# cnn_autoencoder
Convolutional neural network Autoencoder.
Implementation based on Cheng *et al.* **Energy Compaction-Based Image Compression Using Convolutional AutoEncoder**, Transactions on Multimedia 14 (8), 2019.

## Installation
Clone this repository and use the [pytorch_lightnening](https://hub.docker.com/r/pytorchlightning/pytorch_lightning) container.
The extra required packages can be installed using **pip** from  *requirements.txt*.

## Training
Training has been tested on MNIST, ImageNet, and a local histology dataset.

### Configuration files
There are a set of configuration files that can be used to set up the training parameters.
Those can be used by passing the argumen ```-c config.json```, where config.json is a json file containing the parameters of the experiment.
The training paramaters can be reviewed by using the following command.

```
python ./src/train_cae.py -h
```

## Testing
The trained model can be tested using the *test_cae_mode.py* script, or using the modules *src/compress.py*, and/or *src/decompress.py* by separate.

### Compressing and decompressing
Compression and decompression requires a pre-trained model.
Other arguments required to run these modules can be listed with the fllowing commands.

```
python ./test_cae_model.py -h
python ./src/compress.py -h
python ./src/compress.py -h
```
