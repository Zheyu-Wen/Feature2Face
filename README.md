# Feature2Face

[![Build Status](https://travis-ci.com/CoderNoMercy/Feature2Face.svg?branch=master)](https://travis-ci.com/CoderNoMercy/Feature2Face)

Generate steady face from wav data
Language: python 3.6. 

Method: 

1.Vgg-face net to extract human face from vedio.

2.Establish net to learn face from audio data.

3.(this repo) Use rectified style-GAN to generate face from wav data

4.VGG lip learning from audio

# TODO list:

- [x] Download more data from YouTube and extract face(via vgg) and audio in order to get more advanced result.
- [x] Use other kind of net to generature face (may combined with optimization method)
- [x] adjust GAN network to add noise adquately in order to add more high frequency signal

# How to run

The project has been divided into several part. In this repo you can train GAN to learn face from face feature. Run main.py to begin training. Data has been added to this repo.
```python
python main.py
```

# File Instruction:
F2FDataloader.py is the file that build data loader for this part

F2FDiscriminator.py is network of discriminator in GAN

F2FGenerator.py is the network of generator in GAN

Inference_Evaluation.py is validation or test file that can test your trained network.

main.py just run it you can complete your trial

train_gan.py function for trianing network you have built by using the data from dataloader.
