#!/usr/bin/python3

import gdown

try:
    #cifar10 pretrained
    print("attempting to download cifar10 pretrained weights")
    
    url = "https://drive.google.com/file/d/1P7qVBMbyQdtO4gQSgxolqB-iwuzTXwUU/view?usp=sharing"
    destination = "checkpoints/cifar10_pretrained.pt"
    gdown.download(url, destination, quiet=False, fuzzy=True)
    print()

except FileNotFoundError:
    print("missing 'checkpoints' directory\n")

except RuntimeError:
    print("unable to download cifar10 pretrained weights\n")

try:
    #ffh1256 pretrained
    print("attempting to download ffhq256 pretrained weights")
    
    url = "https://drive.google.com/file/d/1Xlwm4i-PI_aSahSWU517afqDYv2dCGJE/view?usp=sharing"
    destination = "checkpoints/ffhq256_pretrained.pt"
    gdown.download(url, destination, quiet=False, fuzzy=True)
    print()

except FileNotFoundError:
    print("missing 'checkpoints' directory\n")

except RuntimeError:
    print("unable to download ffhq256 pretrained weights\n")
