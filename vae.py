#!/usr/bin/python3

"""
Variational Auto Encoder
"""


# pylint: disable=no-member

from argparse import ArgumentParser
from time import time

import torch
import numpy as np

from torch.utils.data import DataLoader
from torchvision.datasets import CelebA, CIFAR10
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize, RandomHorizontalFlip, Normalize
from PIL import Image


from models import VAE
from weight_mapper import transfer_weights

GRAD_CLIP = 100

def main():
    """
    Main function to parse command line args and initiate training/experiments.
    """

    # get command line args
    parser = ArgumentParser(description="Perform VAE experiments")

    # options
    parser.add_argument("-t", "--train", action="store_true", help="train the model")
    parser.add_argument("-b", "--beta", type=float, default=1, help="relative weight of KL divergence loss")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.00015, help="learning rate of optimizer")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("-n", "--batch-size", type=int, default=8, help="batch size")

    #pre-trained options
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--config", default="cifar10", type=str)

    #experiment options
    parser.add_argument("--reconstruct", type=str, default=[], nargs="+", help="encode/decode an image")
    parser.add_argument("--interpolate", type=str, default=[], nargs=2, help="interpolate between 2 images")
    parser.add_argument("--interpolations", type=int, default=3, help="number of interpolations")
    parser.add_argument("--random", type=int, help="number of random samples")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of random samples")

    args = parser.parse_args()

    if args.config == "cifar10":
        encoder_layers = [
            {"channels": 384, "n_blocks": 12, "resolution":  32},
            {"channels": 384, "n_blocks":  7, "resolution":  16},
            {"channels": 384, "n_blocks":  7, "resolution":   8},
            {"channels": 384, "n_blocks":  4, "resolution":   4},
            {"channels": 384, "n_blocks":  3, "resolution":   1}]

        decoder_layers = [
            {"channels": 384, "n_blocks":  1, "resolution":   1},
            {"channels": 384, "n_blocks":  3, "resolution":   4},
            {"channels": 384, "n_blocks":  6, "resolution":   8},
            {"channels": 384, "n_blocks": 11, "resolution":  16},
            {"channels": 384, "n_blocks": 22, "resolution":  32}]

        model = VAE(encoder_layers, decoder_layers)
        model.dataset = CIFAR10
        model.dataset_kwargs = {"root": "cifar10", 
                                "download": True, 
                                "transform": Compose((ToTensor(), Resize((32, 32))))}
#        transfer_weights(model)

    elif args.config == "celeba":
#        encoder_layers = [
#            {"channels": 512, "n_blocks":  3, "resolution": 128},
#            {"channels": 512, "n_blocks":  8, "resolution":  64},
#            {"channels": 512, "n_blocks": 12, "resolution":  32},
#            {"channels": 512, "n_blocks": 17, "resolution":  16},
#            {"channels": 512, "n_blocks":  7, "resolution":   8},
#            {"channels": 512, "n_blocks":  5, "resolution":   4, "downsample_ratio": 2},
#            {"channels": 512, "n_blocks":  4, "resolution":   1, "downsample_ratio": 0}]   
#
#        decoder_layers = [
#            {"channels": 512, "n_blocks":  2, "resolution":   1, "upsample_ratio": 0},
#            {"channels": 512, "n_blocks":  3, "resolution":   4, "upsample_ratio": 4},
#            {"channels": 512, "n_blocks":  4, "resolution":   8},
#            {"channels": 512, "n_blocks":  9, "resolution":  16},
#            {"channels": 512, "n_blocks": 21, "resolution":  32},
#            {"channels": 512, "n_blocks": 13, "resolution":  64, "bias": False},
#            {"channels": 512, "n_blocks":  7, "resolution": 128, "bias": False}]

        encoder_layers = [
            {"channels": 512, "n_blocks":  2, "resolution": 128},
            {"channels": 512, "n_blocks":  2, "resolution":  64},
            {"channels": 512, "n_blocks":  2, "resolution":  32},
            {"channels": 512, "n_blocks":  2, "resolution":  16},
            {"channels": 512, "n_blocks":  2, "resolution":   8},
            {"channels": 512, "n_blocks":  2, "resolution":   4},
            {"channels": 512, "n_blocks":  2, "resolution":   2},
            {"channels": 512, "n_blocks":  2, "resolution":   1, "downsample_ratio": 0}]   

        decoder_layers = [
            {"channels": 512, "n_blocks":  2, "resolution":   1, "upsample_ratio": 0},
            {"channels": 512, "n_blocks":  2, "resolution":   2},
            {"channels": 512, "n_blocks":  2, "resolution":   4},
            {"channels": 512, "n_blocks":  2, "resolution":   8},
            {"channels": 512, "n_blocks":  2, "resolution":  16},
            {"channels": 512, "n_blocks":  2, "resolution":  32},
            {"channels": 512, "n_blocks":  2, "resolution":  64},
            {"channels": 512, "n_blocks":  2, "resolution": 128}]
        
        model = VAE(encoder_layers, decoder_layers)
        model.dataset = CelebA
        model.dataset_kwargs = {"root": "celeba", 
                                "download": True, 
                                "transform": Compose((ToTensor(), Resize((128, 128))))}

    else:
        print("unsupported dataset")
        return

    #send model to cuda before initializing optimizer
    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.9))
    
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if args.train:
        #load the dataset only if training
        model.dataset = model.dataset(**model.dataset_kwargs)
        
        model = train(model=model,
                      optimizer=optimizer,
                      beta=args.beta,
                      epochs=args.epochs,
                      batch_size=args.batch_size)

    if args.reconstruct != []:
        imgs = load_images(args.reconstruct)
        imgs = reconstruct(model, imgs)

        for img, filename in zip(imgs, args.reconstruct):
            ToPILImage()(img.squeeze(0)).save(f"reconstructed.jpg")

    if args.interpolate != []:
        img_0, img_1 = load_images(args.interpolate).split(1)
        interpolations = interpolate(model,
                                     img_0,
                                     img_1,
                                     n_interpolations=args.interpolations)

        montage = get_montage([img_0] + interpolations + [img_1])
        filename_0 = args.interpolate[0].split('.')[0]
        filename_1 = args.interpolate[1].split('.')[0]
        montage.save(f"interpolation.jpg")

    if args.random is not None:
        imgs = sample(model, args.random, temp=args.temperature)
        montage = get_montage(imgs)
        montage.save("random_montage.jpg")


def load_images(img_paths):
    """
    Lead a list of pathnames, preprocess, return as a tensor with shape:
    N x 3 x H x W
    """
    
    if img_paths != []:
        #transform = Compose((ToTensor(), Resize(IMG_SIZE), Normalize((0.5, 0.5, 0.5), (1, 1, 1))))
        transform = Compose((ToTensor(), Resize((32, 32))))
        imgs = [transform(Image.open(path).convert("RGB")).unsqueeze(0) for path in img_paths]
        return torch.cat(imgs)


def get_montage(imgs):
    """
    Build a montage from a list of PIL images.
    """

    montage = Image.new("RGB", (imgs[0].shape[-1] * len(imgs), imgs[0].shape[-2]))

    for idx, img in enumerate(imgs):
        img = img.squeeze(0)
        img = ToPILImage()(img)
        montage.paste(img, (idx * imgs[0].shape[-1], 0))

    return montage


def sample(model, n_samples, temp=1.0):
    """
    Get a number of random samples from the decoder.
    """

    imgs = []

    if torch.cuda.is_available():
        model = model.cuda()

    for _ in range(n_samples):
        imgs.append((model.sample(1, temp=temp)).clamp(0, 1))

    return imgs


def interpolate(model, img_0, img_1, n_interpolations=3):
    """
    Perform a linear interpolation between the latent encodings of one image and another.
    """


    if torch.cuda.is_available():
        model = model.cuda()
        img_0 = img_0.cuda()
        img_1 = img_1.cuda()

    interpolations = []
    model.eval()

    img_0, _ = model.encode(img_0)
    img_1, _ = model.encode(img_1)

    for idx in range(0, 1 + n_interpolations):
        ratio = idx / n_interpolations
        img = (img_0 * (1 - ratio)) + (img_1 * ratio)

        #upsample the latent interpolation and clamp to color channel range
        interpolations.append(model.decode(img).clamp(0, 1))
    

    return interpolations


def reconstruct(model, img):
    """
    Encode/decode an image.  Return the reconstructed tensor.
    """

    if torch.cuda.is_available():
        model = model.cuda()
        img = img.cuda()

    model.eval()
    
    return model.reconstruct(img)

def train(model,
          optimizer,
          beta=1.0,
          epochs=50,
          batch_size=32):
    """
    Train the model using supplied hyperparameters.
    """

    dataloader = DataLoader(model.dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            prefetch_factor=batch_size,
                            drop_last=True)

    #use mixed-precision loss/gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    for _ in range(epochs):
        #print column headers
        print(f"{'':>9} {'sample/':>9} {'gradient':>9} {'overall':>9} {'KL':>9} {'mixture':>9}")
        print(f"{'epoch':>9} {'sec':>9} {'norm':>9} {'loss':>9} {'div':>9} {'NLL':>9}") 

        model.epoch += 1
        epoch_start = time()
        samples = 0

        for batch, _ in dataloader:
            samples += batch.shape[0]

            if torch.cuda.is_available():
                batch = batch.cuda()
            
            loss, loss_kl, loss_nll, grad_norm =  train_step(model, optimizer, beta, batch, scaler)
            samples_sec = samples / (time() - epoch_start)

            print(f"{model.epoch.item():9} {samples_sec: 9.2e} {grad_norm: 9.2e} {loss: 9.2e} {loss_kl: 9.2e} {loss_nll: 9.2e}")

        #save the model weights
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, 
                    f"checkpoints/model_{model.start_time.item()}_{model.epoch.item()}")

    return model


def train_step(model, optimizer, beta, batch, scaler):
    """
    Take one training step for a given batch of data.
    """
  
    try:
        #auto mixed precision
        with torch.cuda.amp.autocast():

            #project the sample to the latent dimensional space
            activations = model.encode(batch)
            
            #reconstruct the original sample from the latent dimension representation
            #batch_prime = model.decode(activations)
            tensor = model.decode(activations)
            loss_nll = model.get_nll(tensor, batch).mean()

            #get the KL divergence loss from the model
            loss_kl = model.get_loss_kl().mean()

            #calculate the reconstruction loss
            #loss_recon = torch.nn.functional.mse_loss(batch_prime, batch)
            
            #sum the losses
            loss = (beta * loss_kl) + loss_nll

        #train the model
        scaler.scale(loss).backward()
    
        #unscale to apply gradient clipping
        scaler.unscale_(optimizer)

        #find the gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        #take the optimizer step if gradients are under control
        if grad_norm < GRAD_CLIP:

            #take an optimization step using the gradient scaler
            scaler.step(optimizer)

        #update the scaler paramters
        scaler.update()
       
        optimizer.zero_grad()
       
        return loss.item(), loss_kl.item(), loss_nll.item(), grad_norm.item()

    except ValueError:
        #the decoder may sporadically throw a value error when creating the p distribution
        #if this happens abort the training step
        return torch.inf, torch.inf, torch.inf, torch.inf


if __name__ == "__main__":
    main()
