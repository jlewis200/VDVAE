#!/usr/bin/python3

"""
Variational Auto Encoder
"""


# pylint: disable=no-member

from argparse import ArgumentParser
from time import time

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize, RandomHorizontalFlip
from PIL import Image


from models import VAE

IMG_SIZE = (128, 128)
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
    parser.add_argument("-l", "--learning-rate", type=float, default=0.0001, help="learning rate of optimizer")
    parser.add_argument("-i", "--iterations", type=int, default=10, help="number of training iterations")

    #pre-trained options
    parser.add_argument("-m", "--model", default="VAE", type=str)

    #experiment options
    parser.add_argument("--reconstruct", type=str, default=[], nargs="+", help="encode/decode an image")
    parser.add_argument("--interpolate", type=str, default=[], nargs=2, help="interpolate between 2 images")
    parser.add_argument("--interpolations", type=int, default=3, help="number of interpolations")
    parser.add_argument("--random", type=int, help="number of random samples")

    args = parser.parse_args()

    model = None

    if args.model == "VAE":
        model = VAE()

    else:
        model = torch.load(args.model)

    if args.train:
        imgs = load_images(args.reconstruct + args.interpolate)
        model = train(model=model,
                      learning_rate=args.learning_rate,
                      beta=args.beta,
                      epochs=args.iterations,
                      imgs=imgs)

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
        imgs = sample(model, args.random)
        montage = get_montage(imgs)
        montage.save("random_montage.jpg")


def load_images(img_paths):
    """
    Lead a list of pathnames, preprocess, return as a tensor with shape:
    N x 3 x H x W
    """
    
    if img_paths != []:
        transform = Compose((ToTensor(), Resize(IMG_SIZE)))
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


def sample(model, n_samples):
    """
    Get a number of random samples from the decoder.
    """

    imgs = []

    if torch.cuda.is_available():
        model = model.cuda()

    for _ in range(n_samples):
        imgs.append(model.sample(1, temp=0.9).clamp(0, 1))

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
    
    model.encode(img)[0]
    return model.decode().clamp(0, 1)


def train(model,
          learning_rate=0.0001,
          beta=0.1,
          epochs=50,
          imgs=None):
    """
    Train the model using supplied hyperparameters.
    """

    start_time = int(time())

    #use random horizontal flip to augment the dataset
    transform = Compose((ToTensor(), Resize(IMG_SIZE), RandomHorizontalFlip()))
    dataset = CelebA("celeba", download=True, transform=transform)
    dataloader = DataLoader(dataset,
                            batch_size=8,
                            shuffle=True,
                            num_workers=2,
                            prefetch_factor=192,
                            drop_last=True)

    if torch.cuda.is_available():
        model = model.cuda()

    #use mixed-precision loss/gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.9))

    for epoch in range(epochs):
        #save the model weights every 10 epochs
        if (1 + epoch) % 1 == 0:
            torch.save(model, f"checkpoints/model_{start_time}_{epoch}")

        epoch_start = time()
        samples = 0

        for batch, _ in dataloader:
            samples += batch.shape[0]

            if imgs is not None:
                batch = torch.cat((batch, imgs))

            if torch.cuda.is_available():
                batch = batch.cuda()

            
            loss, loss_kl, loss_recon =  train_step(model, optimizer, beta, batch, scaler)
            samples_sec = samples / (time() - epoch_start)

            print(f"{epoch} {samples_sec:.2e} {loss:.2e} {loss_kl:.2e} {loss_recon:.2e}")

    return model


def train_step(model, optimizer, beta, batch, scaler):
    """
    Take one training step for a given batch of data.
    """
   
    #auto mixed precision
    with torch.cuda.amp.autocast():

        #project the sample to the latent dimensional space
        model.encode(batch)

        #reconstruct the original sample from the latent dimension representation
        batch_prime = model.decode()

        #get the KL divergence loss from the model
        loss_kl = model.get_loss_kl().mean()

        #calculate the reconstruction loss
        loss_recon = torch.nn.functional.mse_loss(batch_prime, batch)
        
        #sum the losses
        loss = (beta * loss_kl) + loss_recon

    #train the model
    scaler.scale(loss).backward()

    #take an optimization step using the scaled gradients
    scaler.step(optimizer)

    #update the scaler paramters
    scaler.update()
   
    optimizer.zero_grad()
   
    return loss.item(), loss_kl.item(), loss_recon.item()


if __name__ == "__main__":
    main()
