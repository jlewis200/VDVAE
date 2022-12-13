#!/usr/bin/python3

# pylint: disable=no-member
import torch

from models import Encoder, Decoder
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA 
from torchvision.transforms import ToTensor, ToPILImage, Compose, Resize, RandomHorizontalFlip
from argparse import ArgumentParser
from PIL import Image
from time import time
from code import interact

def main():
    # get command line args
    parser = ArgumentParser(description="Perform VAE experiments")

    # options
    mutually_exclusive = parser.add_mutually_exclusive_group()
    mutually_exclusive.add_argument("-t", "--train", action="store_true", help="apply rotation regularization")
    mutually_exclusive.add_argument("-p", "--pre-trained", action="store_true", help="apply rotation regularization")
   
    #training options
    parser.add_argument("-b", "--beta", type=float, default=1, help="relative weight of KL divergence loss")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.0001, help="learning rate of optimizer")
    parser.add_argument("-i", "--iterations", type=int, default=50, help="number of training iterations")

    #pre-trained options
    parser.add_argument("-e", "--encoder", type=str, default="checkpoints/encoder/encoder.pt")
    parser.add_argument("-d", "--decoder", type=str, default="checkpoints/decoder/decoder.pt")

    #experiment options
    parser.add_argument("-r", "--reconstruct", type=str, help="encode/decode an image")
    parser.add_argument("-q", "--interpolate", type=str, nargs=2, help="interpolate between 2 images")
    parser.add_argument("-z", "--interpolations", type=int, default=3, help="interpolate between 2 images")
    parser.add_argument("-o", "--out", type=str, default="out.jpg", help="output file name")

    args = parser.parse_args()
    
    if args.train:
        encoder, decoder = train(lr=args.learning_rate, beta=args.beta, epochs=args.iterations)

    else:
        encoder = torch.load(args.encoder)
        decoder = torch.load(args.decoder)

    if args.reconstruct is not None:
        img = Image.open(args.reconstruct).convert("RGB")
        transform = Compose((ToTensor(), Resize((128, 128))))
        img = transform(img)
        img = img.unsqueeze(0)
        img = reconstruct(encoder, decoder, img)
        img = img.squeeze(0)
        ToPILImage()(img).save(args.out)
    
    elif args.interpolate is not None:
        transform = Compose((ToTensor(), Resize((128, 128))))
        
        img_0 = Image.open(args.interpolate[0]).convert("RGB")
        img_0 = transform(img_0).unsqueeze(0)

        img_1 = Image.open(args.interpolate[1]).convert("RGB")
        img_1 = transform(img_1).unsqueeze(0)

        interpolations = interpolate(encoder, decoder, img_0, img_1, n_interpolations=args.interpolations)
        interpolations = [img_0] + interpolations + [img_1]

        montage = Image.new("RGB", (img_0.shape[-1] * len(interpolations), img_0.shape[-2]))

        for idx, img in enumerate(interpolations):
            img = img.squeeze(0)
            img = ToPILImage()(img)
            montage.paste(img, (idx * img_0.shape[-1], 0))

        montage.save(args.out)
        


def interpolate(encoder, decoder, img_0, img_1, n_interpolations=3):
    """
    """


    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        img_0 = img_0.cuda()
        img_1 = img_1.cuda()
 
    interpolations = []
    
    encoder.eval()
    decoder.eval()
   
    for idx in range(0, 1 + n_interpolations):
        ratio = idx / n_interpolations

        img = (img_0 * (1 - ratio)) + (img_1 * ratio)

        interpolations.append(decoder.forward(encoder.forward(img)[0]))

    return interpolations


def reconstruct(encoder, decoder, img):
    """
    """

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()
        img = img.cuda()
 
    encoder.eval()
    decoder.eval()
    
    return decoder.forward(encoder.forward(img)[0])


def train(encoder=None, decoder=None, lr=0.0001, beta=0.1, epochs=50):
    """
    """

    size = 256

    #use random horizontal flip to augment the dataset
    transform = Compose((ToTensor(), Resize((128, 128)), RandomHorizontalFlip()))
    dataset = CelebA("celeba", download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=192, shuffle=True, num_workers=2, prefetch_factor=192)

    if encoder is None:
        encoder = Encoder()
    
    if encoder is None:
        decoder = Decoder()

    if torch.cuda.is_available():
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr) 

    start_time = int(time())

    for epoch in range(epochs):
        #save the model weights every 10 epochs
        if (1 + epoch) % 10 == 0:
            torch.save(encoder, f"checkpoints/encoder/{start_time}_{epoch}")
            torch.save(decoder, f"checkpoints/decoder/{start_time}_{epoch}")

        for batch, label in dataloader:
            if torch.cuda.is_available():
                batch = batch.cuda()

            #project the sample to the latent dimensional space
            mu, sigma = encoder.forward(batch)
           
            #get reparameterized sample using mu as the mean and sigma as the standard deviation
            std = torch.exp(sigma / 2)
            q = torch.distributions.Normal(mu, std)
            
            #reparameterized sample from N(mu, std)
            z = q.rsample()
        
            #reconstruct the original sample from the latent dimension representation
            batch_prime = decoder.forward(z)
     
            #generate the target normal distribution of the appropriate shape
            p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(mu))
          
            #calculate the KL divergence between N(0, 1) and N(mu, std)
            loss_kl = (q.log_prob(z) - p.log_prob(z)).mean()

            #calculate the reconstruction loss
            loss_recon = torch.nn.functional.mse_loss(batch_prime, batch)

            loss = beta * loss_kl + loss_recon

            #train the model weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            print(f"{epoch} {loss.item():.5e} {loss_kl.item():.5e} {loss_recon.item():.5e}")
        
    return encoder, decoder


if __name__ == "__main__":
    main()
