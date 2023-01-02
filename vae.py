#!/usr/bin/python3

"""
Variational Auto Encoder
"""


import math
from argparse import ArgumentParser
from time import time

# pylint: disable=no-member
import torch

from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
from configs import get_model

GRAD_CLIP = 200
EMA_SCALER = 0.99

def main():
    """
    Main function to parse command line args and initiate training/experiments.
    """

    # get command line args
    parser = ArgumentParser(description="Perform VAE experiments")

    #options
    parser.add_argument("-t", "--train", action="store_true", help="train the model")
    parser.add_argument("-b", "--beta", type=float, default=1, help="relative weight of KL divergence loss")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.00015, help="learning rate of optimizer")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("-n", "--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("-m", "--mixture-net-only", action="store_true", help="only train the mixture net")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="torch device string")

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

    model = get_model(args.config)

    if model is None:
        print("config not found")
        return

    #send model to target device before initializing optimizer
    model = model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.9))

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        #override the loaded learning rate with that specified in the args
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate

    if args.train:
        #load the dataset only if training
        model.dataset = model.get_dataset()

        model = train(model=model,
                      optimizer=optimizer,
                      beta=args.beta,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      mixture_net_only=args.mixture_net_only)

    if args.reconstruct != []:
        imgs = load_images(args.reconstruct)
        imgs = reconstruct(model, imgs)

        for img, filename in zip(imgs, args.reconstruct):
            to_pil_image(img.squeeze(0)).save("reconstructed.jpg")

    if args.interpolate != []:
        img_0, img_1 = load_images(args.interpolate).split(1)
        interpolations = interpolate(model,
                                     img_0,
                                     img_1,
                                     n_interpolations=args.interpolations)

        montage = get_montage([img_0] + interpolations + [img_1])
        montage.save("interpolation.jpg")

    if args.random is not None:
        imgs = sample(model, args.random, temp=args.temperature)
        montage = get_montage(imgs)
        montage.save("random_montage.jpg")


def load_images(img_paths):
    """
    Lead a list of pathnames, preprocess, return as a tensor with shape:
    N x 3 x H x W
    """

    tensor = None

    if img_paths != []:
        imgs = [to_tensor(Image.open(path).convert("RGB")).unsqueeze(0) for path in img_paths]
        tensor = torch.cat(imgs)

    return tensor


def get_montage(imgs):
    """
    Build a montage from a list of PIL images.
    """

    montage = Image.new("RGB", (imgs[0].shape[-1] * len(imgs), imgs[0].shape[-2]))

    for idx, img in enumerate(imgs):
        img = img.squeeze(0)
        img = to_pil_image(img)
        montage.paste(img, (idx * imgs[0].shape[-1], 0))

    return montage


def sample(model, n_samples, temp=1.0):
    """
    Get a number of random samples from the decoder.
    """

    with torch.no_grad():
        return [(model.sample(1, temp=temp)).clamp(0, 1) for _ in range(n_samples)]


def interpolate(model, img_0, img_1, n_interpolations=3):
    """
    Perform a linear interpolation between the latent encodings of one image and another.
    """

    img_0 = img_0.to(model.device)
    img_1 = img_1.to(model.device)

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

    with torch.no_grad():
        return model.reconstruct(img.to(model.device))

def train(model,
          optimizer,
          beta=1.0,
          epochs=50,
          batch_size=32,
          mixture_net_only=False):
    """
    Train the model using supplied hyperparameters.
    train_enc_dec controls whether the encoder and decoder should be trained, or only the dmll net.
    """

    dataloader = DataLoader(model.dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            prefetch_factor=batch_size,
                            drop_last=True)

    #initialize exponential moving averages
    loss_ema, loss_kl_ema, loss_nll_ema, grad_norm_ema = (torch.inf,) * 4

    #initialize mixed-precision loss/gradient scaler
    scaler = torch.cuda.amp.GradScaler()

    for _ in range(epochs):
        #print column headers
        print(f"{'':>9} {'sample/':>9} {'gradient':>9} {'overall':>9} {'KL':>9} {'mixture':>9}")
        print(f"{'epoch':>9} {'sec':>9} {'norm':>9} {'loss':>9} {'div':>9} {'NLL':>9}")

        samples = 0
        epoch_start = time()

        for batch, *_ in dataloader:
            batch = batch.to(model.device)

            stats = train_step(model, optimizer, beta, batch, scaler, mixture_net_only)
            loss, loss_kl, loss_nll, grad_norm = stats

            if all(not math.isnan(stat) and \
                   stat not in (torch.nan, torch.inf, -torch.inf) for stat in stats):
                #update running stats only if no extreme values
                samples += batch.shape[0]
                loss_ema = ema_update(loss_ema, loss)
                loss_kl_ema = ema_update(loss_kl_ema, loss_kl)
                loss_nll_ema = ema_update(loss_nll_ema, loss_nll)
                grad_norm_ema = ema_update(grad_norm_ema, grad_norm)

            samples_sec = samples / (time() - epoch_start)

            print(f"{model.epoch.item():9} "
                  f"{samples_sec: 9.2e} "
                  f"{grad_norm: 9.2e} "
                  f"{loss: 9.2e} "
                  f"{loss_kl: 9.2e} "
                  f"{loss_nll: 9.2e}")

            print(f"{'':9} "
                  f"{' EMAs':9} "
                  f"{grad_norm_ema: 9.2e} "
                  f"{loss_ema: 9.2e} "
                  f"{loss_kl_ema: 9.2e} "
                  f"{loss_nll_ema: 9.2e}")

            print()

            if time() - epoch_start > 3600:
                #save the model weights
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()},
                            f"checkpoints/model_{model.start_time.item()}_{model.epoch.item()}.pt")
                epoch_start = time()
                samples = 0

        #save the model weights
        model.epoch += 1
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()},
                    f"checkpoints/model_{model.start_time.item()}_{model.epoch.item()}.pt")
        print()

    return model


def train_step(model, optimizer, beta, batch, scaler, mixture_net_only):
    """
    Take one training step for a given batch of data.
    """

    try:
        #auto mixed precision
        with torch.cuda.amp.autocast():

            #don't train the encoder/decoder if mixture net only is set
            with torch.set_grad_enabled(not mixture_net_only):
                #encode/decode the sample
                dec_out = model.forward(batch)

            #get the KL divergence loss from the model
            loss_kl = model.get_loss_kl().mean()

            #get the negative log likelihood from the mixture net
            loss_nll = model.get_nll(dec_out, batch).mean()

            #sum the losses
            loss = (beta * loss_kl) + loss_nll

        #calculate gradients using gradient scaler
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


def ema_update(ema, val):
    """
    Perform an ema update given a new value.
    """

    if ema is torch.inf:
        #ema is initialized with an inf value
        ema = val

    else:
        ema = (ema * EMA_SCALER) + ((1 - EMA_SCALER) * val)

    return ema


if __name__ == "__main__":
    main()
