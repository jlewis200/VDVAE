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
SAVE_INTERVAL = 3600


def get_args(args=None):
    """
    Parse command line options.
    """

    # get command line args
    parser = ArgumentParser(description="Perform VAE experiments")

    #options
    parser.add_argument("-t", "--train", action="store_true", help="train the model")
    parser.add_argument("-l", "--learning-rate", type=float, default=0.00015, help="learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=50, help="number of training epochs")
    parser.add_argument("-n", "--batch-size", type=int, default=8, help="batch size")
    parser.add_argument("-m", "--mixture-net-only", action="store_true", help="only train mixture net")
    parser.add_argument("-d", "--device", type=str, default="cuda:0", help="torch device string")

    #pre-trained options
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--config", default="cifar10", type=str)

    #experiment options
    parser.add_argument("--reconstruct", type=str, help="encode/decode an image")
    parser.add_argument("--sample", type=int, help="number of samples")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature of samples")

    return parser.parse_args(args=args)


def main(args):
    """
    Main function to initiate training/experiments.
    """

    model = get_model(args.config)

    #ensure a valid config was specified
    if model is None:
        print("config not found")
        return

    #try sending a tensor to specified device to check validity/availability
    try:
        torch.zeros(1, device=args.device)

    except RuntimeError:
        print(f"inavlid/unavailable device specified:  {args.device}")
        return

    #send model to target device before initializing optimizer
    model = model.to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.9))

    if args.checkpoint is not None:
        #load the model/optimizer state dicts
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        #override the loaded learning rate with that specified in the args
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.learning_rate

    if args.train:
        model = train(model=model,
                      optimizer=optimizer,
                      epochs=args.epochs,
                      batch_size=args.batch_size,
                      mixture_net_only=args.mixture_net_only)

    if args.reconstruct is not None:
        reconstruct(model, args.reconstruct)

    if args.sample is not None:
        sample(model, args.sample, temp=args.temperature)


def load_image(path):
    """
    Lead a path, preprocess, return as a tensor with shape:
    1 x 3 x H x W
    """

    return to_tensor(Image.open(path).convert("RGB")).unsqueeze(0)


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
    Sample from the model.
    """

    with torch.no_grad():
        imgs = [(model.sample(temp=temp)).clamp(0, 1) for _ in range(n_samples)]

    get_montage(imgs).save("samples.jpg")


def reconstruct(model, path):
    """
    Encode/decode an image.  Save the reconstructed image.
    """

    try:
        with torch.no_grad():
            img = model.reconstruct(load_image(path).to(model.device))

        to_pil_image(img.squeeze(0)).save("reconstructed.jpg")

    except FileNotFoundError:
        print(f"image not found:  {path}")


def train(model,
          optimizer,
          epochs=50,
          batch_size=32,
          mixture_net_only=False):
    """
    Train the model using supplied hyperparameters.
    train_enc_dec controls whether the encoder and decoder should be trained, or only the dmll net.
    """

    #load the dataset
    model.dataset = model.get_dataset()
    dataloader = DataLoader(model.dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=2,
                            prefetch_factor=batch_size,
                            drop_last=True)

    #initialize exponential moving averages
    loss_ema, loss_kl_ema, loss_nll_ema, grad_norm_ema = (torch.inf,) * 4

    if "cuda" in model.device.type:
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

            if model.device.type == "cuda":
                #use the CUDA-specific training step if the model is on CUDA
                stats = train_step_cuda(model, optimizer, batch, scaler, mixture_net_only)

            else:
                #if not CUDA, use the generic training step
                stats = train_step(model, optimizer, batch, mixture_net_only)

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

            #save periodically throughout the epoch
            if time() - epoch_start > SAVE_INTERVAL:
                epoch_start = time()
                samples = 0
                save_state(model, optimizer)

        #save at the end of every epoch
        save_state(model, optimizer)
        model.epoch += 1
        print()

    return model


def train_step(model, optimizer, batch, mixture_net_only):
    """
    Take one training step for a given batch of data.
    """

    #don't train the encoder/decoder if mixture net only is set
    with torch.set_grad_enabled(not mixture_net_only):
        #encode/decode the sample
        dec_out = model.forward(batch)

    #get the KL divergence loss from the model
    loss_kl = model.get_loss_kl().mean()

    #get the negative log likelihood from the mixture net
    loss_nll = model.get_nll(dec_out, batch).mean()

    #sum the losses
    loss = loss_kl + loss_nll

    #calculate gradients
    loss.backward()

    #find the gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

    #take the optimizer step if gradients are under control
    if grad_norm < GRAD_CLIP:
        #take an optimization step using the gradient scaler
        optimizer.step()

    optimizer.zero_grad()

    return loss.item(), loss_kl.item(), loss_nll.item(), grad_norm.item()


def train_step_cuda(model, optimizer, batch, scaler, mixture_net_only):
    """
    Take one training step for a given batch of data.
    Use CUDA-specific FP16 optimizations.
    """

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
        loss = loss_kl + loss_nll

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


def save_state(model, optimizer):
    """
    Save the model/optimizer state dict.
    """

    torch.save({"model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()},
                f"checkpoints/model_{model.start_time.item()}_{model.epoch.item()}.pt")


if __name__ == "__main__":
    main(get_args())
