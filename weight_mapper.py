#!/usr/bin/python3
import torch

def transfer_weights(model):
    
    for parameter in model.parameters():
        #TODO:  don't zero the mixture_net
        parameter.data[:] = 0

    breakpoint()


    vdvae = torch.load("vdvae_checkpoints/cifar10-seed3-iter-1050000-model-ema.th")

    #misc
    model.encoder.in_conv.weight.data = state_dict["encoder.in_conv.weight"]
    model.encoder.in_conv.bias.data = state_dict["encoder.in_conv.bias"]

    #encoder blocks
    idx = 0

    for encoder_group in model.encoder.encoder_groups:
        for encoder_block in encoder_groups:

            jdx = 0

            while jdx < 4:
                encoder_block[(jdx * 2) + 1].weight.data = state_dict[f"encoder.enc_blocks.{idx}.c{jdx + 1}.weight"]
                encoder_block[(jdx * 2) + 1].bias.data   = state_dict[f"encoder.enc_blocks.{idx}.c{jdx + 1}.bias"]

                jdx += 1

            idx += 1

    #decoder blocks
    idx = 0

    for decoder_group in model.decoder.decoder_groups:

        try:
            decoder_group.bias = state_dict[f"decoder.bias_xs.{idx}"]

        except KeyError:
            print(f"no bias for decoder_group:  {idx}")


        for decoder_block in decoder_groups:
            
            jdx = 0

            while jdx < 4:
                decoder_block.phi[(jdx * 2) + 1].weight.data = state_dict[f"encoder.dec_blocks.{idx}.enc.c{jdx + 1}.weight"]
                decoder_block.phi[(jdx * 2) + 1].bias.data   = state_dict[f"encoder.dec_blocks.{idx}.enc.c{jdx + 1}.bias"]

                decoder_block.theta[(jdx * 2) + 1].weight.data = state_dict[f"encoder.dec_blocks.{idx}.prior.c{jdx + 1}.weight"]
                decoder_block.theta[(jdx * 2) + 1].bias.data   = state_dict[f"encoder.dec_blocks.{idx}.prior.c{jdx + 1}.bias"]

                decoder_block.z_projection[(jdx * 2) + 1].weight.data = state_dict[f"encoder.dec_blocks.{idx}.z_proj.c{jdx + 1}.weight"]
                decoder_block.z_projection[(jdx * 2) + 1].bias.data   = state_dict[f"encoder.dec_blocks.{idx}.z_proj.c{jdx + 1}.bias"]

                decoder_block.res[(jdx * 2) + 1].weight.data = state_dict[f"encoder.dec_blocks.{idx}.resnet.c{jdx + 1}.weight"]
                decoder_block.res[(jdx * 2) + 1].bias.data   = state_dict[f"encoder.dec_blocks.{idx}.resnet.c{jdx + 1}.bias"]
                
                jdx += 1

            idx += 1


