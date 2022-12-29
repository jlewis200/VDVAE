#!/usr/bin/python3
import torch
from models import VAE

def transfer_weights(model, save_path, donor_weights):

    state_dict = torch.load(donor_weights)
    
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        #ensure all weights have been cleared
        for parameter in model.parameters():
            assert(parameter.sum() == 0)

        #reinitialize the mixture_net weights
        model.decoder.mixture_net.dist_r_0.reset_parameters()
        model.decoder.mixture_net.dist_r_1.reset_parameters()
        model.decoder.mixture_net.dist_g_0.reset_parameters()
        model.decoder.mixture_net.dist_g_1.reset_parameters()
        model.decoder.mixture_net.dist_b_0.reset_parameters()
        model.decoder.mixture_net.dist_b_1.reset_parameters()
        model.decoder.mixture_net.mix_score.reset_parameters()

        #reinitialize the out_conv weights
        model.decoder.out_conv.reset_parameters()

        #misc
        transfer_item(model.encoder.in_conv.weight, state_dict, "encoder.in_conv.weight")
        transfer_item(model.encoder.in_conv.bias, state_dict, "encoder.in_conv.bias")
        transfer_item(model.decoder.out_net.out_conv.weight, state_dict, "decoder.out_net.out_conv.weight")
        transfer_item(model.decoder.out_net.out_conv.bias, state_dict, "decoder.out_net.out_conv.bias")


        #encoder blocks
        idx = 0

        for encoder_group in model.encoder.encoder_groups:
            for encoder_block in encoder_group.encoder_blocks:

                jdx = 0

                while jdx < 4:
                    kdx = (jdx * 2) + 1
                    mdx = jdx + 1

                    transfer_item(encoder_block[kdx].weight, state_dict, f"encoder.enc_blocks.{idx}.c{mdx}.weight")
                    transfer_item(encoder_block[kdx].bias, state_dict, f"encoder.enc_blocks.{idx}.c{mdx}.bias")

                    jdx += 1

                idx += 1

        #decoder blocks
        transfer_item(model.decoder.gain, state_dict, "decoder.gain")
        transfer_item(model.decoder.bias, state_dict, "decoder.bias")
        
        idx = 0
        hdx = 0

        for decoder_group in model.decoder.decoder_groups:

            transfer_item(decoder_group.bias, state_dict, f"decoder.bias_xs.{hdx}")
            hdx += 1

            for decoder_block in decoder_group.decoder_blocks:
                
                jdx = 0

                transfer_item(decoder_block.z_projection.weight, state_dict, f"decoder.dec_blocks.{idx}.z_proj.weight")
                transfer_item(decoder_block.z_projection.bias, state_dict, f"decoder.dec_blocks.{idx}.z_proj.bias")


                while jdx < 4:
                    kdx = (jdx * 2) + 1
                    mdx = jdx + 1

                    transfer_item(decoder_block.phi[kdx].weight, state_dict, f"decoder.dec_blocks.{idx}.enc.c{mdx}.weight")
                    transfer_item(decoder_block.phi[kdx].bias, state_dict, f"decoder.dec_blocks.{idx}.enc.c{mdx}.bias")

                    transfer_item(decoder_block.theta[kdx].weight, state_dict, f"decoder.dec_blocks.{idx}.prior.c{mdx}.weight")
                    transfer_item(decoder_block.theta[kdx].bias, state_dict, f"decoder.dec_blocks.{idx}.prior.c{mdx}.bias")

                    transfer_item(decoder_block.res[kdx].weight, state_dict, f"decoder.dec_blocks.{idx}.resnet.c{mdx}.weight")
                    transfer_item(decoder_block.res[kdx].bias, state_dict, f"decoder.dec_blocks.{idx}.resnet.c{mdx}.bias")
                    
                    jdx += 1

                idx += 1

    
        #check for uninitialized parameters
        for key, val in model.state_dict().items():
            if val.sum() == 0:
                print(f"uninitialized parameter:  {key}")

        print()

        #check for untransferred parameters
        for key in state_dict.keys():
            print(f"untransferred parameter:  {key}")

        print()

        breakpoint()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.9))
        torch.save({"model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()}, 
                    save_path)
        exit()

def transfer_item(dst, state_dict, src):

    item = state_dict[src]

    if dst.shape != item.shape:
        print(f"source and destination shapes don't match")
        print(f"src:  {src}")
        breakpoint()

    else:
        dst.data = state_dict.pop(src)
