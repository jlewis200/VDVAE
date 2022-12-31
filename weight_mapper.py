#!/usr/bin/python3
import torch

from argparse import ArgumentParser
from configs import get_model

def main():
    """
    """

    # get command line args
    parser = ArgumentParser(description="Transfer weights from original VDVAE to refactored version.")

    #pre-trained options
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--donor-path", type=str, required=True)

    args = parser.parse_args()

    model = get_model(args.config)

    if model is None:
        return

    transfer_weights(model, args.save_path, args.donor_path) 

def transfer_weights(model, save_path, donor_weights):

    state_dict = torch.load(donor_weights)

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

        #ensure all weights have been cleared
        for parameter in model.parameters():
            assert(parameter.sum() == 0)

 
        #disect dmolnet weights
        #this uses the same confusing steps in the original but the reshapes/slicing is at the first dims instead of the last
        dmll_net = model.decoder.dmll_net 
        
        conv_weight = state_dict.pop("decoder.out_net.out_conv.weight")
        #shape 100 x 512 x 1 x 1

        conv_bias = state_dict.pop("decoder.out_net.out_conv.bias")
        #shape 100
       
        dmll_net.logits.weight.data = conv_weight[0 : 10]
        dmll_net.logits.bias.data = conv_bias[0 : 10]

        conv_weight = torch.reshape(conv_weight[10:], (3, 30) + conv_weight.shape[1:])
        conv_bias   = torch.reshape(conv_bias[10:], (3, 30))

        means_weight = conv_weight[:, 0:10]
        means_bias = conv_bias[:, 0:10]
 
        log_scales_weight = conv_weight[:, 10:20]
        log_scales_bias = conv_bias[:, 10:20]
 
        coeffs_weight = conv_weight[:, 20:30]
        coeffs_bias = conv_bias[:, 20:30]

        #dmll_net.r_mean.weight.data = means_weight[0]
        dmll_net.r_mean.weight.data = means_weight[0]
        dmll_net.r_logscale.weight.data = log_scales_weight[0]
        dmll_net.gr_coeff.weight.data = coeffs_weight[0]
        
        dmll_net.r_mean.bias.data = means_bias[0]
        dmll_net.r_logscale.bias.data = log_scales_bias[0]
        dmll_net.gr_coeff.bias.data = coeffs_bias[0]

        dmll_net.g_mean.weight.data = means_weight[1]
        dmll_net.g_logscale.weight.data = log_scales_weight[1]
        dmll_net.br_coeff.weight.data = coeffs_weight[1]
        
        dmll_net.g_mean.bias.data = means_bias[1]
        dmll_net.g_logscale.bias.data = log_scales_bias[1]
        dmll_net.br_coeff.bias.data = coeffs_bias[1]

        dmll_net.b_mean.weight.data = means_weight[2]
        dmll_net.b_logscale.weight.data = log_scales_weight[2]
        dmll_net.bg_coeff.weight.data = coeffs_weight[2]
        
        dmll_net.b_mean.bias.data = means_bias[2]
        dmll_net.b_logscale.bias.data = log_scales_bias[2]
        dmll_net.bg_coeff.bias.data = coeffs_bias[2]
    
        #reinitialize the mixture_net weights
        #model.decoder.mixture_net.dist_r_0.reset_parameters()
        #model.decoder.mixture_net.dist_r_1.reset_parameters()
        #model.decoder.mixture_net.dist_g_0.reset_parameters()
        #model.decoder.mixture_net.dist_g_1.reset_parameters()
        #model.decoder.mixture_net.dist_b_0.reset_parameters()
        #model.decoder.mixture_net.dist_b_1.reset_parameters()
        #model.decoder.mixture_net.mix_score.reset_parameters()

        #reinitialize the out_conv weights
        #model.decoder.out_conv.reset_parameters()

        #misc
        transfer_item(model.encoder.in_conv.weight, state_dict, "encoder.in_conv.weight")
        transfer_item(model.encoder.in_conv.bias, state_dict, "encoder.in_conv.bias")
        #transfer_item(model.decoder.out_net.out_conv.weight, state_dict, "decoder.out_net.out_conv.weight")
        #transfer_item(model.decoder.out_net.out_conv.bias, state_dict, "decoder.out_net.out_conv.bias")


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
            if decoder_group.bias is not None:
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
        print(f"trying to permute.")
        breakpoint()

    else:
        dst.data = state_dict.pop(src)

if __name__ == "__main__":
    main()
