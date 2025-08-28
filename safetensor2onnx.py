import torch
import torch.nn as nn
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler

class UNetModel(nn.Module):
	def __init__(self, unet):
		super(UNetModel, self).__init__()
		self.unet = unet
	def forward(self, sample, timestep, encoder_hidden_states, text_embeds, time_ids):
		out_sample = self.unet(return_dict=False,
			sample=sample, timestep=timestep, encoder_hidden_states=encoder_hidden_states,
			added_cond_kwargs={ "text_embeds":text_embeds, "time_ids":time_ids })
		return out_sample

class VAED(nn.Module):
    def __init__(self, vae):
        super(VAED, self).__init__()
        self.vae = vae
    def forward(self, latents):
        self.vae.enable_slicing()
        self.vae.enable_tiling()
        image = self.vae.decode(latents, return_dict=False)[0] # / self.vae.config.scaling_factor
        return image

class TE1(nn.Module):
    def __init__(self, te1):
        super(TE1, self).__init__()
        self.te1 = te1
    def forward(self, input_ids):
        output=self.te1(input_ids, output_hidden_states=True,return_dict=False)
        return [output[0], output[1], *output[2]] # 15

class TE2(nn.Module):
    def __init__(self, te2):
        super(TE2, self).__init__()
        self.te2 = te2
    def forward(self, input_ids):
        output=self.te2(input_ids, output_hidden_states=True,return_dict=False)
        return [output[0], output[1], *output[2]] # 35

def safetensor2onnx(input_file, lora_file, lora_weight, output_unet_path, output_vaed_path, output_te1_path, output_te2_path):

    with torch.no_grad():

        pipe = StableDiffusionXLPipeline.from_single_file(input_file)
        if lora_file is not None:
            pipe.load_lora_weights(str(lora_file.parent), weight_name=str(lora_file.name), adapter_name="lora")
            pipe.set_adapters(["lora"], adapter_weights=[lora_weight])
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

        if output_unet_path:

            dummy_input = ( torch.randn(1, 4, 64, 64), torch.randn(1),
                torch.randn(1, 77, 2048),
                torch.randn(1, 1280), torch.randn(1, 6))
            input_names = [ "sample", "timestep",
                "encoder_hidden_states",
                "text_embeds", "time_ids" ]
            output_names = [ "out_sample" ]
            
            torch.onnx.export(UNetModel(pipe.unet), dummy_input, str(output_unet_path), verbose=False,
                input_names=input_names, output_names=output_names,
                opset_version=14, do_constant_folding=True,
                dynamic_axes={"sample": {2: "dim0", 3: "dim1"}})

        if output_vaed_path:

            dummy_input = ( torch.randn(1, 4, 128, 128), )
            input_names = [ "latent_sample" ]
            output_names = [ "sample" ]
            
            torch.onnx.export(VAED(pipe.vae), dummy_input, str(output_vaed_path), verbose=False,
                input_names=input_names, output_names=output_names,
                opset_version=14, do_constant_folding=True,
                dynamic_axes={"latent_sample": {2: "dim0", 3: "dim1"}})

        if output_te1_path:

            dummy_input = ( torch.randint(0, 100, (1, 77)), )
            input_names = [ "input_ids" ]
            output_names = [ "out_" + str(i) for i in range(15) ]
            
            torch.onnx.export(TE1(pipe.text_encoder.float()), dummy_input, str(output_te1_path), verbose=False,
                input_names=input_names, output_names=output_names,
                opset_version=14, do_constant_folding=True)

        if output_te2_path:

            dummy_input = ( torch.randint(0, 100, (1, 77)), )
            input_names = [ "input_ids" ]
            output_names = [ "out_" + str(i) for i in range(35) ]
            
            torch.onnx.export(TE2(pipe.text_encoder_2.float()), dummy_input, str(output_te2_path), verbose=False,
                input_names=input_names, output_names=output_names,
                opset_version=14, do_constant_folding=True)
