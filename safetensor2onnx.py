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
        
class TextEncoder(nn.Module):
	def __init__(self, text_encoder):
		super(TextEncoder, self).__init__()
		self.text_encoder = text_encoder
	def forward(self, input_ids):
		out = self.text_encoder(input_ids, output_hidden_states=True)
		return out
        
class VaeDecoder(nn.Module):
	def __init__(self, vae):
		super(VaeDecoder, self).__init__()
		self.vae = vae
	def forward(self, latents):
		out = self.vae.decode(latents, return_dict=False)[0]
		return out

def safetensor2onnx(input_file, lora_file, lora_weight, output_unet_path): #, output_text_encoder_2_path):
    with torch.no_grad():
        pipe = StableDiffusionXLPipeline.from_single_file(input_file)
        if lora_file is not None:
            pipe.load_lora_weights(str(lora_file.parent), weight_name=str(lora_file.name), adapter_name="lora")
            pipe.set_adapters(["lora"], adapter_weights=[lora_weight])
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
        
        # import struct;
        # import numpy as np;
        # f = open("m_prompt_embeds_1", "rb")
        # data = f.read()
        # m_prompt_embeds_1 = []
        # for i in range(0,len(data),4):
           # m_prompt_embeds_1.append(struct.unpack('f', data[i:i+4]))
        # m_prompt_embeds_1 = np.array(m_prompt_embeds_1, dtype=np.float32)
        
        # f = open("m_prompt_embeds_2", "rb")
        # data = f.read()
        # m_prompt_embeds_2 = []
        # for i in range(0,len(data),4):
           # m_prompt_embeds_2.append(struct.unpack('f', data[i:i+4]))
        # m_prompt_embeds_2 = np.array(m_prompt_embeds_2, dtype=np.float32)
        
        # f = open("m_pooled_prompt_embeds", "rb")
        # data = f.read()
        # m_pooled_prompt_embeds = []
        # for i in range(0,len(data),4):
           # m_pooled_prompt_embeds.append(struct.unpack('f', data[i:i+4]))
        # m_pooled_prompt_embeds = np.array(m_pooled_prompt_embeds, dtype=np.float32)

        
        # dummy_input = ( pipe.tokenizer(
            # "a photo of an astronaut riding a horse on mars",
            # padding="max_length",
            # max_length=pipe.tokenizer.model_max_length,
            # truncation=True,
            # return_tensors="pt",
        # ).input_ids.type(torch.int32), )
        
        # abc = pipe.text_encoder_2(dummy_input[0], output_hidden_states=True)
        # ghi = abc[0]
        # jkl = abc.hidden_states[-2]
        
        # import pdb; pdb.set_trace();
        # pipe("snow, hdautomaton, robot, non-humanoid robot, soldier, side view, glowing eyes, machine", num_inference_steps=8, guidance_scale=0.0).images[0].show();
        
        #latents = latents.reshape((1,4,128,128))
        #latents = torch.from_numpy(latents)
        #image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        #pipe.image_processor.postprocess(image, output_type="pil").show()
        
        with torch.no_grad():
            dummy_input = ( torch.randn(1, 4, 64, 64), torch.randn(1),
                torch.randn(1, 77, 2048),
                torch.randn(1, 1280), torch.randn(1, 6))
            input_names = [ "sample", "timestep",
                "encoder_hidden_states",
                "text_embeds", "time_ids" ]
            output_names = [ "out_sample" ]
            
            torch.onnx.export(UNetModel(pipe.unet), dummy_input, str(output_unet_path), verbose=False,
                input_names=input_names, output_names=output_names,
                opset_version=14, do_constant_folding=True)
           
            # dummy_input = ( pipe.tokenizer(
                # "example prompt",
                # padding="max_length",
                # max_length=pipe.tokenizer.model_max_length,
                # truncation=True,
                # return_tensors="pt",
            # ).input_ids.type(torch.int32), )
            # input_names = [ "input_ids" ]
            # output_names = [ "out_"+str(i) for i in range(34) ]
            
            # torch.onnx.export(TextEncoder(pipe.text_encoder_2), dummy_input, str(output_text_encoder_2_path), verbose=False,
                 # input_names=input_names, output_names=output_names,
                 # opset_version=14, do_constant_folding=True)