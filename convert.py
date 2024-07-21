from safetensor2onnx import safetensor2onnx
from simplify_large_onnx import simplify_large_onnx
from onnx2txt import onnx2txt

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

import tkinter as tk
from tkinter import simpledialog, filedialog, messagebox
from pathlib import Path

root = tk.Tk()
root.withdraw()

safetensor_path = filedialog.askopenfilename()
safetensor_path = Path(safetensor_path)

lora = messagebox.askquestion("Lora?", "Do you want to use a lora?") == 'yes'

lora_path = None
lora_weight = 1
if lora:
    lora_path = filedialog.askopenfilename()
    lora_path = Path(lora_path)
    lora_weight = float(simpledialog.askfloat("Lora weight", "Please enter the lora weight between 0 and 1:"))

onnx_path = safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'onnx') / safetensor_path.with_suffix('.onnx').name
#text_encoder_2_path = safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'text_encoder_2') / safetensor_path.with_suffix('.onnx').name
simp_onnx_path = safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'onnx') / safetensor_path.with_suffix('.sim.onnx').name
#simp_text_encoder_2_path = safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'text_encoder_2') / safetensor_path.with_suffix('.sim.onnx').name
unet_txt_path = safetensor_path.parent / 'sdxl_unet_fp16'
#text_encoder_2_txt_path = safetensor_path.parent / 'sdxl_text_encoder_2_fp32'

onnx_path.parent.mkdir(exist_ok=True)
#text_encoder_2_path.parent.mkdir(exist_ok=True)

safetensor2onnx(safetensor_path, lora_path, lora_weight, onnx_path) #, text_encoder_2_path)

input_shape = '{"sample":[1,4,64,64],"timestep":[1],"encoder_hidden_state":[1,77,2048],"text_embeds":[1,1280],"time_ids":[1,6]}'
simplify_large_onnx(AttrDict({
    "in_model_path": onnx_path,
    "out_model_path": simp_onnx_path, 
    "input_shape": input_shape,
    "size_th_kb": 1024,
    "save_extern_data": 1,
    "skip": "",
}))

#input_shape = '{"input_ids": [1,77]}'
#simplify_large_onnx(AttrDict({
#    "in_model_path": text_encoder_2_path,
#    "out_model_path": simp_text_encoder_2_path, 
#    "input_shape": input_shape,
#    "size_th_kb": 1024,
#    "save_extern_data": 1,
#    "skip": "",
#}))

onnx2txt(simp_onnx_path, unet_txt_path)
#onnx2txt(simp_text_encoder_2_path, text_encoder_2_txt_path, fp16=False)
