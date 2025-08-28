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

safetensor_path = filedialog.askopenfilename()
safetensor_path = Path(safetensor_path)

lora = messagebox.askquestion("Lora?", "Do you want to use a lora?") == 'yes'

lora_path = None
lora_weight = 1
if lora:
    lora_path = filedialog.askopenfilename()
    lora_path = Path(lora_path)
    lora_weight = float(simpledialog.askfloat("Lora weight", "Please enter the lora weight between 0 and 1:"))

root = tk.Tk()
root.title("Models to export:")
checkbox_states = {
    "text_encoder_1": tk.IntVar(value=1),
    "text_encoder_2": tk.IntVar(value=1),
    "unet_model": tk.IntVar(value=1),
    "vae_decoder": tk.IntVar(value=1)
}
checkbox_frame = tk.Frame(root, padx=100, pady=10)
checkbox_frame.pack()
tk.Checkbutton(checkbox_frame, text="Text Encoder 1", variable=checkbox_states["text_encoder_1"]).pack(anchor="w")
tk.Checkbutton(checkbox_frame, text="Text Encoder 2", variable=checkbox_states["text_encoder_2"]).pack(anchor="w")
tk.Checkbutton(checkbox_frame, text="UNET Model", variable=checkbox_states["unet_model"]).pack(anchor="w")
tk.Checkbutton(checkbox_frame, text="VAE Decoder", variable=checkbox_states["vae_decoder"]).pack(anchor="w")

onnx_path = None
vaed_path = None
text_encoder_1_path = None
text_encoder_2_path = None

def on_ok_button_click():
    global onnx_path, vaed_path, text_encoder_1_path, text_encoder_2_path
    if bool(checkbox_states['unet_model'].get()):
        onnx_path = safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'onnx') / safetensor_path.with_suffix('.onnx').name
    if bool(checkbox_states['vae_decoder'].get()):
        vaed_path = safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'vaed') / safetensor_path.with_suffix('.onnx').name
    if bool(checkbox_states['text_encoder_1'].get()):
        text_encoder_1_path = safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'text_encoder_1') / safetensor_path.with_suffix('.onnx').name
    if bool(checkbox_states['text_encoder_2'].get()):
        text_encoder_2_path = safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'text_encoder_2') / safetensor_path.with_suffix('.onnx').name
    root.destroy()

tk.Button(root, text="OK", command=on_ok_button_click).pack(pady=5)
root.update_idletasks()
window_width = root.winfo_width()
window_height = root.winfo_height()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_pos = int((screen_width - window_width) / 2)
y_pos = int((screen_height - window_height) / 2)
root.geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
root.mainloop()

simp_onnx_path = None if not onnx_path else safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'onnx') / safetensor_path.with_suffix('.sim.onnx').name
simp_vaed_path = None if not vaed_path else safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'vaed') / safetensor_path.with_suffix('.sim.onnx').name
simp_text_encoder_1_path = None if not text_encoder_1_path else safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'text_encoder_1') / safetensor_path.with_suffix('.sim.onnx').name
simp_text_encoder_2_path = None if not text_encoder_2_path else safetensor_path.parent / "{}_{}".format(safetensor_path.stem, 'text_encoder_2') / safetensor_path.with_suffix('.sim.onnx').name

unet_txt_path = None if not onnx_path else safetensor_path.parent / 'sdxl_unet_fp16'
vaed_txt_path = None if not vaed_path else safetensor_path.parent / 'sdxl_vae_decoder_fp16'
text_encoder_1_txt_path = None if not text_encoder_1_path else safetensor_path.parent / 'sdxl_text_encoder_1_fp32'
text_encoder_2_txt_path = None if not text_encoder_2_path else safetensor_path.parent / 'sdxl_text_encoder_2_fp32'

if onnx_path:
    onnx_path.parent.mkdir(exist_ok=True)
if vaed_path:
    vaed_path.parent.mkdir(exist_ok=True)
if text_encoder_1_path:
    text_encoder_1_path.parent.mkdir(exist_ok=True)
if text_encoder_2_path:
    text_encoder_2_path.parent.mkdir(exist_ok=True)

safetensor2onnx(safetensor_path, lora_path, lora_weight, onnx_path, vaed_path, text_encoder_1_path, text_encoder_2_path)

if onnx_path:
    simplify_large_onnx(AttrDict({
        "in_model_path": onnx_path,
        "out_model_path": simp_onnx_path, 
        "input_shape": "",
        "size_th_kb": 1024,
        "save_extern_data": 1,
        "skip": "",
    }))

if vaed_path:
    simplify_large_onnx(AttrDict({
        "in_model_path": vaed_path,
        "out_model_path": simp_vaed_path, 
        "input_shape": "",
        "size_th_kb": 1024,
        "save_extern_data": 1,
        "skip": "",
    }))

if text_encoder_1_path:
    simplify_large_onnx(AttrDict({
        "in_model_path": text_encoder_1_path,
        "out_model_path": simp_text_encoder_1_path, 
        "input_shape": "",
        "size_th_kb": 1024,
        "save_extern_data": 1,
        "skip": "",
    }))

if text_encoder_2_path:
    simplify_large_onnx(AttrDict({
        "in_model_path": text_encoder_2_path,
        "out_model_path": simp_text_encoder_2_path, 
        "input_shape": "",
        "size_th_kb": 1024,
        "save_extern_data": 1,
        "skip": "",
    }))

if onnx_path:
    onnx2txt(simp_onnx_path, unet_txt_path)
if vaed_path:
    onnx2txt(simp_vaed_path, vaed_txt_path)
if text_encoder_1_path:
    onnx2txt(simp_text_encoder_1_path, text_encoder_1_txt_path, fp16=False)
if text_encoder_2_path:
    onnx2txt(simp_text_encoder_2_path, text_encoder_2_txt_path, fp16=False)
