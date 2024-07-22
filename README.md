# safetensor2onnx2txt

Run `convert.py`, this will then ask for a `*.safetensors` file which is the SDXL Turbo model you want to load.
The program will then generate a `sdxl_unet_fp16` folder which can then be used to replace the folder in https://huggingface.co/AeroX2/stable-diffusion-xl-turbo-1.0-onnxstream

Note:
- Some models don't seem to work, no idea why, still investigating
