# safetensor2onnx2txt

This project allows to export a checkpoint (+LoRA optionally) of SDXL (Base or Turbo) to the [OnnxStream](https://github.com/vitoplantamura/OnnxStream) compatible text format.

Run `convert.py`, this will then ask for a `*.safetensors` file which is the SDXL model you want to load.
The program will then generate one or more folders which can then be used to replace the folders in https://huggingface.co/vitoplantamura/stable-diffusion-xl-base-1.0-onnxstream or https://huggingface.co/vitoplantamura/stable-diffusion-xl-turbo-1.0-anyshape-onnxstream.
