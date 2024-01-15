# Introduction
Popular large language models (LLMs) like ChatGPT 3.5/4, Claude 2, Google Gemini, and Meta Llama 70 are gigantic models which have tens or hunderends billions of parameters and require hundreds/thousands Gigbytes of memory and server-grade GPUs to train, finetune and inference. This sets a high bar for individual developers to quickly explore possibilities of LLMs and make applications on top of it. Inspired by @karpathy's [llama2.c](https://github.com/karpathy/llama2.c) project, I trained a simple GPT model called **quicktypeGPT** to assist typing and completing daily dialogues. 

QuicktypeGPT only has 15M parameters (dim = 288, 6 layers, 6 heads and 6 kv heads) and 27MB. The model was trained using Pytorch on single A40 GPU for a 2-3 hours and can be inferenced through a pure C program on a laptop CPU (e.g. AMD, Intel) with decent quality and speed. The purpose of this project is to demonstrate that   

