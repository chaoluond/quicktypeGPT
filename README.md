# Introduction
Popular large language models (LLMs) like ChatGPT 3.5/4, Claude 2, Google Gemini, and Meta Llama 70 are gigantic models which have tens or hunderends billions of parameters and require hundreds/thousands Gigbytes of memory and server-grade GPUs to train, finetune and inference. This sets a high bar for individual developers to quickly explore possibilities of LLMs and develop applications on top of it. Inspired by @karpathy's [llama2.c](https://github.com/karpathy/llama2.c) project, I trained a simple GPT model called **QuicktypeGPT** to **assist typing and completing daily conversations**. 

**QuicktypeGPT** only has 15M parameters (dim = 288, 6 layers, 6 heads and 6 kv heads) and 27MB. The model is pre-trained on a single A40 GPU and can be inferenced through a pure C program on a laptop CPU (e.g. AMD, Intel) with decent quality and speed. This project is to demonstrate that (1) we do not need to train a very sophisticated LLM but can still achieve santisfactory performance if the LLM is only focused on a small and dedicated domain or task, (2) we can deploy small LLMs on edge devices (e.g. desktop, laptop, tablet or phone) to perform inference tasks without relying on the servers in the cloud. 

In the following sections, training data collection, pretraining and inference will be discussed. 

# Training Data
As I mentioned in the introduction, to train a small LLM we need to ensure it only focuses on a small and specific task. For **QuicktypeGPT**, it is intended to help you to auto complete a reply based on a previous multi-turn conversation. Ideally, it will be integrated with keyword input method to help you type and reply quicker. 

Following the philosophy of **knowledge distillation**, I used ChatGPT 3.5 API to generate 30k two-person multi-turn conversations across 40+ common topics (e.g. travel, food, music, movie/TV, education, hobbies, family, sports, technology, books, etc.) Here we can also use other mature LLMs, for instance Llama 2 70B-chat or Google Gemini Ultra, to generate the conversations. Here is how I generated the train data:

- Step 1. Use ChatGPT to "spawn" icebreaker questions based on conversation [topics](https://github.com/chaoluond/quicktypeGPT/blob/main/training_data/topics.txt). (generation [script](https://github.com/chaoluond/quicktypeGPT/blob/main/training_data/chatgpt_generate_icebreaker_question.py) and data process [script](https://github.com/chaoluond/quicktypeGPT/blob/main/training_data/extract_icebreaker_question.py)) In total, I used ChatGPT to generate 7k [icebreaker questions](https://github.com/chaoluond/quicktypeGPT/blob/main/training_data/icebreaker_questions.txt).  
- Step 2. Use ChatGPT to generate two-person multi-turn conversations based on icebreaker questions. (generation [script](https://github.com/chaoluond/quicktypeGPT/blob/main/training_data/chatgpt_generate_conversation.py))

For instance, the following conversation is generated using the icebreaker question of **What is your opinion on artificial intelligence?**:

```
Tom: what's your take on artificial intelligence?    
Sarah: Hi I think artificial intelligence is a game-changer. It has the potential to revolutionize industries and solve complex problems.    
Tom: That's true. But do you have any concerns about AI's impact on privacy?    
Sarah: Privacy is definitely a concern. As AI becomes more advanced, we need robust regulations to protect personal data and ensure transparency.    
Tom: I agree. Striking a balance between innovation and privacy protection is crucial for the widespread acceptance of AI.    
Sarah: Absolutely, Privacy should never be compromised in the pursuit of technological advancements.
```
The train/val datasets can be downloaded from huggingface. 



