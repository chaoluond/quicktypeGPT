# Introduction
Popular large language models (LLMs) like ChatGPT 3.5/4, Claude 2, Google Gemini, and Meta Llama 70 are gigantic models which have tens or hunderends billions of parameters and require hundreds/thousands Gigbytes of memory and server-grade GPUs to train, finetune and inference. This sets a high bar for individual developers to quickly explore possibilities of LLMs and develop applications on top of it. Inspired by @karpathy's [llama2.c](https://github.com/karpathy/llama2.c) project, I trained a simple GPT model called **QuicktypeGPT** to **assist typing and completing daily conversations**. This repository is forked from the llama2.c project.  

**QuicktypeGPT** only has 15M parameters (dim = 288, 6 layers, 6 heads and 6 kv heads) and 27MB. The model is pre-trained on a single A40 GPU and can be inferenced through a pure C program on a laptop CPU (e.g. AMD, Intel) with decent quality and speed. This project is to demonstrate that (1) we do not need to train a very sophisticated LLM but can still achieve santisfactory performance if the LLM is only focused on a small and dedicated domain or task, (2) we can deploy small LLMs on edge devices (e.g. desktop, laptop, tablet or phone) to perform inference tasks without relying on the servers in the cloud. 

In the following sections, training data collection, pretraining and inference will be discussed. 

# Training
## Training Data Collection
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
The train/val datasets can be downloaded from huggingface. ([link](https://huggingface.co/datasets/safetyllm/daily_conversations)) 

## Custom Tokenizer
First of all, we need to train a new tokenizer based on all texts (training and validation datasets). We apply [Byte-Pair Encoding (BPE)](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt) to train our tokenizer. Since the train/val datasets are all English text and no special characters are used, I set the vocabulary size to be 4096. The vocabulary size for Llama2 is 32000. The use of a small vocabulary size has several advantages: (1) reduce model parameters and size; (2) increase train and inference speed. If your text is simple and easy words, a small vocabulary size should suffice to tokenize the text efficiently. Here is how I train the custom tokenizer:
```
python dataprocess.py train_vocab --vocab_size=4096
python dataprocess.py pretokenize --vocab_size=4096
```
`train_vocab` is to train our tokenizer and `pretokenize` is to tokenize train/val datasets for model training. The tokenized dataset will be saved as `.bin` files in the corresponding folder.  

## Model Training
After we have our own tokenizer, we can now go ahead to do the real training work. 
```
python train.py --vocab_source=custom --vocab_size=4096
```



