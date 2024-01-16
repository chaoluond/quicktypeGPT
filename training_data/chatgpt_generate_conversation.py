from openai import OpenAI
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

SLICE = 10
API_KEY = "your-openai-api-key"

# Set your OpenAI API key here
client = OpenAI(api_key=API_KEY)

OUTPUT_PATH = "dialogues.jsonl"
TOPIC_PATH = "icebreaker_questions.txt"
PROMPT_TEMP_PATH = "prompt_template.txt"

EXAMPLE_QUESTION = 'Tom is learning English. Can you generate some written dialogues for him to practice English? \nThe dialogue is about "Introductions: Name, age, and where you\'re from." Please use "Tom" and "Sarah" as two persons in the dialogue. \nCan you generate 4 totally different dialogues?'
EXAMPLE_ANSWER = '[\n  {\n    "Tom": "Hi, I\'m Tom. Nice to meet you.",\n    "Sarah": "Hi Tom, I\'m Sarah. It\'s a pleasure to meet you too.",\n    "Tom": "How old are you, Sarah?",\n    "Sarah": "I\'m 25 years old. And you, Tom?",\n    "Tom": "I\'m 28. Where are you from, Sarah?",\n    "Sarah": "I\'m from New York. How about you, Tom?",\n    "Tom": "I\'m from London."\n  },\n  {\n    "Tom": "Hello, I\'m Tom. What\'s your name?",\n    "Sarah": "Hi Tom, I\'m Sarah. Nice to meet you.",\n    "Tom": "Nice to meet you too, Sarah. How old are you?",\n    "Sarah": "I\'m 30 years old. And you, Tom?",\n    "Tom": "I\'m 32. Where are you from, Sarah?",\n    "Sarah": "I\'m originally from Paris. How about you, Tom?",\n    "Tom": "I\'m from Sydney."\n  },\n  {\n    "Tom": "Hey, I\'m Tom. What\'s your name?",\n    "Sarah": "Hi Tom, I\'m Sarah. It\'s great to meet you.",\n    "Tom": "Great to meet you too, Sarah. How old are you?",\n    "Sarah": "I\'m 22 years old. And you, Tom?",\n    "Tom": "I\'m 24. Where are you from, Sarah?",\n    "Sarah": "I\'m from Los Angeles. How about you, Tom?",\n    "Tom": "I\'m from Toronto."\n  },\n  {\n    "Tom": "Hi, I\'m Tom. What\'s your name?",\n    "Sarah": "Hi Tom, I\'m Sarah. Nice to meet you.",\n    "Tom": "Nice to meet you too, Sarah. How old are you?",\n    "Sarah": "I\'m 35 years old. And you, Tom?",\n    "Tom": "I\'m 38. Where are you from, Sarah?",\n    "Sarah": "I\'m originally from Berlin. How about you, Tom?",\n    "Tom": "I\'m from Madrid."\n  }\n]'

def load_prompt_template(path=PROMPT_TEMP_PATH):
    with open(path, "r", encoding="utf-8") as f:
        prompt = f.read()
    return prompt

def load_topics(path=TOPIC_PATH):
    data = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def gen_all_prompts():
    prompt_template = load_prompt_template()
    topics = load_topics()
    sources = [
        prompt_template.format(topic=topic.strip()) 
        for topic in topics
    ]

    prompts = []
    for requirement in sources:
        requirement = requirement.strip()
        prompt = [
            {
              "role": "system",
              "content": "You are a helpful and respectful assistant. Please always generate JSON."
            },
            {
              "role": "user",
              "content": EXAMPLE_QUESTION,
            }, 
            {
              "role": "assistant",
              "content": EXAMPLE_ANSWER,
            },             
            {
              "role": "user",
              "content": requirement,
            },
        ]
        prompts.append(prompt)
    return prompts
              

def gen_answer(prompt):
    # Try 5 times to get a well-formed response
    try:
        for i in range(5):
            print(prompt)
            response = client.chat.completions.create(
                messages=prompt,
                model="gpt-3.5-turbo",
                temperature=0.5,
                max_tokens=3300,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
            )
            print(response)
            if response.choices[0].finish_reason == 'stop':
                break
    except:
        return None
    print(response.choices[0].message.content)
    return response.choices[0]


def main(max_workers=3, qps_limit=50):
    start_time = time.time()    
    non_finish_counter = 0
    finish_counter = 0
    dialogs = gen_all_prompts()
    
    # Calculate the time interval between API calls based on the QPS limit
    time_interval = 1 / qps_limit + 20
    batch_num = len(dialogs) // SLICE
    for i in range(0, batch_num):
        batch_start_time = time.time()
        # Create a ThreadPoolExecutor with the specified number of workers (threads)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Keep track of the futures and questions separately
            futures_and_questions = {}
            for question in dialogs[i * SLICE : (i + 1) * SLICE]:
                # Submit the API call to the ThreadPoolExecutor and record the future
                future = executor.submit(gen_answer, question)
                futures_and_questions[future] = question
            
                # Sleep to ensure QPS limit compliance
                time.sleep(time_interval)
        
            # Process the completed futures and write the answers to the output file
            for future in as_completed(futures_and_questions):
                question = futures_and_questions[future]
                response = future.result()
                if (response is None or response.finish_reason != 'stop'):
                    non_finish_counter = non_finish_counter + 1
                    print("Not finished!")
                    print("Non finish count:")
                    print(non_finish_counter)
                else:
                    with open(OUTPUT_PATH, "a", encoding="utf-8") as output_fp:
                        conversation = {
                            "prompt": question,
                            "response": response.message.content,
                        }
                        output_fp.write(json.dumps(conversation) + "\n")
                
                    finish_counter = finish_counter + 1
                    if (finish_counter % 50 == 0):
                        print("Finished:")
                        print(finish_counter)
            print("batch: " + str(i))
            print("Batch process time: " + str(time.time() - batch_start_time))
                    
    print("Total time elapsed:")
    print(time.time() - start_time)
    
if __name__ == "__main__":
    main(max_workers=3, qps_limit=50)