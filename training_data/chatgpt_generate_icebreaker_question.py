# This script is to use chatgpt 3.5 API to generate icebreaker questions from given conversation topics
from openai import OpenAI
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "your-openai-api-key"

# Set your OpenAI API key here
client = OpenAI(api_key=API_KEY)

OUTPUT_PATH = "icebreaker_questions.jsonl"
TOPIC_PATH = "topics.txt"
PROMPT_TEMP = "Can you generate 100 icebreaker questions related to {topic}?"

EXAMPLE_QUESTION = 'Can you generate 10 icebreaker questions related to Museum?'
EXAMPLE_ANSWER = '["1. Do you enjoy visiting museums, and if so, what type of museum is your favorite?","2. What\'s the most memorable exhibit you\'ve ever seen in a museum?","3. Have you ever been surprised or moved by an unexpected discovery in a museum?","4. Do you prefer exploring museums alone or with friends or family?","5. What\'s the most unusual or niche museum you\'ve ever visited?","6. Are there any museums on your bucket list that you\'d like to visit in the future?","7. Have you ever attended a special event or exhibition opening at a museum?","8. What\'s your favorite museum memory from a school field trip or childhood visit?","9. Do you collect souvenirs from the museums you\'ve visited?","10. If you could design your own museum exhibit, what topic or theme would it focus on?"]'

def load_topics(path=TOPIC_PATH):
    data = []
    with open(path, mode="r", encoding="utf-8") as f:
        for line in f:
            data.append(line.strip())
    return data

def gen_all_prompts():
    topics = load_topics()
    sources = [
        PROMPT_TEMP.format(topic=topic.strip()) 
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
    return response.choices[0]


def main(max_workers=3, qps_limit=50):
    start_time = time.time()    
    non_finish_counter = 0
    finish_counter = 0
    dialogs = gen_all_prompts()
    
    # Calculate the time interval between API calls based on the QPS limit
    time_interval = 1 / qps_limit + 3
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