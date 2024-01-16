# Extract icebreaker questions
import json

INPUT = 'icebreaker_questions.jsonl'
OUTPUT = 'icebreaker_questions.txt'

data = []
with open(INPUT, 'r', encoding="utf-8") as input_fp, open(OUTPUT, 'a', encoding="utf-8") as output_fp:
    for line in input_fp:
        raw_data = json.loads(line)
        response = json.loads(raw_data['response'])
        questions = response if isinstance(response, list) else response['questions']
        questions = [s.split('. ', 1)[1] if s.startswith(tuple(str(i) + '.' for i in range(1, 101))) else s for s in questions]
        for question in questions:
            output_fp.write(question.strip() + "\n")
        
        
  