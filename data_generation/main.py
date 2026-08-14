import os
import json
import uuid
import requests
import pandas as pd
from dotenv import load_dotenv

print("Starting data generation...")
# LOADING INPUT QUESTIONS
data = pd.read_csv("../synthetic_data_generation/synthetic_data/final_extended_data.csv")
input_questions = data['input']

# CONNECTION TO CHATBOT ENDPOINT
endpoint = "https://hia-search-dev.azurewebsites.net/chat-dummy"
# loading variables from .env file
load_dotenv()
key = os.getenv('chatbot_key')
# making the HTTPS request to the chatbot endpoint (as client)
print("Sending requests to chatbot endpoint...")
responses = []
for i, q in enumerate(input_questions):
  try:
    r = requests.post(params={"api_key": key, "threadId": str(uuid.uuid4()), "include_context": True}, url=endpoint, json={"message": q})
  except requests.RequestException as e:
    print(f"Error: Failed to send request for question: {q}. Error: {e}")
    continue
  if r.status_code != 200:
    print(f"Error: Received status code {r.status_code} for question: {q}")
    continue
  response_dict = r.json()
  responses.append({
    'user_input' : q,
    'bot_output': response_dict['response'],
    'context': json.dumps(response_dict['context']),
    'orig_idx' : i})
print("Data generation completed.")

# OUTPUT AS DATAFRAME
results = []
for i in range(len(responses)):
    idx = responses[i]['orig_idx']
    results.append({
        "user_input": responses[i]['user_input'],
        # "expected_output": data['expected_output'][idx],
        "bot_output": responses[i]['bot_output'],
        "context": responses[i]['context']

    })

results_df = pd.DataFrame(results)
results_df.to_csv( "../data/extended_data.csv", index=False, encoding='utf-8-sig')
print("Questions, responses, and contexts saved.")
