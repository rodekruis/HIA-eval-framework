import os
import ast
import pandas as pd
import numpy as np
import requests
from deepeval import evaluate
from deepeval.metrics import FaithfulnessMetric
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv, dotenv_values
from deepeval.models import AzureOpenAIModel

# LOADING INPUT QUESTIONS
data = pd.read_csv("../data_generation/questions.csv")

# CONNECTION TO CHATBOT ENDPOINT
endpoint = "https://hia-search-dev.azurewebsites.net/chat-dummy"
# loading variables from .env file
load_dotenv()
key = os.getenv("chatbot_key")
# making the HTTPS request to the chatbot endpoint (as client)
responses = []
for q in data['query']:
  try:
    r = requests.post(params={"api_key": key, "include_context": True}, url=endpoint, json={"message": q})
  except requests.RequestException as e:
    print(f"Error: Failed to send request for question: {q}. Error: {e}")
    continue
  if r.status_code != 200:
    print(f"Error: Received status code {r.status_code} for question: {q}")
    continue
  response_dict = r.json()
  responses.append({'user_input' : q,
    'bot_output': response_dict['response'],
    'context': response_dict['context']})

# CUSTOM MODEL
endpoint = "https://510-ai-research.openai.azure.com/"
model = "gpt-4.1"
deployment = "gpt-4.1-students"
# loading variables from .env file
subscription_key = os.getenv("azure_subscription_key")
api_version = "2024-12-01-preview"
# model
custom_model = AzureOpenAIModel(
  model=deployment,
  api_key=subscription_key,
  azure_endpoint=endpoint,
  api_version=api_version,
  deployment_name=deployment
)

# TEST CASE CREATION
test_cases = []
for i in range(len(responses)):
    tc = LLMTestCase(
        input=responses[i]['user_input'],
        actual_output=responses[i]['bot_output'],
        retrieval_context=responses[i]['context']
    )
    test_cases.append(tc)

# METRIC CALCULATION
metric = FaithfulnessMetric(
    threshold=0.5,
    include_reason=True,
    model=custom_model
)

evaluation_results = evaluate(test_cases=test_cases, metrics=[metric])

# OUTPUT AS DATAFRAME
results = []
for test_result in evaluation_results.test_results:
    row = {
        "input": test_result.input,
        "bot_output": test_result.actual_output,
        "retrieval_context": test_result.retrieval_context
    }
    for metric_data in test_result.metrics_data:
        row[metric_data.name] = metric_data.score
        row[metric_data.name + "_reason"] = metric_data.reason
    results.append(row)
results_df = pd.DataFrame(results)
results_df.to_csv( "./faithfulness_results.csv", index=False)
