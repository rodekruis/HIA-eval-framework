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
goldens = pd.read_csv("../synthetic_data/goldens.csv")

# CONNECTION TO CHATBOT ENDPOINT
endpoint = "https://hia-search-dev.azurewebsites.net/chat-dummy"
# loading variables from .env file
load_dotenv()
key = os.getenv("chatbot_key")
# making the HTTPS request to the chatbot endpoint (as client)
#r = requests.post(params={"api_key": key}, url=endpoint, json={"question": goldens['question'][0]})

# EXTRACTING NECESSARY DATA FOR METRIC CALCULATION
# actual_output = r.json()['answer']
# expected_output = goldens['answer'][0]

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
for i in range(len(goldens)):
    tc = LLMTestCase(
        input=goldens['input'][i],
        actual_output=goldens['expected_output'][i],
        #expected_output=goldens['expected_output'][i],
        retrieval_context=ast.literal_eval(goldens['context'][i]),
        context=ast.literal_eval(goldens['context'][i])
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
  results.append({
      "input": test_result.input,
      #"expected_output": test_result.expected_output,
      "actual_output": test_result.actual_output
  })
  for metric_data in test_result.metrics_data:
      results.append({
          "score": metric_data.score,
          "reason": metric_data.reason
      })

results_df = pd.DataFrame(results)
results_df.to_csv( "./faithfulness_results.csv", index=False)
