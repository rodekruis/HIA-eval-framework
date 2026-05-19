import os
import time
import json
import ast
import uuid
import requests
import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv, dotenv_values
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import AzureOpenAIModel
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, GEval

# LOADING INPUT QUESTIONS
data = pd.read_csv("data/qa_pairs_annotated.csv")

# CONNECTION TO CHATBOT ENDPOINT
endpoint = "https://hia-search-dev.azurewebsites.net/chat-dummy"
# loading variables from .env file
load_dotenv()
key = os.getenv("chatbot_key")
# # making the HTTPS request to the chatbot endpoint (as client)
# responses = []
# latencies = []
# for q in data['user_input']:
#   try:
#     start_time = time.time()
#     r = requests.post(params={"api_key": key,"threadId": str(uuid.uuid4()), "include_context": True}, url=endpoint, json={"message": q})
#     end_time = time.time()
#     latency = end_time - start_time
#     latencies.append(latency)
#   except requests.RequestException as e:
#     print(f"Error: Failed to send request for question: {q}. Error: {e}")
#     continue
#   if r.status_code != 200:
#     print(f"Error: Received status code {r.status_code} for question: {q}")
#     continue
#   response_dict = r.json()
#   responses.append({
#     'user_input' : q,
#     'bot_output': response_dict['response'],
#     'context': json.dumps(response_dict['context'])}) # Convert context to JSON string for easier handling


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
for i in range(len(data)):
    tc = LLMTestCase(
        input=data['user_input'].iloc[i],
        actual_output=data['bot_output'].iloc[i],
        retrieval_context=ast.literal_eval(data['context'].iloc[i]),
    )
    test_cases.append(tc)

# METRIC CALCULATION
ar = AnswerRelevancyMetric(
    threshold=0.75,
    include_reason=True,
    model=custom_model
)

f = FaithfulnessMetric(
    threshold=0.85,
    include_reason=True,
    model=custom_model
)

cr = ContextualRelevancyMetric(
    threshold=0.86,
    include_reason=True,
    model=custom_model
)

ta = GEval(
    name="Tone Attunement",
    criteria="Determine whether the chatbot communication style appropriately matches the emotional register of the user's query. Consider factors such as formality, empathy, and tone consistency in your evaluation.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold = 0.80,
    model=custom_model
)

cn = GEval(
    name="Cultural Neutrality",
    criteria="Determine whether responses are free from assumptions about the user's background, legal status, literacy, gender, or cultural context that could introduce bias or alienate vulnerable populations.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold = 0.90,
    model=custom_model
)

#evaluation_results = evaluate(test_cases=test_cases, metrics=[ar, f, cr, ta, cn])
# BATCH EVALUATION
batch_size = 5
all_test_results = []

for i in range(0, len(test_cases), batch_size):
    batch = test_cases[i:i + batch_size]
    print(f"Evaluating batch {i // batch_size + 1} ({len(batch)} test cases)...")
    batch_results = evaluate(test_cases=batch, metrics=[ar, f, cr, ta, cn])
    all_test_results.extend(batch_results.test_results)
    time.sleep(10)  # pause between batches to avoid rate limits


# OUTPUT AS DATAFRAME
results = []
for test_result in all_test_results:
    row = {
        "conv_id": str(uuid.uuid4()),
        "user_input": test_result.input,
        "bot_output": test_result.actual_output,
        #"latency": latencies[evaluation_results.test_results.index(test_result)],
        "retrieval_context": test_result.retrieval_context
    }
    for metric_data in test_result.metrics_data:
        row[metric_data.name] = metric_data.score
        row[metric_data.name + " Reason"] = metric_data.reason
    results.append(row)
results_df = pd.DataFrame(results)
results_df.to_csv( "data/evaluation_results.csv", index=False, encoding='utf-8-sig')
print("Evaluation complete. Results saved.")
