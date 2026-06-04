import os
import time
import json
import uuid
import requests
import pandas as pd
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv
from deepeval.test_case import LLMTestCaseParams
from deepeval.models import AzureOpenAIModel
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, GEval

# LOADING INPUT QUESTIONS
data = pd.read_csv("data/goldens_final.csv")

# loading variables from .env file
load_dotenv()

# CONNECTION TO CHATBOT ENDPOINT
endpoint = "https://hia-search-dev.azurewebsites.net/chat-dummy"
key = os.getenv("chatbot_key")

responses = []
latencies = []
for q in data['query']:
    try:
        start_time = time.time()
        r = requests.post(
            params={"api_key": key, "threadId": str(uuid.uuid4()), "include_context": True},
            url=endpoint,
            json={"message": q}
        )
        latency = time.time() - start_time
    except requests.RequestException as e:
        print(f"Error: Failed to send request for question: {q}. Error: {e}")
        continue
    if r.status_code != 200:
        print(f"Error: Received status code {r.status_code} for question: {q}")
        continue
    response_dict = r.json()
    responses.append({
        'user_input': q,
        'bot_output': response_dict['response'],
        'context': response_dict['context']
    })
    latencies.append(latency)

# CUSTOM MODEL
subscription_key = os.getenv("azure_subscription_key")
custom_model = AzureOpenAIModel(
    model="gpt-4.1-students",
    api_key=subscription_key,
    azure_endpoint="https://510-ai-research.openai.azure.com/",
    api_version="2024-12-01-preview",
    deployment_name="gpt-4.1-students"
)

# TEST CASE CREATION (from responses, not CSV)
test_cases = []
for resp in responses:
    tc = LLMTestCase(
        input=resp['user_input'],
        actual_output=resp['bot_output'],
        retrieval_context=resp['context'],
    )
    test_cases.append(tc)

# METRIC CALCULATION
ar = AnswerRelevancyMetric(threshold=0.75, include_reason=True, model=custom_model)
f = FaithfulnessMetric(threshold=0.85, include_reason=True, model=custom_model)
cr = ContextualRelevancyMetric(threshold=0.86, include_reason=True, model=custom_model)
ta = GEval(
    name="Tone Attunement",
    criteria="Determine whether the chatbot communication style appropriately matches the emotional register of the user's query.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.80, model=custom_model
)
cn = GEval(
    name="Cultural Neutrality",
    criteria="Determine whether responses are free from assumptions about the user's background, legal status, literacy, gender, or cultural context.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.90, model=custom_model
)

# BATCH EVALUATION
batch_size = 5
all_test_results = []
for i in range(0, len(test_cases), batch_size):
    batch = test_cases[i:i + batch_size]
    print(f"Evaluating batch {i // batch_size + 1} ({len(batch)} test cases)...")
    batch_results = evaluate(test_cases=batch, metrics=[ar, f, cr, ta, cn])
    all_test_results.extend(batch_results.test_results)
    time.sleep(10)

# OUTPUT AS DATAFRAME
results = []
for idx, test_result in enumerate(all_test_results):
    row = {
        "conv_id": str(uuid.uuid4()),
        "user_input": test_result.input,
        "bot_output": test_result.actual_output,
        "latency": latencies[idx],
        "latency_pass": 1 if latencies[idx] < 7.45 else 0,
        "retrieval_context": test_result.retrieval_context
    }
    for metric_data in test_result.metrics_data:
        row[metric_data.name] = metric_data.score
        row[metric_data.name + "Reason"] = metric_data.reason
    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv("data/evaluation_results_goldens.csv", index=False, encoding='utf-8-sig')
print("Evaluation complete. Results saved.")
