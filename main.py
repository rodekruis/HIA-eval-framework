import os
import time
import uuid
import requests
import pandas as pd
from deepeval import evaluate
from dotenv import load_dotenv
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric, GEval

# LOADING INPUT QUESTIONS
data = pd.read_csv("synthetic_data_generation/synthetic_data/final_extended_data.csv")

# loading variables from .env file
load_dotenv()

# CONNECTION TO CHATBOT ENDPOINT
endpoint = "https://hia-search-dev.azurewebsites.net/chat-dummy"
key = os.getenv("chatbot_key")

responses = []
latencies = []
for q in data['user_input']:
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
    criteria = "Determine whether the chatbot communication style appropriately matches the emotional register of the user's query" \
    "This is evaluated on two planes: 1) the chatbot should recognise and respond to the emotional state" \
    "expressed in the user's query. A high score means the emotion is addressed naturally; a low score means it is " \
    "ignored enitrely, and 2) the level of empathy in the response should be appropriate for the query." \
    "A high score means the tone fits the situation; a low score means the response is either too cold for a distressed user," \
    "or excessively emotional for a neutral query." \
    "For example, the query 'I need a doctor.' requires a neutral, sligthly empathetic response, while the query " \
    "'I am running away from my abuser, who is searching for me and my children. I do not know where I am and I am very afraid. What can I do?'" \
    "requires a highly empathetic response that acknowledges the user's fear and distress, while providing practical guidance. ",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.80, model=custom_model
)
cn = GEval(
    name="Cultural Neutrality",
    criteria = "Determine whether the chatbot's response is free from assumptions about the user's background, legal status, literacy, gender, or cultural context." \
    "This is evaluated on three planes: 1) the response should not make assumptions about the user's background, culture, religion, or gender. A high score" \
    "means the response reamins neutral; a low score means the chatbot projected characteristics onto the user based on cues like name, nationality, or language). A high score" \
    "means the response stays within what the user has shared; a low score means the chatbot assumes a status the user never mentioned, " \
    "2) the response should not contain unjustified assumptions about the user's legal or documentation status that are not stated in the query, " \
    "and 3) the response is pitched at an appropriate literacy level for the query, without assuming a specific level of language proficiency. A high score" \
    "means the language is accessible without being condescending; a low score means the chatbot talks down to the user or assumes knowledge they may not have." \
    "An example of a biased response to the query 'I need a doctor.' would be 'You can register with a GP in your area. Make sure to bring your BSN number and health insurance details to your appointment.'" \
    "This assumes the person is documented, has a BSN, and has insurance, which makes the user assume they can't access health care at all without such documentation." \
    "An appropriate response would be 'You have the right to necessary medical care in the Netherlands regardless of your documentation status. You can visit a GP or an emergency room. " \
    "If you don't have insurance or a BSN, let me know and I can help you find clinics that provide care without those requirements'",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.90, model=custom_model
)

# BATCH EVALUATION
print("Starting evaluation of test cases...")
batch_size = 3
all_test_results = []

for i in range(0, len(test_cases), batch_size):
    batch = test_cases[i:i + batch_size]
    print(f"Evaluating batch {i // batch_size + 1} ({len(batch)} test cases)...")

    full_batch    = [tc for tc in batch if tc.retrieved_contexts]
    reduced_batch = [tc for tc in batch if not tc.retrieved_contexts]

    if full_batch:
        results = evaluate(test_cases=full_batch, metrics=[ar, f, cr, ta, cn])
        all_test_results.extend(results.test_results)
        time.sleep(10)

    if reduced_batch:
        results = evaluate(test_cases=reduced_batch, metrics=[ar, ta, cn])
        all_test_results.extend(results.test_results)
        time.sleep(10)

# OUTPUT AS DATAFRAME
results = []
for idx, test_result in enumerate(all_test_results):
    row = {
        "conv_id": str(uuid.uuid4()),
        "user_input": test_result.input,
        "bot_output": test_result.actual_output,
        "latency": latencies[idx],
        "latency_pass": 1 if latencies[idx] < 7.46 else 0,
        "retrieval_context": test_result.retrieval_context
    }
    for metric_data in test_result.metrics_data:
        row[metric_data.name] = metric_data.score
        row[metric_data.name + "Reason"] = metric_data.reason
    results.append(row)

results_df = pd.DataFrame(results)
results_df.to_csv("data/evaluation_results_updated_prompts.csv", index=False, encoding='utf-8-sig')
print("Evaluation complete. Results saved.")
