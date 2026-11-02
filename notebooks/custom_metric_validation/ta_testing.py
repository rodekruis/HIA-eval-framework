import os
import time
import uuid
import pandas as pd
from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.metrics.g_eval import Rubric
from deepeval.models import AzureOpenAIModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# Load the tone attunement target queries, and the positive and negative responses from the CSV file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ta_data = pd.read_csv(os.path.join(BASE_DIR, "../../synthetic_data/tone_attunement_responses.csv"))
ta_data = ta_data[:5]

# CUSTOM MODEL
subscription_key = os.getenv("azure_subscription_key")
custom_model = AzureOpenAIModel(
    model="gpt-4.1-students",
    api_key=subscription_key,
    azure_endpoint="https://510-ai-research.openai.azure.com/",
    api_version="2024-12-01-preview",
    deployment_name="gpt-4.1-students"
)

# TEST CASE CREATION
pos_test_cases = []
neg_test_cases = []
for _, row in ta_data.iterrows():
    pos_tc = LLMTestCase(
        input=row['user_input'],
        actual_output=row['positive_cluster']
    )
    pos_test_cases.append(pos_tc)

    neg_tc = LLMTestCase(
        input=row['user_input'],
        actual_output=row['negative_cluster']
    )
    neg_test_cases.append(neg_tc)

# METRIC CREATION
ta = GEval(
    name="Tone Attunement",
    criteria = "Determine whether the chatbot communication style appropriately matches the emotional register of the user's query" \
    "This is evaluated on two planes: 1) the chatbot should recognise and respond to the emotional state" \
    "expressed in the user's query. A high score means the emotion is addressed naturally; a low score means it is " \
    "ignored entirely, and 2) the level of empathy in the response should be appropriate for the query." \
    "A high score means the tone fits the situation; a low score means the response is either too cold for a distressed user," \
    "or excessively emotional for a neutral query." \
    "For example, the query 'I need a doctor.' requires a neutral, sligthly empathetic response, while the query " \
    "'I am running away from my abuser, who is searching for me and my children. I do not know where I am and I am very afraid. What can I do?'" \
    "requires a highly empathetic response that acknowledges the user's fear and distress, while providing practical guidance. ",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.80, model=custom_model,
    rubric=[
        Rubric(score_range=(0,0), expected_outcome="Emotion is ignored entirely or the response is excessively emotional for a neutral query or too cold for a distressed user."),
        Rubric(score_range=(1,1), expected_outcome="Emotion is addressed naturally and the response matches the user tone")]
)

# BATCH EVALUATION (due to clogging of the API, we will evaluate in batches of 3)
batch_size = 3
all_pos_results = []
all_neg_results = []

for i in range(0, len(pos_test_cases), batch_size):
    pos_batch = pos_test_cases[i:i + batch_size]
    neg_batch = neg_test_cases[i:i + batch_size]
    print(f"Evaluating batch {i // batch_size + 1} ({len(pos_batch)} pos / {len(neg_batch)} neg)...")
    all_pos_results.extend(evaluate(test_cases=pos_batch, metrics=[ta]).test_results)
    all_neg_results.extend(evaluate(test_cases=neg_batch, metrics=[ta]).test_results)
    time.sleep(10)

# OUTPUT AS DATAFRAME
# OUTPUT: build each side from its OWN results, then join on the query
pos_df = pd.DataFrame([{
    "query":               r.input,
    "pos_output":          r.actual_output,
    "pos_score":           r.metrics_data[0].score,
    "REASONING_pos_score": r.metrics_data[0].reason,
} for r in all_pos_results])

neg_df = pd.DataFrame([{
    "query":               r.input,            # <- the column that was missing before
    "neg_output":          r.actual_output,
    "neg_score":           r.metrics_data[0].score,
    "REASONING_neg_score": r.metrics_data[0].reason,
} for r in all_neg_results])

assert pos_df['query'].is_unique and neg_df['query'].is_unique
results_df = pos_df.merge(neg_df, on="query", how="inner", validate="one_to_one")

# VALIDATION: every output must match the source text for its query
src_pos = dict(zip(ta_data['user_input'], ta_data['positive_cluster']))
src_neg = dict(zip(ta_data['user_input'], ta_data['negative_cluster']))
assert len(results_df) == len(ta_data)
assert (results_df['query'].map(src_pos) == results_df['pos_output']).all()
assert (results_df['query'].map(src_neg) == results_df['neg_output']).all()

# Restore the source row order
order = {q: i for i, q in enumerate(ta_data['user_input'])}
results_df = results_df.sort_values('query', key=lambda s: s.map(order)).reset_index(drop=True)

results_df.to_csv(os.path.join(BASE_DIR, "ta_testing_results.csv"), index=False, encoding='utf-8-sig')
