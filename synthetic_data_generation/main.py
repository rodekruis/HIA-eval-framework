import os
import pandas as pd
import numpy as np
from deepeval.synthesizer import Synthesizer
from resources.model import custom_model
from deepeval.models import AzureOpenAIModel
from dotenv import load_dotenv, dotenv_values

# Read source file for generating synthetic data
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(SCRIPT_DIR, "hia_undocumented_migrants_faq_en.xlsx")
baseline = pd.read_excel(file_path)
# Transform the data into a format suitable for the synthesizer
baseline = baseline.dropna() # Drop rows with any missing values
baseline = np.array(baseline)

# Create a list of question-answer pairs as strings (expected format for the synthesizer)
qa_pairs = [(baseline[i][0], baseline[i][1]) for i in range(len(baseline))]
qa_strings = [f"Q: {q} A: {a}" for q, a in qa_pairs]

# Azure OpenAI credentials
endpoint = "https://510-foundry-research.cognitiveservices.azure.com/"
deployment = "gpt-4.1-students"
api_version = "2024-12-01-preview"

# loading variables from .env file
load_dotenv()
subscription_key = os.getenv("azure_subscription_key")

# Patch the AzureOpenAIModel to handle None cost values gracefully
class PatchedAzureModel(AzureOpenAIModel):
    async def a_generate(self, prompt, schema=None):
        result = await super().a_generate(prompt, schema)
        # result is (response, cost) - force cost to 0 if None
        if isinstance(result, tuple):
            return (result[0], result[1] if result[1] is not None else 0)
        return result

    def generate(self, prompt, schema=None):
        result = super().generate(prompt, schema)
        if isinstance(result, tuple):
            return (result[0], result[1] if result[1] is not None else 0)
        return result

custom_model = PatchedAzureModel(
    model=deployment,
    api_key=subscription_key,
    azure_endpoint=endpoint,
    api_version=api_version,
    deployment_name=deployment
)

# Filter out entries where the answer is NaN
qa_pairs = [(q, a) for q, a in qa_pairs if pd.notna(a)]

# Assuming all in one context, or you can split into chunks
qa_strings = [f"Q: {q} A: {a}" for q, a in qa_pairs]

# Split into multiple contexts for more diverse synthetic data
chunk_size = 10
contexts = [qa_strings[i:i + chunk_size] for i in range(0, len(qa_strings), chunk_size)]

synthesizer = Synthesizer(model=custom_model, cost_tracking=False, async_mode=False)
goldens = synthesizer.generate_goldens_from_contexts(
    contexts=contexts,
    include_expected_output=True,
    max_goldens_per_context=17,  # Increase to generate more per context
)

os.makedirs("./synthetic_data", exist_ok=True)

data = [{
    "input": g.input,
    "expected_output": g.expected_output,
    "context": g.context,
} for g in goldens]

df = pd.DataFrame(data)
df.to_csv("./synthetic_data/goldens.csv", index=False, encoding="utf-8-sig")
print(f"Saved {len(df)} goldens to ./synthetic_data/goldens.csv")
