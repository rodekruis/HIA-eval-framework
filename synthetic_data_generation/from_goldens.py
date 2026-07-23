import ast
import os
import time
import pandas as pd
from dotenv import load_dotenv
from deepeval.dataset import Golden
from resources.model import custom_model
from deepeval.models import AzureOpenAIModel
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig
from deepeval.synthesizer.config import EvolutionConfig
from deepeval.synthesizer import Evolution, Synthesizer

# Azure OpenAI credentials
endpoint = "https://510-ai-research.openai.azure.com/"
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

df = pd.read_csv("../data/qa_pairs_context.csv")
goldens = []
for _, row in df.iterrows():
    context = row["context"]
    if pd.isna(context):
        context_list = None
    else:
        try:
            context_list = ast.literal_eval(context)
            if not isinstance(context_list, list):
                context_list = [str(context_list)]
        except (ValueError, SyntaxError):
            context_list = [str(context)]
    goldens.append(Golden(input=row["user_input"], context=context_list))

# # Styling and evolution configuration for the 21 original questions developed methodologically
# styling_config = StylingConfig(
#     scenario=("Undocumented migrants, asylum seekers, EU labour migrants, or residence permit holders/ individuals with Dutch nationality "
#               "seeking information regarding shelter, healthcare/ wellbeing, food, clothing, or safety/ protection in the Netherlands."),
#     task=("A cahtbot developed by the Red Cross that answers queries from vulnerable individuals seeking information regarding "
#           "shelter, healthcare/ wellbeing, food, clothing, or safety/ protection in the Netherlands."),
#     input_format=("User queries in English which can either be questions, inquiries, clarifications, statements, or unclear expressions of intent. "
#     "The queries may contain the following emotional undertones: fear, despair, fatigue, neutrality, frustration."\
#     "Not every query should be emotionally charged."\
#     "The queries can include either one topic or multiple from: Shelter, Healthcare/Wellbeing, Food, Clothing, Safety/Protection. Remember you are a vulnerable user in distress, not an analyst or a researcher. "\
#     "Occassionally, the queries should contain spelling errors or grammatical errors."\
#     "The queries should be written in informal, human-like language, and should not be overly formal or technical."\
#     "The queries will most likely be in first-person." \
#     "Example of human-like queries: 'I don't have a place to stay because I escaped from my trafficker. I don't know where to start.'; 'I have been to 3 hospitals, but they refuse me because I am undocumented. I am in immense pain and need immediate attention. I am in Amsterdam. What do I do?'; 'I lost my job and got kicked out of my house because I could not pay rent. I am from Poland. I have been homeless for a couple of days, and need to shower and do laundry.'; 'I need food!'"\
#     "Address the chatbot directly as if you are a vulnerable individual seeking information."\
#     "Don't mention that you are seeking help in the Netherlands, as the chatbot already knows this." \
#     "Not every query should be a question. Some queries should be statements or clarifications."\
#     "The queries should be diverse in terms of the topics they cover, the emotional undertones")
# )

# evolution_config = EvolutionConfig(
#     evolutions={
#         Evolution.REASONING: 0.1,
#         Evolution.COMPARATIVE: 0.1,
#         Evolution.HYPOTHETICAL: 0.3,
#         Evolution.IN_BREADTH: 0.4,
#     },
#     num_evolutions=4
# )

# # Styling and evolution configuration for the subset queries developed to target the Faithfulness metric
# styling_config = StylingConfig(
#     scenario=("Undocumented migrants, asylum seekers, EU labour migrants, or residence permit holders/ individuals with Dutch nationality "
#               "seeking information regarding shelter, healthcare/ wellbeing, food, clothing, or safety/ protection in the Netherlands."),
#     task=("A cahtbot developed by the Red Cross that answers queries from vulnerable individuals seeking information regarding "
#           "shelter, healthcare/ wellbeing, food, clothing, or safety/ protection in the Netherlands."),
#     input_format=("User queries in English which can either be questions, inquiries, clarifications, statements, or unclear expressions of intent. "
#     "The queries can include either one topic or multiple from: Shelter, Healthcare/Wellbeing, Food, Clothing, Safety/Protection."\
#     "The queries will most likely be in first-person." \
#     "Address the chatbot directly as if you are a vulnerable individual seeking information."\
#     "Don't mention that you are seeking help in the Netherlands, as the chatbot already knows this." \
#     "Not every query should be a question. Some queries should be statements or clarifications."\
#     "The queries should be diverse in terms of the topics they cover.")
# )

# evolution_config = EvolutionConfig(
#     evolutions={
#         Evolution.REASONING: 0.1,
#         Evolution.MULTICONTEXT: 0.4,
#         Evolution.CONCRETIZING: 0.3,
#         Evolution.COMPARATIVE: 0.1,
#         Evolution.IN_BREADTH: 0.1,
#     },
#     num_evolutions=4
# )

# Styling and evolution configuration for the subset queries developed to target the Cultural Neutrality metric
styling_config = StylingConfig(
    scenario=("Undocumented migrants, asylum seekers, EU labour migrants, or residence permit holders/ individuals with Dutch nationality "
              "seeking information regarding shelter, healthcare/ wellbeing, food, clothing, or safety/ protection in the Netherlands."),
    task=("A cahtbot developed by the Red Cross that answers queries from vulnerable individuals seeking information regarding "
          "shelter, healthcare/ wellbeing, food, clothing, or safety/ protection in the Netherlands."),
    input_format=("User queries in English which can either be questions, inquiries, clarifications, statements, or unclear expressions of intent. "
    "The queries can include either one topic or multiple from: Shelter, Healthcare/Wellbeing, Food, Clothing, Safety/Protection. Remember you are a vulnerable user in distress, not an analyst or a researcher. "\
    "Queries should ssometimes contain spelling errors or grammatical errors."\
    "The queries should be written in informal, human-like language, and should not be overly formal or technical."\
    "The queries will most likely be in first-person." \
    "Address the chatbot directly as if you are a vulnerable individual seeking information."\
    "Don't mention that you are seeking help in the Netherlands, as the chatbot already knows this." \
    "Not every query should be a question. Some queries should be statements or clarifications."\
    "The queries should be diverse in terms of the topics they cover"\
    "The queries MUST include one or more of the following: references to the user's nationality, legal status, cultural background, or literacy level.")
)

evolution_config = EvolutionConfig(
    evolutions={
        Evolution.REASONING: 0.1,
        Evolution.COMPARATIVE: 0.1,
        Evolution.HYPOTHETICAL: 0.3,
        Evolution.IN_BREADTH: 0.4,
    },
    num_evolutions=4
)

synthesizer = Synthesizer(model=custom_model, cost_tracking=False, async_mode=False, styling_config=styling_config)

# Generating new goldens in batches
BATCH_SIZE = 5          # how many goldens to process per batch
PAUSE_BETWEEN_BATCHES = 10  # seconds to rest between batches
MAX_GOLDENS_PER_GOLDEN = 5

all_results = []
failed_goldens = []

for i in range(0, len(goldens), BATCH_SIZE):
    batch = goldens[i:i + BATCH_SIZE]
    print(f"\n=== Processing batch {i // BATCH_SIZE + 1} ({len(batch)} goldens) ===")

    try:
        new_goldens = synthesizer.generate_goldens_from_goldens(
            goldens=batch,
            include_expected_output=False,
            max_goldens_per_golden=MAX_GOLDENS_PER_GOLDEN,
        )
        batch_df = synthesizer.to_pandas()
        all_results.append(batch_df)

        # checkpoint to disk after every batch, so a later crash doesn't lose earlier work
        pd.concat(all_results, ignore_index=True).to_csv("synthetic_data/checkpoint_goldens.csv", index=False)
        print(f"Batch {i // BATCH_SIZE + 1} succeeded, checkpoint saved.")

    except AttributeError as e:
        print(f"Batch {i // BATCH_SIZE + 1} FAILED: {e}")
        failed_goldens.extend(batch)

    if i + BATCH_SIZE < len(goldens):
        print(f"Pausing {PAUSE_BETWEEN_BATCHES}s before next batch...")
        time.sleep(PAUSE_BETWEEN_BATCHES)

final_df = pd.concat(all_results, ignore_index=True)
print(f"\nDone. {len(final_df)} goldens generated. {len(failed_goldens)} goldens failed and can be retried.")

# Keep only the columns you need
export_df = final_df[["input", "expected_output", "context"]]
export_df.to_csv("synthetic_data/cultural_neutrality_target.csv", index=False, encoding="utf-8-sig")
print(f"Saved {len(export_df)} goldens to synthetic_data/cultural_neutrality_target.csv")
