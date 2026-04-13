from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from dotenv import load_dotenv, dotenv_values

# loading variables from .env file
load_dotenv()
subscription_key = os.getenv("azure_subscription_key")

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "Custom Azure OpenAI Model"

# Replace these with real values
custom_model = AzureChatOpenAI(
    openai_api_version='2024-12-01-preview',
    azure_deployment='gpt-4.1-students',
    azure_endpoint='https://510-foundry-research.cognitiveservices.azure.com/',
    openai_api_key=subscription_key,
)
