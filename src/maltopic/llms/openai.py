from openai import OpenAI


class OpenAIClient:
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.temperature = 0.2
        self.seed = 12345
        self.top_p = 0.9

    def generate(self, *, instructions: str, input: str) -> str:
        response = self.client.responses.create(
            model=self.model_name,
            instructions=instructions,
            input=input,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return response.output_text
