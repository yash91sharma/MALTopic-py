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
        response = self.client.chat.completions.create(
            model=self.model_name,
            store=False,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": input},
            ],
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed,
        )
        if not response or not response.choices:
            raise ValueError("No response received from OpenAI API.")
        if not response.choices[0].message:
            raise ValueError("No message in the response from OpenAI API.")
        content = response.choices[0].message.content
        if content is None:
            raise ValueError("No content in the response message from OpenAI API.")
        return content
