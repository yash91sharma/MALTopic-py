import time
from typing import Optional, TYPE_CHECKING

from openai import OpenAI

if TYPE_CHECKING:
    from ..stats import MALTopicStats


class OpenAIClient:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        stats_tracker: Optional["MALTopicStats"] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.temperature = 0.2
        self.seed = 12345
        self.top_p = 0.9
        self.stats_tracker = stats_tracker

    def generate(self, *, instructions: str, input: str) -> str:
        start_time = time.time()

        try:
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

            response_time = time.time() - start_time

            if not response or not response.choices:
                raise ValueError("No response received from OpenAI API.")
            if not response.choices[0].message:
                raise ValueError("No message in the response from OpenAI API.")
            content = response.choices[0].message.content
            if content is None:
                raise ValueError("No content in the response message from OpenAI API.")

            # Extract token usage from OpenAI response
            input_tokens = 0
            output_tokens = 0
            total_tokens = 0

            if hasattr(response, "usage") and response.usage:
                input_tokens = getattr(response.usage, "prompt_tokens", 0)
                output_tokens = getattr(response.usage, "completion_tokens", 0)
                total_tokens = getattr(response.usage, "total_tokens", 0)

            # Record successful call in stats
            if self.stats_tracker:
                metadata = {
                    "model": (
                        response.model
                        if hasattr(response, "model")
                        else self.model_name
                    ),
                    "finish_reason": (
                        response.choices[0].finish_reason
                        if response.choices
                        else "unknown"
                    ),
                    "system_fingerprint": getattr(response, "system_fingerprint", None),
                }

                self.stats_tracker.record_successful_call(
                    model_name=self.model_name,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    response_time=response_time,
                    metadata=metadata,
                )

            return content

        except Exception as e:
            response_time = time.time() - start_time
            if self.stats_tracker:
                self.stats_tracker.record_failed_call(
                    model_name=self.model_name,
                    error_message=str(e),
                    response_time=response_time,
                )
            raise
