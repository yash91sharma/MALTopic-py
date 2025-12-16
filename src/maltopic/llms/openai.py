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
        override_model_params: Optional[dict] = None,
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        self.temperature = 0.2
        self.seed = 12345
        self.top_p = 0.9
        self.stats_tracker = stats_tracker
        self.override_model_params = override_model_params

    def _is_reasoning_model(self, model_name: str) -> bool:
        """
        Check if the model is a reasoning model that has parameter restrictions.
        
        These models don't support: temperature, top_p, presence_penalty, frequency_penalty, seed
        
        Includes:
        - o1 series: o1-preview, o1-mini
        - o3 series: o3, o3-mini, o3.5
        - gpt-5 series: gpt-5-mini, gpt-5-preview, etc.
        """
        model_lower = model_name.lower()
        
        # Check for o1/o3 series
        reasoning_prefixes = ['o1', 'o3']
        if any(model_lower.startswith(prefix) for prefix in reasoning_prefixes):
            return True
        
        # Check for gpt-5 series (reasoning models)
        if 'gpt-5' in model_lower:
            return True
            
        return False

    def generate(self, *, instructions: str, input: str) -> str:
        start_time = time.time()

        try:
            # Build base parameters
            params = {
                "model": self.model_name,
                "store": False,
                "messages": [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": input},
                ],
            }
            
            # If override_model_params is provided, use it exclusively
            if self.override_model_params is not None:
                params.update(self.override_model_params)
            else:
                # Only add sampling parameters if not a reasoning model
                if not self._is_reasoning_model(self.model_name):
                    params["temperature"] = self.temperature
                    params["top_p"] = self.top_p
                    params["seed"] = self.seed
            
            response = self.client.chat.completions.create(**params)

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
