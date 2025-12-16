from typing import Optional
from unittest.mock import patch

from src.maltopic.llms.openai import OpenAIClient


class DummyMessage:
    def __init__(self, content):
        self.content = content


class DummyChoice:
    def __init__(self, message):
        self.message = message


class DummyResponse:
    def __init__(self, content):
        self.choices = [DummyChoice(DummyMessage(content))]


class DummyCompletions:
    def __init__(
        self, output_text: Optional[str], with_message=True, with_choices=True
    ):
        self._output_text = output_text
        self.last_kwargs = None
        self.with_message = with_message
        self.with_choices = with_choices

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if not self.with_choices:

            class NoChoicesResponse:
                choices = []

            return NoChoicesResponse()
        if not self.with_message:

            class NoMessage:
                pass

            class NoMessageChoice:
                message = None

            class NoMessageResponse:
                choices = [NoMessageChoice()]

            return NoMessageResponse()
        return DummyResponse(self._output_text)


class DummyChat:
    def __init__(self, output_text, with_message=True, with_choices=True):
        self.completions = DummyCompletions(output_text, with_message, with_choices)


class DummyOpenAI:
    def __init__(
        self,
        api_key=None,
        output_text: Optional[str] = "dummy output",
        with_message=True,
        with_choices=True,
    ):
        self.api_key = api_key
        self.chat = DummyChat(output_text, with_message, with_choices)
        self.api_key = api_key
        self.chat = DummyChat(output_text, with_message, with_choices)


class TestOpenAIClient:
    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key),
    )
    def test_openai_client_init(self, mock_api_key="test-key", mock_model_name="gpt-4"):
        client = OpenAIClient(api_key=mock_api_key, model_name=mock_model_name)
        assert client.api_key == mock_api_key
        assert client.model_name == mock_model_name
        assert isinstance(client.client, DummyOpenAI)
        assert client.temperature == 0.2
        assert client.seed == 12345
        assert client.top_p == 0.9

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, output_text="output"),
    )
    def test_generate_calls_create_with_correct_args(self):
        client = OpenAIClient(api_key="k", model_name="m")
        result = client.generate(instructions="do this", input="my input")
        assert result == "output"
        last_kwargs = client.client.chat.completions.last_kwargs  # type: ignore[attr-defined]
        assert last_kwargs["model"] == "m"
        assert last_kwargs["messages"] == [
            {"role": "system", "content": "do this"},
            {"role": "user", "content": "my input"},
        ]
        assert last_kwargs["temperature"] == 0.2
        assert last_kwargs["top_p"] == 0.9
        assert last_kwargs["seed"] == 12345
        assert last_kwargs["store"] is False

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(
            api_key=api_key, output_text="expected output"
        ),
    )
    def test_generate_returns_output_text(self):
        client = OpenAIClient(api_key="k", model_name="m")
        out = client.generate(instructions="foo", input="bar")
        assert out == "expected output"

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, output_text="zzz"),
    )
    def test_generate_with_different_params(self):
        client = OpenAIClient(api_key="k2", model_name="m2")
        client.temperature = 0.7
        client.top_p = 0.5
        out = client.generate(instructions="abc", input="xyz")
        assert out == "zzz"
        last_kwargs = client.client.chat.completions.last_kwargs  # type: ignore[attr-defined]
        assert last_kwargs["temperature"] == 0.7
        assert last_kwargs["top_p"] == 0.5

    def test_openai_client_repr(self):
        client = OpenAIClient.__new__(OpenAIClient)
        assert "OpenAIClient" in str(client)

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, with_choices=False),
    )
    def test_generate_raises_on_no_choices(self):
        client = OpenAIClient(api_key="k", model_name="m")
        try:
            client.generate(instructions="foo", input="bar")
        except ValueError as e:
            assert "No response received from OpenAI API." in str(e)
        else:
            assert False, "Expected ValueError for no choices"

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, with_message=False),
    )
    def test_generate_raises_on_no_message(self):
        client = OpenAIClient(api_key="k", model_name="m")
        try:
            client.generate(instructions="foo", input="bar")
        except ValueError as e:
            assert "No message in the response from OpenAI API." in str(e)
        else:
            assert False, "Expected ValueError for no message"

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, output_text=None),
    )
    def test_generate_raises_on_no_content(self):
        client = OpenAIClient(api_key="k", model_name="m")
        try:
            client.generate(instructions="foo", input="bar")
        except ValueError as e:
            assert "No content in the response message from OpenAI API." in str(e)
        else:
            assert False, "Expected ValueError for no content"

    def test_is_reasoning_model_o1_series(self):
        """Test that o1 models are correctly identified as reasoning models"""
        client = OpenAIClient.__new__(OpenAIClient)
        assert client._is_reasoning_model("o1-preview") is True
        assert client._is_reasoning_model("o1-mini") is True
        assert client._is_reasoning_model("O1-preview") is True  # case insensitive

    def test_is_reasoning_model_o3_series(self):
        """Test that o3 models are correctly identified as reasoning models"""
        client = OpenAIClient.__new__(OpenAIClient)
        assert client._is_reasoning_model("o3-mini") is True
        assert client._is_reasoning_model("o3") is True
        assert client._is_reasoning_model("O3-mini") is True  # case insensitive

    def test_is_reasoning_model_gpt5_series(self):
        """Test that gpt-5 models are correctly identified as reasoning models"""
        client = OpenAIClient.__new__(OpenAIClient)
        assert client._is_reasoning_model("gpt-5-mini") is True
        assert client._is_reasoning_model("gpt-5-preview") is True
        assert client._is_reasoning_model("GPT-5-mini") is True  # case insensitive

    def test_is_not_reasoning_model(self):
        """Test that regular GPT models are not identified as reasoning models"""
        client = OpenAIClient.__new__(OpenAIClient)
        assert client._is_reasoning_model("gpt-4") is False
        assert client._is_reasoning_model("gpt-4-turbo") is False
        assert client._is_reasoning_model("gpt-3.5-turbo") is False
        assert client._is_reasoning_model("gpt-4o") is False
        assert client._is_reasoning_model("gpt-4o-mini") is False

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, output_text="output"),
    )
    def test_generate_excludes_params_for_reasoning_models(self):
        """Test that temperature, top_p, and seed are not sent for reasoning models"""
        client = OpenAIClient(api_key="k", model_name="o1-preview")
        result = client.generate(instructions="do this", input="my input")
        assert result == "output"
        last_kwargs = client.client.chat.completions.last_kwargs  # type: ignore[attr-defined]
        assert last_kwargs["model"] == "o1-preview"
        assert last_kwargs["messages"] == [
            {"role": "system", "content": "do this"},
            {"role": "user", "content": "my input"},
        ]
        # These parameters should NOT be present for reasoning models
        assert "temperature" not in last_kwargs
        assert "top_p" not in last_kwargs
        assert "seed" not in last_kwargs
        assert last_kwargs["store"] is False

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, output_text="output"),
    )
    def test_generate_includes_params_for_regular_models(self):
        """Test that temperature, top_p, and seed ARE sent for regular models"""
        client = OpenAIClient(api_key="k", model_name="gpt-4")
        result = client.generate(instructions="do this", input="my input")
        assert result == "output"
        last_kwargs = client.client.chat.completions.last_kwargs  # type: ignore[attr-defined]
        # These parameters SHOULD be present for regular models
        assert "temperature" in last_kwargs
        assert "top_p" in last_kwargs
        assert "seed" in last_kwargs
        assert last_kwargs["temperature"] == 0.2
        assert last_kwargs["top_p"] == 0.9
        assert last_kwargs["seed"] == 12345

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, output_text="output"),
    )
    def test_override_model_params_with_custom_values(self):
        """Test that override_model_params overrides all default parameters"""
        override_params = {
            "temperature": 0.8,
            "max_tokens": 100,
            "top_p": 0.95,
        }
        client = OpenAIClient(
            api_key="k", model_name="gpt-4", override_model_params=override_params
        )
        result = client.generate(instructions="do this", input="my input")
        assert result == "output"
        last_kwargs = client.client.chat.completions.last_kwargs  # type: ignore[attr-defined]
        # Override parameters should be present
        assert last_kwargs["temperature"] == 0.8
        assert last_kwargs["max_tokens"] == 100
        assert last_kwargs["top_p"] == 0.95
        # Default seed should NOT be present (not in override)
        assert "seed" not in last_kwargs

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, output_text="output"),
    )
    def test_override_model_params_for_reasoning_models(self):
        """Test that override_model_params works even for reasoning models"""
        override_params = {
            "max_tokens": 500,
        }
        client = OpenAIClient(
            api_key="k", model_name="o1-preview", override_model_params=override_params
        )
        result = client.generate(instructions="do this", input="my input")
        assert result == "output"
        last_kwargs = client.client.chat.completions.last_kwargs  # type: ignore[attr-defined]
        # Override parameters should be present
        assert last_kwargs["max_tokens"] == 500
        # Default parameters should NOT be present
        assert "temperature" not in last_kwargs
        assert "top_p" not in last_kwargs
        assert "seed" not in last_kwargs

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, output_text="output"),
    )
    def test_override_model_params_empty_dict(self):
        """Test that empty override_model_params dict removes all default parameters"""
        client = OpenAIClient(
            api_key="k", model_name="gpt-4", override_model_params={}
        )
        result = client.generate(instructions="do this", input="my input")
        assert result == "output"
        last_kwargs = client.client.chat.completions.last_kwargs  # type: ignore[attr-defined]
        # No default parameters should be present
        assert "temperature" not in last_kwargs
        assert "top_p" not in last_kwargs
        assert "seed" not in last_kwargs
        # Only base parameters should be present
        assert last_kwargs["model"] == "gpt-4"
        assert last_kwargs["store"] is False
        assert "messages" in last_kwargs

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(api_key=api_key, output_text="output"),
    )
    def test_override_model_params_none_uses_defaults(self):
        """Test that override_model_params=None uses default behavior"""
        client = OpenAIClient(
            api_key="k", model_name="gpt-4", override_model_params=None
        )
        result = client.generate(instructions="do this", input="my input")
        assert result == "output"
        last_kwargs = client.client.chat.completions.last_kwargs  # type: ignore[attr-defined]
        # Default parameters should be present for regular models
        assert last_kwargs["temperature"] == 0.2
        assert last_kwargs["top_p"] == 0.9
        assert last_kwargs["seed"] == 12345
