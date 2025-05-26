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
