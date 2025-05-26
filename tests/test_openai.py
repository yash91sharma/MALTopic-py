from unittest.mock import patch

from src.maltopic.llms.openai import OpenAIClient


class DummyResponse:
    def __init__(self, output_text):
        self.output_text = output_text


class DummyResponses:
    def __init__(self, output_text):
        self._output_text = output_text
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return DummyResponse(self._output_text)


class DummyOpenAI:
    def __init__(self, api_key=None, responses=None):
        self.api_key = api_key
        self.responses = responses or DummyResponses("dummy output")


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
        new=lambda api_key=None: DummyOpenAI(
            api_key=api_key, responses=DummyResponses("output")
        ),
    )
    def test_generate_calls_create_with_correct_args(self):
        client = OpenAIClient(api_key="k", model_name="m")
        result = client.generate(instructions="do this", input="my input")
        assert result == "output"
        last_kwargs = client.client.responses.last_kwargs  # type: ignore[attr-defined]
        assert last_kwargs["model"] == "m"
        assert last_kwargs["instructions"] == "do this"
        assert last_kwargs["input"] == "my input"
        assert last_kwargs["temperature"] == 0.2
        assert last_kwargs["top_p"] == 0.9

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(
            api_key=api_key, responses=DummyResponses("expected output")
        ),
    )
    def test_generate_returns_output_text(self):
        client = OpenAIClient(api_key="k", model_name="m")
        out = client.generate(instructions="foo", input="bar")
        assert out == "expected output"

    @patch(
        "src.maltopic.llms.openai.OpenAI",
        new=lambda api_key=None: DummyOpenAI(
            api_key=api_key, responses=DummyResponses("zzz")
        ),
    )
    def test_generate_with_different_params(self):
        client = OpenAIClient(api_key="k2", model_name="m2")
        client.temperature = 0.7
        client.top_p = 0.5
        out = client.generate(instructions="abc", input="xyz")
        assert out == "zzz"
        last_kwargs = client.client.responses.last_kwargs  # type: ignore[attr-defined]
        assert last_kwargs["temperature"] == 0.7
        assert last_kwargs["top_p"] == 0.5

    def test_openai_client_repr(self):
        client = OpenAIClient.__new__(OpenAIClient)
        assert "OpenAIClient" in str(client)
