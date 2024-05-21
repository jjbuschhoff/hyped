from unittest.mock import MagicMock, patch

import pytest
from datasets import Features, Sequence, Value
from openai import RateLimitError
from openai.types.chat.chat_completion import (
    ChatCompletion,
    ChatCompletionMessage,
    Choice,
    CompletionUsage,
)

from hyped.data.flow.processors.api.openai_chat import (
    OpenAIChatCompletion,
    OpenAIChatCompletionConfig,
)
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


async def dummy_chat_completion(*args, **kwargs):
    return ChatCompletion(
        id="0",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(
                    content="This is a completion message",
                    role="assistant",
                ),
            )
        ],
        created=0,
        model="dummy_model",
        object="chat.completion",
        usage=CompletionUsage(
            completion_tokens=10,
            prompt_tokens=15,
            total_tokens=25,
        ),
    )


class dummy_chat_completion_with_rate_limit(object):
    NUM_CALLS = 0

    @classmethod
    async def call(cls, *args, **kwargs):
        cls.NUM_CALLS += 1

        if cls.NUM_CALLS < 2:
            raise RateLimitError(
                "Dummy Rate Limit Error", response=MagicMock(), body=None
            )

        return await dummy_chat_completion(*args, **kwargs)


class TestOpenAIChatCompletion(BaseDataProcessorTest):
    # processor
    processor_type = OpenAIChatCompletion
    processor_config = OpenAIChatCompletionConfig()
    # input specification
    input_features = Features(
        {
            "messages": Sequence(
                {
                    "role": Value("string"),
                    "content": Value("string"),
                }
            )
        }
    )
    input_data = {
        "messages": [
            [
                {"role": "system", "content": "This is a system message"},
                {"role": "user", "content": "This is a user message"},
            ]
        ]
    }
    input_index = [0]

    @pytest.fixture(
        params=[
            dummy_chat_completion,
            dummy_chat_completion_with_rate_limit().call,
        ]
    )
    def processor(self, request):
        cls = type(self)
        # create a mock chat client to be used in the processor
        mock_chat_client = MagicMock()
        mock_chat_client.chat = MagicMock()
        mock_chat_client.chat.completions = MagicMock()
        mock_chat_client.chat.completions.create.side_effect = request.param
        # patch the async openai client with the mock client
        with patch(
            "hyped.data.flow.processors.api.openai_chat.AsyncOpenAI",
            return_value=mock_chat_client,
        ):
            return cls.processor_type.from_config(cls.processor_config)
