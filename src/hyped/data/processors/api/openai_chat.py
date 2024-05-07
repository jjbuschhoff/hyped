"""OpenAI LLM API Data Processor."""
import asyncio
import random
import warnings
from contextlib import nullcontext
from functools import partial
from typing import Annotated, Any, Literal

from datasets import Features, Sequence, Value
from openai import AsyncOpenAI, RateLimitError
from openai._constants import DEFAULT_MAX_RETRIES
from pydantic import Field
from typing_extensions import TypedDict

from hyped.common.feature_checks import raise_feature_is_sequence
from hyped.common.feature_key import FeatureCollection, FeatureKey
from hyped.common.lazy import LazyInstance
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


class OpenAIToolFunction(TypedDict):
    """OpenAI compatible Tool Function."""

    description: str
    """description of the function"""
    name: str
    """name of the function"""
    parameters: object
    """description of the function parameters in json-schema format"""


class OpenAITool(TypedDict):
    """OpenAI compatible Tool."""

    type: Literal["function"]
    """type of the tool, currently only supports `function`"""
    function: OpenAIToolFunction
    """description of the function"""


class OpenAIChatCompletionConfig(BaseDataProcessorConfig):
    """OpenAI Chat Completion Data Processor Config.

    For more information about the arguments, please refer to the
    openai documentation.

    Attributes:
        messages (FeatureKey | FeatureCollection):
            input chat messages, following the openai chat message format.
            This can either be a feature key, refering to a dataset feature
            already in the correct format, or a feature collection describing
            the input feature.
        rate_limit_max_retries (int):
            Maximum number of retries in case of rate limit error. Defaults
            to 3.
        rate_limit_exp_backoff (int):
            Exponential backoff factor (i.e. the base of the exponent) for
            rate limit error handling. Defaults to 2.
        max_concurrent_calls (None | int):
            the maximum number of concurrent calls to the api. When using
            multiple processes, each process can have up to a total of
            `max_concurrent_calls` calls to the api at a time.

    Client Arguments:
        api_key (str):
            By default, this is loaded from the `OPENAI_API_KEY` environment
            variable.
        organization (str):
            By default, this is loaded from the `OPENAI_ORG_ID` environment
            variable.
        project (str):
            By default, this is loaded from the `OPENAI_PROJECT_ID` environment
            variable.
        base_url (str):
            Base API URL.
        timeout (str):
            Timeout duration.
        max_retries (int):
            Maximum number of retries.
        default_headers (dict[str, str]):
            Default headers.
        default_query (dict[str, str]):
            Default query parameters.

    Completion Arguments:
        model (str):
            id of the model to use
        frequency_penalty (float):
            Number between -2.0 and 2.0. Positive values penalize new tokens
            based on their existing frequency in the text so far, decreasing
            the model's likelihood to repeat the same line verbatim.
        presence_penalty (float):
            Number between -2.0 and 2.0. Positive values penalize new tokens
            based on whether they appear in the text so far, increasing the
            model's likelihood to talk about new topics.
        logit_bias (dict[int, float]):
            Modify the likelihood of specified tokens appearing in the
            completion. See the openai documentation for more information.
        logprobs (bool):
            Whether to return log probabilities of the output tokens or not.
            If true, returns the log probabilities of each output token
            returned in the content of message. Default to `False`.
        top_logprobs (None | int):
            An integer between 0 and 20 specifying the number of most likely
            tokens to return at each token position, each with an associated
            log probability. logprobs must be set to true if this parameter
            is used.
        temperature (float):
            What sampling temperature to use, between 0 and 2. Higher values
            like 0.8 will make the output more random, while lower values like
            0.2 will make it more focused and deterministic.
        top_p (float):
            An alternative to sampling with temperature, called nucleus sampling,
            where the model considers the results of the tokens with top_p
            probability mass. So 0.1 means only the tokens comprising the top
            10% probability mass are considered.
        max_tokens (int):
            The maximum number of tokens that can be generated in the chat
            completion.
        tools (None | list[OpenAITool]):
            A list of tools the model may call. Currently, only functions are
            supported as a tool. Use this to provide a list of functions the
            model may generate JSON inputs for. A max of 128 functions are
            supported.
        tool_choice (str | OpenAITool):
            Controls which (if any) tool is called by the model. `none` means the
            model will not call any tool and instead generates a message. `auto`
            means the model can pick between generating a message or calling one
            or more tools. `required` means the model must call one or more tools.
            Specifying a particular tool via
            `{"type": "function", "function": {"name": "my_function"}}`
            forces the model to call that tool.
        response_format (None | dict[str, str]):
            An object specifying the format that the model must output. Setting
            to `{ "type": "json_object" }` enables JSON mode, which guarantees the
            message the model generates is valid JSON.
        seed (None | int):
            If specified, the openai system will make a best effort to sample
            deterministically, such that repeated requests with the same seed
            and parameters should return the same result.
        stop (None | str):
            Sequence where the API will stop generating further tokens.
        extra_headers (None | dict[str, str]):
            Send extra headers
        extra_query (None | dict[str, str]):
            Add additional query parameters to the request
        extra_body (None | dict[str, str]):
            Add additional JSON properties to the request
    """

    messages: FeatureCollection | FeatureKey
    max_concurrent_calls: None | int = None
    rate_limit_max_retries: int = 10
    rate_limit_exp_backoff: int = 2
    # client arguments
    api_key: str | None = None
    organization: str | None = None
    project: str | None = None
    base_url: str | None = None
    timeout: float | None = None
    max_retries: int = DEFAULT_MAX_RETRIES
    default_headers: dict[str, str] | None = None
    default_query: dict[str, str] | None = None
    # completion arguments
    model: str = "gpt-3.5-turbo-0125"
    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    presence_penalty: float = Field(default=0, ge=-2, le=2)
    logit_bias: dict[int, Annotated[float, Field(ge=-100, le=100)]] = {}
    logprobs: bool = False
    top_logprobs: None | int = None
    temperature: float = Field(default=1, ge=0, le=2)
    top_p: float = Field(default=1, ge=0, le=1)
    max_tokens: None | int = None
    tools: None | list[OpenAITool] = None
    tool_choice: None | str | OpenAITool = None
    response_format: None | dict[str, str] = None
    seed: None | int = None
    stop: None | str = None
    # extra
    extra_headers: None | dict[str, str] = None
    extra_query: None | dict[str, str] = None
    extra_body: None | dict[str, str] = None


class OpenAIChatCompletion(BaseDataProcessor[OpenAIChatCompletionConfig]):
    """OpenAI Chat Completion Data Processor.

    Check the following example on how to use the processor
    using a feature collection:

    .. code-block:: python

        processor = OpenAIChatCompletion(
            OpenAIChatCompletionConfig(
                model=...,
                messages=FeatureCollection(
                    [
                        {
                            "role": Const("system"),
                            "content": Const("This is a system message")
                        },
                        {
                            "role": Const("user"),
                            "content": FeatureKey("prompt")
                        }
                    ]
                )
            )
        )


    MacOS users might have to set the following environment variable when using
    this processor in a multiprocessing setting:

    .. code-block:: bash

        OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    """

    def __init__(self, config: OpenAIChatCompletionConfig) -> None:
        """Construct a new processor.

        Arguments:
            config (OpenAIChatCompletionConfig):
                configuration of the processor
        """
        super(OpenAIChatCompletion, self).__init__(config)
        # create semaphore object to control the maximum
        # number of concurrent calls to the api
        self.sem = (
            asyncio.Semaphore(value=self.config.max_concurrent_calls)
            if self.config.max_concurrent_calls is not None
            else nullcontext()
        )
        # create a lazy instance of the openai client
        self.client = LazyInstance(
            partial(
                AsyncOpenAI,
                api_key=self.config.api_key,
                organization=self.config.organization,
                project=self.config.project,
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                max_retries=self.config.max_retries,
                default_headers=self.config.default_headers,
                default_query=self.config.default_query,
            )
        )

        # check if there are any non-constant values in the messages
        if (
            isinstance(self.config.messages, FeatureCollection)
            and len(list(self.config.messages.feature_keys)) == 0
        ):
            warnings.warn(
                "The specified messages do not contain any variables, i.e. "
                "FeatureKeys. The prompt will be constant for all examples.",
                UserWarning,
            )

    def map_features(self, features: Features) -> Features:
        """Map dataset features.

        Arguments:
            features (Features):
                input dataset features

        Returns:
            out (Features):
                completion features including the completion message
                as well as meta information such as the token usage.
        """
        # check the messages feature
        raise_feature_is_sequence(
            self.config.messages,
            self.config.messages.index_features(features),
            Features({"role": Value("string"), "content": Value("string")}),
        )

        return {
            "completion": {
                "run_id": Value("string"),
                "message": Value("string"),
                "logprobs": Sequence(
                    {
                        "token": Value("string"),
                        "logprob": Value("float32"),
                        "top_logprobs": Sequence(
                            {
                                "token": Value("string"),
                                "logprob": Value("float32"),
                            },
                            length=0
                            if self.config.top_logprobs is None
                            else self.config.top_logprobs,
                        ),
                    }
                ),
                "tool_calls": Sequence(
                    {
                        "type": Value("string"),
                        "function": {
                            "name": Value("string"),
                            "arguments": Value("string"),
                        },
                    }
                ),
                "usage": {
                    "completion_tokens": Value("int32"),
                    "prompt_tokens": Value("int32"),
                    "total_tokens": Value("int32"),
                },
            }
        }

    async def api_call(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """OpenAI API Call.

        Arguments:
            example (dict[str, Any]):
                example to process
            index (int):
                dataset index of the example
            rank (int):
                execution process rank

        Returns:
            out (dict[str, Any]):
                processed example
        """
        resp = await self.client.chat.completions.create(
            messages=self.config.messages.index_example(example),
            model=self.config.model,
            frequency_penalty=self.config.frequency_penalty,
            presence_penalty=self.config.presence_penalty,
            logit_bias=self.config.logit_bias,
            logprobs=self.config.logprobs,
            top_logprobs=self.config.top_logprobs,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            tools=self.config.tools,
            tool_choice=self.config.tool_choice,
            response_format=self.config.response_format,
            seed=self.config.seed,
            stop=self.config.stop,
            extra_headers=self.config.extra_headers,
            extra_query=self.config.extra_query,
            extra_body=self.config.extra_body,
        )

        return {
            "run_id": resp.id,
            "completion": {
                "message": resp.choices[0].message.content,
                "logprobs": (
                    None
                    if not self.config.logprobs
                    else [
                        {
                            "token": token.token,
                            "logprob": token.logprob,
                            "top_logprobs": (
                                None
                                if self.config.top_logprobs is None
                                else [
                                    {
                                        "token": top_token.token,
                                        "logprob": top_token.logprob,
                                    }
                                    for top_token in token.top_logprobs
                                ]
                            ),
                        }
                        for token in resp.choices[0].logprobs.content
                    ]
                ),
                "function_call": resp.choices[0].message.function_call,
                "tool_calls": resp.choices[0].message.tool_calls,
            },
            "usage": {
                "completion_tokens": resp.usage.completion_tokens,
                "prompt_tokens": resp.usage.prompt_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
        }

    async def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Process example.

        Implements basic error handling for api call.

        Arguments:
            example (dict[str, Any]):
                example to process
            index (int):
                dataset index of the example
            rank (int):
                execution process rank

        Returns:
            out (dict[str, Any]):
                processed example
        """
        # TODO: outsource this logic into a base api data processor
        with self.sem:
            for i in range(0, 1 + self.config.rate_limit_max_retries):
                try:
                    return await self.api_call(example, index, rank)
                except RateLimitError:
                    # Increment the delay
                    delay = self.config.rate_limit_exp_backoff ** (
                        1 + i + random.random()
                    )
                    warnings.warn(
                        "API rate limit exceeded. Retrying in %.01f seconds."
                        % delay,
                        UserWarning,
                    )
                    await asyncio.sleep(delay)

            raise Exception("Maximum number of retries exceeded.")
