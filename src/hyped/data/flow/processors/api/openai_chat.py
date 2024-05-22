"""Module for handling interactions with the OpenAI Chat Completion API.

This module provides classes and utilities for processing text data using the
OpenAI Chat Completion API. It includes a processor class,
`OpenAIChatCompletion`, which allows users to interact with the API to generate
chat completions based on input messages. The processor handles API calls, rate
limiting, and parsing of response data.
"""
import asyncio
import random
import warnings
from contextlib import nullcontext
from functools import partial
from typing import Annotated, Literal

from datasets import Features, Sequence, Value
from openai import AsyncOpenAI, RateLimitError
from openai._constants import DEFAULT_MAX_RETRIES
from pydantic import Field
from typing_extensions import TypedDict

from hyped.common.lazy import LazyInstance
from hyped.data.flow.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Sample,
)
from hyped.data.flow.refs.inputs import CheckFeatureIsSequence, InputRefs
from hyped.data.flow.refs.outputs import (
    LambdaOutputFeature,
    OutputFeature,
    OutputRefs,
)
from hyped.data.flow.refs.ref import FeatureRef


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


class OpenAIChatCompletionInputRefs(InputRefs):
    """Input features required for OpenAI Chat Completion."""

    messages: Annotated[
        FeatureRef,
        CheckFeatureIsSequence(
            value_type=Features(
                {"role": Value("string"), "content": Value("string")}
            )
        ),
    ]
    """
    Input messages for chat completion.

    Must be a sequence of dictionaries with the following entries:

     - role (string): The role of the message (e.g., 'system' or 'user').
     - content (string): The content of the message.
    """


class OpenAIChatCompletionOutputRefs(OutputRefs):
    """Outputs generated by OpenAI Chat Completion Processor."""

    run_id: Annotated[FeatureRef, OutputFeature(Value("string"))]
    """
    ID associated with the chat completion run.

    FeatureType: :code:`Value("string")`
    """

    message: Annotated[FeatureRef, OutputFeature(Value("string"))]
    """
    Generated chat message.

    FeatureType: :code:`Value("string")`
    """

    logprobs: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, _: Sequence(
                {
                    "token": Value("string"),
                    "logprob": Value("float32"),
                    "top_logprobs": Sequence(
                        {
                            "token": Value("string"),
                            "logprob": Value("float32"),
                        },
                        length=(
                            0 if (c.top_logprobs is None) else c.top_logprobs
                        ),
                    ),
                }
            )
        ),
    ]
    """
    Log probabilities associated with the generated tokens.

    FeatureType:

    .. code-block:: python
        
        Sequence(
            {
                "token": Value("string"),
                "logprob": Value("float32"),
                "top_logprobs": Sequence(
                    {
                        "token": Value("string"),
                        "logprob": Value("float32"),
                    },
                )
            }
        )
    """

    tool_calls: Annotated[
        FeatureRef,
        OutputFeature(
            Sequence(
                {
                    "type": Value("string"),
                    "function": {
                        "name": Value("string"),
                        "arguments": Value("string"),
                    },
                }
            )
        ),
    ]
    """
    Tool calls generated by the completion process.
    
    FeatureType:

    .. code-block:: python

        Sequence(
            {
                "type": Value("string"),
                "function": {
                    "name": Value("string"),
                    "arguments": Value("string"),
                },
            }
        )

    """

    usage: Annotated[
        FeatureRef,
        OutputFeature(
            Features(
                {
                    "completion_tokens": Value("int32"),
                    "prompt_tokens": Value("int32"),
                    "total_tokens": Value("int32"),
                }
            )
        ),
    ]
    """
    Usage statistics for the completion process.

    FeatureType:

    .. code-block:: python

        Features(
            {
                "completion_tokens": Value("int32"),
                "prompt_tokens": Value("int32"),
                "total_tokens": Value("int32"),
            }
        )
    """


class OpenAIChatCompletionConfig(BaseDataProcessorConfig):
    """Configuration class for the OpenAI Chat Completion Data Processor.

    This class contains various attributes to configure the behavior
    of the OpenAI Chat Completion API calls.
    """

    max_concurrent_calls: None | int = None
    """The maximum number of concurrent calls to the API.
    
    When using multiple processes, each process can have up to `max_concurrent_calls`
    calls to the API at a time.
    """

    rate_limit_max_retries: int = 10
    """Maximum number of retries in case of rate limit error. Defaults to 10."""

    rate_limit_exp_backoff: int = 2
    """Exponential backoff factor (i.e., the base of the exponent) for rate limit error handling. Defaults to 2."""

    api_key: str | None = None
    """API key for authenticating with the OpenAI service.
    
    By default, this is loaded from the `OPENAI_API_KEY` environment variable.
    """

    organization: str | None = None
    """Organization ID for the OpenAI service.
    
    By default, this is loaded from the `OPENAI_ORG_ID` environment variable.
    """

    project: str | None = None
    """Project ID for the OpenAI service.
    
    By default, this is loaded from the `OPENAI_PROJECT_ID` environment variable.
    """

    base_url: str | None = None
    """Base URL for the OpenAI API."""

    timeout: float | None = None
    """Timeout duration for API calls."""

    max_retries: int = DEFAULT_MAX_RETRIES
    """Maximum number of retries for API calls."""

    default_headers: dict[str, str] | None = None
    """Default headers to include in the API requests."""

    default_query: dict[str, str] | None = None
    """Default query parameters to include in the API requests."""

    model: str = "gpt-3.5-turbo-0125"
    """ID of the model to use for chat completion."""

    frequency_penalty: float = Field(default=0, ge=-2, le=2)
    """Frequency penalty
    
    Number between -2.0 and 2.0. Positive values penalize new tokens based on
    their existing frequency in the text so far, decreasing the model's
    likelihood to repeat the same line verbatim.
    """

    presence_penalty: float = Field(default=0, ge=-2, le=2)
    """Presence penalty
    
    Number between -2.0 and 2.0. Positive values penalize new tokens based on
    whether they appear in the text so far, increasing the model's likelihood
    to talk about new topics.
    """

    logit_bias: dict[int, Annotated[float, Field(ge=-100, le=100)]] = {}
    """Modify the likelihood of specified tokens appearing in the completion."""

    logprobs: bool = False
    """Whether to return log probabilities of the output tokens. Defaults to `False`."""

    top_logprobs: None | int = None
    """Top-k logprobs to return.
    
    An integer between 0 and 20 specifying the number of most likely tokens to
    return at each token position, each with an associated log probability.
    """

    temperature: float = Field(default=1, ge=0, le=2)
    """Temperature
    
    Sampling temperature to use, between 0 and 2. Higher values like 0.8 make the
    output more random, while lower values like 0.2 make it more focused and
    deterministic.
    """

    top_p: float = Field(default=1, ge=0, le=1)
    """Probability cutoff
    
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass.
    """

    max_tokens: None | int = None
    """The maximum number of tokens that can be generated in the chat completion."""

    tools: None | list[OpenAITool] = None
    """A list of tools the model may call. Currently, only functions are supported as tools."""

    tool_choice: None | str | OpenAITool = None
    """Choice of tool to generate a call for.
    
    Controls which (if any) tool is called by the model. Options include `none`,
    `auto`, or a specific tool.
    """

    response_format: None | dict[str, str] = None
    """Response format
    
    Specifies the format that the model must output. Setting to
    :code:`{ "type": "json_object" }` enables JSON mode, ensuring the message the
    model generates is valid JSON.
    """

    seed: None | int = None
    """Ranomd seed
    
    If specified, the OpenAI system will make a best effort to sample deterministically,
    such that repeated requests with the same seed and parameters should return the same
    result.
    """

    stop: None | str = None
    """Sequence where the API will stop generating further tokens."""

    extra_headers: None | dict[str, str] = None
    """Additional headers to include in the API requests."""

    extra_query: None | dict[str, str] = None
    """Additional query parameters to include in the API requests."""

    extra_body: None | dict[str, str] = None
    """Additional JSON properties to include in the API request body."""


class OpenAIChatCompletion(
    BaseDataProcessor[
        OpenAIChatCompletionConfig,
        OpenAIChatCompletionInputRefs,
        OpenAIChatCompletionOutputRefs,
    ]
):
    """Processor for OpenAI Chat Completion.

    This processor handles interactions with the OpenAI Chat Completion API.
    """

    def __init__(
        self, config: None | OpenAIChatCompletionConfig = None, **kwargs
    ) -> None:
        """Initialize the OpenAIChatCompletion processor.

        Args:
            config (OpenAIChatCompletionConfig, optional): Configuration for the processor.
            **kwargs: Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        super(OpenAIChatCompletion, self).__init__(config, **kwargs)
        # create semaphore object to control the maximum
        # number of concurrent calls to the api
        self.sem = (
            LazyInstance(
                partial(
                    asyncio.Semaphore, value=self.config.max_concurrent_calls
                )
            )
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
                # temporary fix: see https://github.com/encode/httpx/discussions/2959
                default_headers=(self.config.default_headers or {})
                | {"Connection": "close"},
                default_query=self.config.default_query,
            )
        )

    async def api_call(self, inputs: Sample, index: int, rank: int) -> Sample:
        """Make an API call to the OpenAI Chat Completion endpoint.

        Args:
            inputs (Sample): Input sample containing messages for chat completion.
            index (int): Index of the sample in the dataset.
            rank (int): Rank of the sample.

        Returns:
            Sample: Output sample containing the completion results.
        """
        resp = await self.client.chat.completions.create(
            messages=inputs["messages"],
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

        # parse output
        return {
            "run_id": resp.id,
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
            "usage": {
                "completion_tokens": resp.usage.completion_tokens,
                "prompt_tokens": resp.usage.prompt_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
        }

    async def process(self, inputs: Sample, index: int, rank: int) -> Sample:
        """Process the input sample using the OpenAI Chat Completion API.

        Args:
            inputs (Sample): Input sample containing messages for chat completion.
            index (int): Index of the sample in the dataset.
            rank (int): Rank of the sample.

        Returns:
            Sample: Output sample containing the completion results.

        Raises:
            Exception: If the maximum number of retries for rate limiting is exceeded.
        """
        # TODO: outsource this logic into a base api data processor
        async with self.sem:
            for i in range(0, 1 + self.config.rate_limit_max_retries):
                try:
                    return await self.api_call(inputs, index, rank)
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
