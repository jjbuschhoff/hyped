"""This module implements the HuggingFace Transformers Tokenizer Processor.

It contains the implementation of a data processor for tokenizing text using
Transformers. The processor utilizes the Transformers library to tokenize
input text and produce various output features, such as input IDs, token types,
attention masks, special tokens masks, offset mappings, and word IDs. It offers
flexible configuration options to customize tokenization behavior and control
the output features generated during tokenization.
"""
from __future__ import annotations

from typing import Literal

from datasets import Sequence, Value
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import TruncationStrategy
from transformers.utils import PaddingStrategy
from typing_extensions import Annotated

from hyped.data.flow.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
)
from hyped.data.flow.refs.inputs import CheckFeatureEquals, InputRefs
from hyped.data.flow.refs.outputs import (
    ConditionalOutputFeature,
    LambdaOutputFeature,
    OutputRefs,
)
from hyped.data.flow.refs.ref import NONE_REF, FeatureRef


def _get_output_sequence_length(config: TransformersTokenizerConfig) -> int:
    """Determine the sequence length based on tokenizer configuration."""
    # check for constant length
    is_constant = (
        (config.max_length is not None)
        and (config.padding == "max_length")
        and (
            config.truncation
            in (True, "longest_first", "only_first", "only_second")
        )
    )
    # get sequence length in case it's constant
    return config.max_length if is_constant else -1


class TransformersTokenizerInputRefs(InputRefs):
    """Inputs to the Transformers Tokenizer processor."""

    # required input features
    text: Annotated[FeatureRef, CheckFeatureEquals(Value("string"))]
    """Input feature representing the input text."""

    # optional input features
    text_pair: Annotated[
        FeatureRef, CheckFeatureEquals(Value("string"))
    ] = NONE_REF
    """Optional input feature representing the paired text."""

    text_target: Annotated[
        FeatureRef, CheckFeatureEquals(Value("string"))
    ] = NONE_REF
    """Optional input feature representing the target text."""

    text_pair_target: Annotated[
        FeatureRef, CheckFeatureEquals(Value("string"))
    ] = NONE_REF
    """Optional input feature representing the paired target text."""


class TransformersTokenizerOutputRefs(OutputRefs):
    """Outputs produced by the Transformers Tokenizer processor."""

    input_ids: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, _: Sequence(
                Value("int32"), length=_get_output_sequence_length(c)
            )
        ),
    ]
    """Output feature representing the input IDs."""

    tokens: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, _: Sequence(
                Value("string"), length=_get_output_sequence_length(c)
            )
            if c.return_tokens
            else None
        ),
    ]
    """Optional output feature representing the tokens.

    Present if the configuration setting :code:`return_tokens` is :class:`True`.
    """

    token_type_ids: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, _: Sequence(
                Value("int32"), length=_get_output_sequence_length(c)
            )
            if c.return_token_type_ids
            else None
        ),
    ]
    """Optional output feature representing the token type IDs. 

    Present if the configuration setting :class:`return_token_type_ids` is :class:`True`.
    """

    attention_mask: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, _: Sequence(
                Value("int32"), length=_get_output_sequence_length(c)
            )
            if c.return_attention_mask
            else None
        ),
    ]
    """Optional output feature representing the attention mask.

    Present if the configuration setting :code:`return_attention_mask` is :code:`True`.
    """

    special_tokens_mask: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, _: Sequence(
                Value("int32"), length=_get_output_sequence_length(c)
            )
            if c.return_special_tokens_mask
            else None
        ),
    ]
    """Optional output feature representing the special tokens mask.

    Present if the configuration setting :code:`return_special_tokens_mask` is :code:`True`.
    """

    word_ids: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, _: Sequence(
                Value("int32"), length=_get_output_sequence_length(c)
            )
            if c.return_word_ids
            else None
        ),
    ]
    """Optional output feature representing the word IDs.

    Present if the configuration setting :code:`return_word_ids` is :code:`True`.
    """

    offset_mapping: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda c, _: Sequence(
                Sequence(Value("int32"), length=2),
                length=_get_output_sequence_length(c),
            )
            if c.return_offsets_mapping
            else None
        ),
    ]
    """Optional output feature representing the offset mappings.

    Present if the configuration setting :code:`return_offsets_mapping` is :code:`True`.
    """

    length: Annotated[
        FeatureRef,
        ConditionalOutputFeature(Value("int32"), lambda c, _: c.return_length),
    ]
    """Conditional output feature representing the length.

    Present if the configuration setting :code:`return_length` is :code:`True`.
    """


class TransformersTokenizerConfig(BaseDataProcessorConfig):
    """Configuration for the Transformers Tokenizer processor."""

    tokenizer: str
    """The name or path of the pre-trained tokenizer."""

    add_special_tokens: bool = True
    """Whether to add special tokens during tokenization."""

    padding: bool | str | PaddingStrategy = False
    """Padding strategy for sequences."""

    truncation: bool | str | TruncationStrategy = False
    """Truncation strategy for sequences."""

    max_length: None | int = None
    """Maximum sequence length after tokenization."""

    stride: int = 0
    """Stride for tokenization."""

    is_split_into_words: Literal[False] = False  # TODO
    """Flag indicating whether inputs are already split into words."""

    pad_to_multiple_of: None | int = None
    """Pad tokenized sequences to a multiple of this value."""

    return_tokens: bool = False
    """Whether to include tokenized tokens in the output."""

    return_token_type_ids: bool = False
    """Whether to include token type IDs in the output."""

    return_attention_mask: bool = False
    """Whether to include attention masks in the output."""

    return_special_tokens_mask: bool = False
    """Whether to include special tokens masks in the output."""

    return_offsets_mapping: bool = False
    """Whether to include offsets mappings in the output."""

    return_length: bool = False
    """Whether to include the length of sequences in the output."""

    return_word_ids: bool = False
    """Whether to include word IDs in the output."""


class TransformersTokenizer(
    BaseDataProcessor[
        TransformersTokenizerConfig,
        TransformersTokenizerInputRefs,
        TransformersTokenizerOutputRefs,
    ]
):
    """Transformers Tokenizer data processor.

    This processor tokenizes input text using a specified tokenizer.
    """

    def __init__(
        self, config: None | TransformersTokenizerConfig = None, **kwargs
    ) -> None:
        """Initialize the Transformers Tokenizer processor.

        Args:
            config (TransformersTokenizerConfig): Processor configuration.
            **kwargs: Additional keyword arguments that update the provided configuration
                or create a new configuration if none is provided.
        """
        super(TransformersTokenizer, self).__init__(config, **kwargs)
        # load the tokenizer instance
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer, use_fast=True, add_prefix_space=True
        )

    async def batch_process(
        self, inputs: Batch, index: list[int], rank: 0
    ) -> Batch:
        """Tokenize input batch.

        Args:
            inputs (Batch): Batch of input data.
            index (list[int]): Batch index.
            rank (int): Rank of the processor.

        Returns:
            Batch: The batch of tokenizer outputs.
        """
        # apply tokenizer
        enc = self.tokenizer(
            text=inputs["text"],
            text_pair=inputs.get("text_pair", None),
            text_target=inputs.get("text_target", None),
            text_pair_target=inputs.get("text_pair_target", None),
            add_special_tokens=self.config.add_special_tokens,
            padding=self.config.padding,
            truncation=self.config.truncation,
            max_length=self.config.max_length,
            stride=self.config.stride,
            is_split_into_words=self.config.is_split_into_words,
            pad_to_multiple_of=self.config.pad_to_multiple_of,
            return_token_type_ids=self.config.return_token_type_ids,
            return_attention_mask=self.config.return_attention_mask,
            return_special_tokens_mask=self.config.return_special_tokens_mask,
            return_offsets_mapping=self.config.return_offsets_mapping,
            return_length=self.config.return_length,
        )

        # create output batch
        out = Batch(input_ids=enc["input_ids"])

        # add all features
        if self.config.return_tokens:
            out["tokens"] = list(
                map(self.tokenizer.convert_ids_to_tokens, enc.input_ids)
            )

        if self.config.return_token_type_ids:
            out["token_type_ids"] = enc.token_type_ids

        if self.config.return_attention_mask:
            out["attention_mask"] = enc.attention_mask

        if self.config.return_special_tokens_mask:
            out["special_tokens_mask"] = enc.special_tokens_mask

        if self.config.return_word_ids:
            out["word_ids"] = [
                [(i if i is not None else -1) for i in enc.word_ids(j)]
                for j in range(len(index))
            ]

        if self.config.return_offsets_mapping:
            out["offset_mapping"] = [
                list(map(list, item)) for item in enc["offset_mapping"]
            ]

        if self.config.return_length:
            out["length"] = enc["length"]

        # return output and index
        return out
