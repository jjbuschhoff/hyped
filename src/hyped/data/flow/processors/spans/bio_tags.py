"""Module for BIO (Begin-In-Out) tagging processors.

This module defines the data processor to generate BIO tags from span annotations. BIO Tagging
is a common tagging scheme used in Natural Language Processing for tasks like Named Entity
Recognition.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from datasets import ClassLabel, Sequence, Value
from typing_extensions import Annotated

from hyped.common.feature_checks import (
    INT_TYPES,
    UINT_TYPES,
    check_feature_equals,
    check_feature_is_sequence,
    get_sequence_feature,
    get_sequence_length,
)
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    IOContext,
    Sample,
)
from hyped.data.flow.core.refs.inputs import (
    CheckFeatureEquals,
    CheckFeatureIsSequence,
    FeatureValidator,
    InputRefs,
)
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef

from .utils import validate_spans_feature


class BioTagsInputRefs(InputRefs):
    """Input references for the BioTags processor."""

    spans: Annotated[FeatureRef, FeatureValidator(validate_spans_feature)]
    """The feature reference to the span annotations. Must be a sequence of spans."""

    labels: Annotated[
        FeatureRef, CheckFeatureIsSequence([Value("string"), ClassLabel])
    ]
    """The feature reference for the label annotations, which should be a sequence
    of :code:`strings` or :class:`ClassLabels`.
    """

    length: Annotated[FeatureRef, CheckFeatureEquals(INT_TYPES + UINT_TYPES)]
    """The feature reference to the length, which should be of integer type
    indicating the target length of the tags sequence.
    """

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization checks.

        Ensures that the lengths of spans and labels match if
        they are both specified and not dynamic.

        Raises:
            RuntimeError: If the lengths of spans and labels do not match.
        """
        if (
            (get_sequence_length(self.spans.feature_) != -1)
            and (get_sequence_length(self.labels.feature_) != -1)
        ) and (
            get_sequence_length(self.spans.feature_)
            != get_sequence_length(self.labels.feature_)
        ):
            raise RuntimeError(
                f"Mismatch in sequence lengths: 'spans' and 'labels' "
                f"must have the same length, got {get_sequence_length(self.spans.feature_)} "
                f"!= {get_sequence_length(self.labels.feature_)}."
            )


def build_bio_tags_feature(
    config: BioTagsConfig, inputs: BioTagsInputRefs
) -> Sequence:
    """Builds the BIO tags feature from the input spans and labels.

    Args:
        config (BioTagsConfig): The configuration for the BioTags processor.
        inputs (BioTagsInputRefs): The input references containing spans and labels.

    Returns:
        Sequence: The sequence feature representing the BIO tags.
    """
    # TODO: we can infer the length of the tags sequence
    #       from the length input feature in case the length
    #       feature comes from a constant

    # read labels feature
    labels_feature = get_sequence_feature(inputs.labels.feature_)

    # keep string feature if input labels are also strings
    if check_feature_equals(labels_feature, Value("string")):
        return Sequence(Value("string"))

    assert check_feature_equals(labels_feature, ClassLabel)
    # build class labels from source class labels
    bio_tags = ClassLabel(
        names=[config.out_tag]
        + [
            "%s%s" % (prefix, label)
            for label in labels_feature.names
            for prefix in [config.begin_tag_prefix, config.in_tag_prefix]
        ]
    )

    return Sequence(bio_tags)


class BioTagsOutputRefs(OutputRefs):
    """Output references for the BioTags processor."""

    tags: Annotated[FeatureRef, LambdaOutputFeature(build_bio_tags_feature)]
    """The feature reference for the BIO tags."""


class BioTagsConfig(BaseDataProcessorConfig):
    """Configuration for the BioTags processor."""

    begin_tag_prefix: str = "B-"
    """The prefix for the begin tag. Defaults to "B-"."""
    in_tag_prefix: str = "I-"
    """The prefix for the in tag. Defaults to "I-"."""
    out_tag: str = "O"
    """The tag for outside spans. Defaults to "O"."""


class BioTags(
    BaseDataProcessor[BioTagsConfig, BioTagsInputRefs, BioTagsOutputRefs]
):
    """Processor for generating BIO tags from spans and labels.

    This processor takes spans and labels as input and generates BIO tags,
    """

    def process(
        self, inputs: Sample, index: int, rank: int, io: IOContext
    ) -> Sample:
        """Processes the inputs to generate BIO tags.

        Args:
            inputs (Sample): The input sample containing spans and labels.
            index (int): The index of the current sample.
            rank (int): The rank of the current sample.
            io (IOContext): The IO context for handling input and output features.

        Returns:
            Sample: The output sample containing BIO tags.

        Raises:
            ValueError: If there is an overlap between entities.
        """
        spans = inputs["spans"]
        labels = inputs["labels"]
        length = inputs["length"]

        # convert labels to strings
        if check_feature_is_sequence(io.inputs["labels"], ClassLabel):
            labels = io.inputs["labels"].feature.int2str(labels)

        # build initial tag sequence of all out and invalid tags
        tags = np.full(length, fill_value=self.config.out_tag, dtype=object)

        # insert all entity spans
        for label, (b, e) in zip(labels, spans):
            # check for overlaps with previous annotations
            if (tags[b:e] != self.config.out_tag).any():
                # get the overlapping entity types
                overlap_types = [label] + [
                    (
                        tag.removeprefix(
                            self.config.begin_tag_prefix
                        ).removeprefix(self.config.in_tag_prefix)
                    )
                    for tag in tags[b:e]
                    if tag != self.config.out_tag
                ]
                # raise error on overlap
                raise ValueError(
                    "Detected overlap between entities of types %s"
                    % ", ".join(overlap_types)
                )

            # add entity to tag sequence
            tags[b:e] = "%s%s" % (self.config.in_tag_prefix, label)
            tags[b] = "%s%s" % (self.config.begin_tag_prefix, label)

        # convert tags to list
        tags = tags.tolist()
        # convert strings to class label ids
        if check_feature_is_sequence(io.outputs["tags"], ClassLabel):
            tags = io.outputs["tags"].feature.str2int(tags)

        return Sample(tags=tags)
