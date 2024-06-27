"""Module for resolving overlapping spans in data processing workflows.

This module defines a processor and associated classes for resolving overlapping spans
in sequence data. The processor uses various strategies to identify and manage overlaps,
ensuring that the output spans are disjoint. This is essential for tasks that require
unique span annotations, such as named entity recognition or chunking.
"""

from itertools import compress

from datasets import Sequence, Value
from typing_extensions import Annotated

from hyped.common.feature_checks import (
    get_sequence_feature,
    get_sequence_length,
)
from hyped.data.flow.core.nodes.processor import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    IOContext,
    Sample,
)
from hyped.data.flow.core.refs.inputs import FeatureValidator, InputRefs
from hyped.data.flow.core.refs.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.flow.core.refs.ref import FeatureRef

from .utils import (
    ResolveOverlapsStrategy,
    resolve_overlaps,
    validate_spans_feature,
)


class ResolveOverlapsInputRefs(InputRefs):
    """Input references for the ResolveOverlaps processor."""

    spans: Annotated[FeatureRef, FeatureValidator(validate_spans_feature)]
    """Reference to the spans feature, validated to ensure it is a sequence of spans."""


class ResolveOverlapsOutputRefs(OutputRefs):
    """Output references for the ResolveOverlaps processor."""

    spans: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, i: Sequence(
                get_sequence_feature(i.spans.feature_),
            )
        ),
    ]
    """Reference to the output spans feature guaranteed to have no span overlaps.
    The output spans maintain the same sequence type as the input spans.
    """

    mask: Annotated[
        FeatureRef,
        LambdaOutputFeature(
            lambda _, i: Sequence(
                Value("bool"), length=get_sequence_length(i.spans.feature_)
            )
        ),
    ]
    """Reference to the output mask feature, indicating which spans were retained
    after resolving overlaps.
    """


class ResolveOverlapsConfig(BaseDataProcessorConfig):
    """Configuration for the ResolveOverlaps processor."""

    strategy: ResolveOverlapsStrategy = ResolveOverlapsStrategy.APPROX
    """The strategy to use for resolving overlaps."""


class ResolveOverlaps(
    BaseDataProcessor[
        ResolveOverlapsConfig,
        ResolveOverlapsInputRefs,
        ResolveOverlapsOutputRefs,
    ]
):
    """Processor for resolving overlapping spans.

    This processor uses a specified strategy to resolve overlapping spans and produces
    a mask indicating which spans were retained.
    """

    def process(
        self, inputs: Sample, index: int, rank: int, io: IOContext
    ) -> Sample:
        """Process the input sample to resolve overlapping spans.

        Args:
            inputs (Sample): The input sample containing spans.
            index (int): The index of the current sample.
            rank (int): The rank of the current process.
            io (IOContext): The IO context for managing input and output features.

        Returns:
            Sample: The output sample with resolved spans and the mask.
        """
        if len(inputs["spans"]) == 0:
            # handle trivial case separately
            return Sample(spans=[], mask=[])

        # resolve overlaps in spans
        mask = resolve_overlaps(inputs["spans"], strategy=self.config.strategy)
        spans = list(compress(inputs["spans"], mask))

        # return output features
        return Sample(spans=spans, mask=mask)
