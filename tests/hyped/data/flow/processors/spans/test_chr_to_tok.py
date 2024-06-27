from datasets import Features, Sequence, Value

from hyped.data.flow.processors.spans.chr_to_tok import (
    ChrToTokSpans,
    ChrToTokSpansConfig,
)
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestChrToTokSpans(BaseDataProcessorTest):
    # processor
    processor_type = ChrToTokSpans
    processor_config = ChrToTokSpansConfig()
    # input specification
    input_features = Features(
        {
            "chr_spans": Sequence(Sequence(Value("int32"), length=2)),
            "query_spans": Sequence(Sequence(Value("int32"), length=2)),
        }
    )
    input_data = {
        "chr_spans": [
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
            [(0, 5), (6, 9), (10, 13), (14, 18), (18, 25)],
        ],
        "query_spans": [
            [],
            [(0, 9)],
            [(0, 9), (14, 18)],
            [(6, 9), (14, 18)],
        ],
    }
    input_index = [0, 1, 2, 3]
    # expected output specification
    expected_output_feature = Features(
        {"tok_spans": Sequence(Sequence(Value("int32"), length=2))}
    )
    expected_output_data = {
        "tok_spans": [
            [],
            [(0, 2)],
            [(0, 2), (3, 4)],
            [(1, 2), (3, 4)],
        ]
    }
