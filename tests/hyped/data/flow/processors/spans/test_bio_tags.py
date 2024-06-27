from datasets import ClassLabel, Features, Sequence, Value

from hyped.data.flow.processors.spans.bio_tags import BioTags, BioTagsConfig
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestBioTags(BaseDataProcessorTest):
    # processor
    processor_type = BioTags
    processor_config = BioTagsConfig()
    # input specification
    input_features = Features(
        {
            "spans": Sequence(Sequence(Value("int32"), length=2)),
            "labels": Sequence(Value("string")),
            "length": Value("int32"),
        }
    )
    input_data = {
        "spans": [[], [(1, 2), (3, 5)], [(3, 5), (1, 2)]],
        "labels": [[], ["X", "Y"], ["Y", "X"]],
        "length": [6, 6, 6],
    }
    input_index = [0, 1, 2]
    # expected output specification
    expected_output_feature = Features({"tags": Sequence(Value("string"))})
    expected_output_data = {
        "tags": [
            ["O", "O", "O", "O", "O", "O"],
            ["O", "B-X", "O", "B-Y", "I-Y", "O"],
            ["O", "B-X", "O", "B-Y", "I-Y", "O"],
        ]
    }


class TestBioTags_ClassLabels(BaseDataProcessorTest):
    # processor
    processor_type = BioTags
    processor_config = BioTagsConfig()
    # input specification
    input_features = Features(
        {
            "spans": Sequence(Sequence(Value("int32"), length=2)),
            "labels": Sequence(ClassLabel(names=["X", "Y"])),
            "length": Value("int32"),
        }
    )
    input_data = {
        "spans": [[], [(1, 2), (3, 5)], [(3, 5), (1, 2)]],
        "labels": [[], [0, 1], [1, 0]],
        "length": [6, 6, 6],
    }
    input_index = [0, 1, 2]
    # expected output specification
    expected_output_feature = Features({"tags": Sequence(Value("string"))})
    expected_output_data = {
        "tags": [
            [0, 0, 0, 0, 0, 0],
            [0, 1, 0, 3, 4, 0],
            [0, 1, 0, 3, 4, 0],
        ]
    }


class TestBioTags_OverlapError(BaseDataProcessorTest):
    # processor
    processor_type = BioTags
    processor_config = BioTagsConfig()
    # input specification
    input_features = Features(
        {
            "spans": Sequence(Sequence(Value("int32"), length=2)),
            "labels": Sequence(ClassLabel(names=["X", "Y"])),
            "length": Value("int32"),
        }
    )
    input_data = {
        "spans": [
            [(1, 2), (1, 5)],
        ],
        "labels": [
            [0, 1],
        ],
        "length": [6],
    }
    input_index = [0]
    # expected runtime error
    expected_execution_error = ValueError


class TestBioTags_LengthMismatch(BaseDataProcessorTest):
    # processor
    processor_type = BioTags
    processor_config = BioTagsConfig()
    # input specification
    input_features = Features(
        {
            "spans": Sequence(Sequence(Value("int32"), length=2), length=2),
            "labels": Sequence(ClassLabel(names=["X", "Y"]), length=1),
            "length": Value("int32"),
        }
    )
    input_data = {
        "spans": [
            [(1, 2), (1, 5)],
        ],
        "labels": [
            [0],
        ],
        "length": [6],
    }
    input_index = [0]
    # expected runtime error
    expected_input_verification_error = RuntimeError
