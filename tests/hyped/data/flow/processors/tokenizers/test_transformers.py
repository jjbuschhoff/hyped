from datasets import Features, Sequence, Value

from hyped.data.flow.processors.tokenizers.transformers import (
    TransformersTokenizer,
    TransformersTokenizerConfig,
)
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestTransformersTokenizer(BaseDataProcessorTest):
    # processor
    processor_type = TransformersTokenizer
    processor_config = TransformersTokenizerConfig(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased"
    )
    # inputs
    input_features = Features({"text": Value("string")})
    input_data = {
        "text": [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Vivamus lacinia odio vitae vestibulum vestibulum",
        ]
    }
    input_index = [0, 1]
    # expected outputs
    expected_output_features = Features(
        {"input_ids": Sequence(Value("int32"))}
    )
    expected_output_data = {
        "input_ids": [
            [
                101,
                19544,
                2213,
                12997,
                17421,
                2079,
                10626,
                4133,
                2572,
                3388,
                1010,
                9530,
                3366,
                6593,
                3388,
                3126,
                27133,
                18136,
                6129,
                12005,
                2102,
                1012,
                102,
            ],
            [
                101,
                20022,
                7606,
                18749,
                23309,
                21045,
                2080,
                19300,
                2063,
                17447,
                12322,
                25100,
                17447,
                12322,
                25100,
                102,
            ],
        ]
    }


class TestTransformersTokenizer_WithPair(BaseDataProcessorTest):
    # processor
    processor_type = TransformersTokenizer
    processor_config = TransformersTokenizerConfig(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased"
    )
    # inputs
    input_features = Features({"text": Value("string")})
    input_data = {
        "text": [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Vivamus lacinia odio vitae vestibulum vestibulum",
        ],
        "text_pair": [
            "Cras ultricies ligula sed magna dictum porta.",
            "Curabitur aliquet quam id dui posuere blandit.",
        ],
    }
    input_index = [0, 1]
    # expected output
    expected_output_features = Features(
        {"input_ids": Sequence(Value("int32"))}
    )
    expected_output_data = {
        "input_ids": [
            [
                101,
                19544,
                2213,
                12997,
                17421,
                2079,
                10626,
                4133,
                2572,
                3388,
                1010,
                9530,
                3366,
                6593,
                3388,
                3126,
                27133,
                18136,
                6129,
                12005,
                2102,
                1012,
                102,
                13675,
                3022,
                17359,
                12412,
                3111,
                5622,
                24848,
                2050,
                7367,
                2094,
                20201,
                4487,
                27272,
                3417,
                2050,
                1012,
                102,
            ],
            [
                101,
                20022,
                7606,
                18749,
                23309,
                21045,
                2080,
                19300,
                2063,
                17447,
                12322,
                25100,
                17447,
                12322,
                25100,
                102,
                12731,
                2527,
                16313,
                3126,
                4862,
                12647,
                24209,
                3286,
                8909,
                4241,
                2072,
                13433,
                6342,
                7869,
                20857,
                4183,
                1012,
                102,
            ],
        ]
    }


class TestTransformersTokenizer_WithPairAndTarget(BaseDataProcessorTest):
    # processor
    processor_type = TransformersTokenizer
    processor_config = TransformersTokenizerConfig(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased"
    )
    # inputs
    input_features = Features({"text": Value("string")})
    input_data = {
        "text": [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Vivamus lacinia odio vitae vestibulum vestibulum",
        ],
        "text_pair": [
            "Cras ultricies ligula sed magna dictum porta.",
            "Curabitur aliquet quam id dui posuere blandit.",
        ],
        "text_target": [
            "Pellentesque in ipsum id orci porta dapibus.",
            "Donec sollicitudin molestie malesuada.",
        ],
        "text_pair_target": [
            "Vivamus suscipit tortor eget felis porttitor volutpat.",
            "Sed porttitor lectus nibh.",
        ],
    }
    input_index = [0, 1]
    # expected output
    expected_output_features = Features(
        {"input_ids": Sequence(Value("int32"))}
    )
    expected_output_data = {
        "input_ids": [
            [
                101,
                19544,
                2213,
                12997,
                17421,
                2079,
                10626,
                4133,
                2572,
                3388,
                1010,
                9530,
                3366,
                6593,
                3388,
                3126,
                27133,
                18136,
                6129,
                12005,
                2102,
                1012,
                102,
                13675,
                3022,
                17359,
                12412,
                3111,
                5622,
                24848,
                2050,
                7367,
                2094,
                20201,
                4487,
                27272,
                3417,
                2050,
                1012,
                102,
            ],
            [
                101,
                20022,
                7606,
                18749,
                23309,
                21045,
                2080,
                19300,
                2063,
                17447,
                12322,
                25100,
                17447,
                12322,
                25100,
                102,
                12731,
                2527,
                16313,
                3126,
                4862,
                12647,
                24209,
                3286,
                8909,
                4241,
                2072,
                13433,
                6342,
                7869,
                20857,
                4183,
                1012,
                102,
            ],
        ]
    }


class TestTransformersTokenizerWithPaddingAndTruncation(BaseDataProcessorTest):
    # processor
    processor_type = TransformersTokenizer
    processor_config = TransformersTokenizerConfig(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased",
        padding="max_length",
        truncation=True,
        max_length=4,
    )
    # inputs
    input_features = Features({"text": Value("string")})
    input_data = {
        "text": [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Vivamus lacinia odio vitae vestibulum vestibulum",
        ]
    }
    input_index = [0, 1]
    # expected outputs
    expected_output_features = Features(
        {"input_ids": Sequence(Value("int32"), length=4)}
    )
    expected_output_data = {
        "input_ids": [[101, 19544, 2213, 102], [101, 20022, 7606, 102]]
    }


class TestTransformersTokenizerReturnAll(BaseDataProcessorTest):
    # processor
    processor_type = TransformersTokenizer
    processor_config = TransformersTokenizerConfig(
        tokenizer="./tests/artifacts/tokenizers/bert-base-uncased",
        return_tokens=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        return_special_tokens_mask=True,
        return_offsets_mapping=True,
        return_length=True,
        return_word_ids=True,
    )
    # inputs
    input_features = Features({"text": Value("string")})
    input_data = {
        "text": [
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Vivamus lacinia odio vitae vestibulum vestibulum",
        ]
    }
    input_index = [0, 1]
    # expected outputs
    expected_output_features = Features(
        {
            "tokens": Sequence(Value("string")),
            "input_ids": Sequence(Value("int32")),
            "token_type_ids": Sequence(Value("int32")),
            "attention_mask": Sequence(Value("int32")),
            "special_tokens_mask": Sequence(Value("int32")),
            "word_ids": Sequence(Value("int32")),
            "offset_mapping": Sequence(Sequence(Value("int32"), length=2)),
            "length": Value("int32"),
        }
    )
    expected_output_data = {
        "input_ids": [
            [
                101,
                19544,
                2213,
                12997,
                17421,
                2079,
                10626,
                4133,
                2572,
                3388,
                1010,
                9530,
                3366,
                6593,
                3388,
                3126,
                27133,
                18136,
                6129,
                12005,
                2102,
                1012,
                102,
            ],
            [
                101,
                20022,
                7606,
                18749,
                23309,
                21045,
                2080,
                19300,
                2063,
                17447,
                12322,
                25100,
                17447,
                12322,
                25100,
                102,
            ],
        ],
        "token_type_ids": [
            [
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        "attention_mask": [
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        "special_tokens_mask": [
            [
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
            ],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        "offset_mapping": [
            [
                [0, 0],
                [0, 4],
                [4, 5],
                [6, 8],
                [8, 11],
                [12, 14],
                [14, 17],
                [18, 21],
                [22, 24],
                [24, 26],
                [26, 27],
                [28, 31],
                [31, 33],
                [33, 35],
                [35, 37],
                [37, 39],
                [40, 43],
                [43, 46],
                [46, 50],
                [51, 54],
                [54, 55],
                [55, 56],
                [0, 0],
            ],
            [
                [0, 0],
                [0, 4],
                [4, 7],
                [8, 11],
                [11, 15],
                [16, 19],
                [19, 20],
                [21, 25],
                [25, 26],
                [27, 31],
                [31, 33],
                [33, 37],
                [38, 42],
                [42, 44],
                [44, 48],
                [0, 0],
            ],
        ],
        "length": [23, 16],
        "tokens": [
            [
                "[CLS]",
                "lore",
                "##m",
                "ip",
                "##sum",
                "do",
                "##lor",
                "sit",
                "am",
                "##et",
                ",",
                "con",
                "##se",
                "##ct",
                "##et",
                "##ur",
                "adi",
                "##pis",
                "##cing",
                "eli",
                "##t",
                ".",
                "[SEP]",
            ],
            [
                "[CLS]",
                "viva",
                "##mus",
                "lac",
                "##inia",
                "odi",
                "##o",
                "vita",
                "##e",
                "vest",
                "##ib",
                "##ulum",
                "vest",
                "##ib",
                "##ulum",
                "[SEP]",
            ],
        ],
        "word_ids": [
            [
                -1,
                0,
                0,
                1,
                1,
                2,
                2,
                3,
                4,
                4,
                5,
                6,
                6,
                6,
                6,
                6,
                7,
                7,
                7,
                8,
                8,
                9,
                -1,
            ],
            [-1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5, 5, 5, -1],
        ],
    }
