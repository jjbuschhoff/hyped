import json
import pickle

from datasets import Features, Value

from hyped.data.flow.processors.templates.jinja2 import Jinja2, Jinja2Config
from tests.hyped.data.flow.processors.base import BaseDataProcessorTest


class TestJinja2(BaseDataProcessorTest):
    # processor type
    processor_type = Jinja2
    # processor config
    processor_config = Jinja2Config(
        template="""A is {{ inputs.A }}, B is {{ inputs.B }}"""
    )
    # input
    input_features = Features(
        {
            "features": {
                "A": Value("int64"),
                "B": Value("string"),
            }
        }
    )
    input_data = {
        "features": [
            {"A": 0, "B": "a"},
            {"A": 1, "B": "b"},
            {"A": 2, "B": "c"},
        ]
    }
    input_index = [0, 1, 2]
    # expected output
    expected_output_data = {
        "parsed": [
            "A is 0, B is a",
            "A is 1, B is b",
            "A is 2, B is c",
        ]
    }

    def test_pickle_processor(self, processor: Jinja2):
        # pickle and unpickle processor
        reconstructed = pickle.loads(pickle.dumps(processor))
        # make sure the underlying template is the template
        assert processor.template.render(
            inputs=type(self).input_data["features"][0]
        ) == reconstructed.template.render(
            inputs=type(self).input_data["features"][0]
        )
