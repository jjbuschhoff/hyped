import pytest
from datasets import ClassLabel, Features, Sequence, Value

from hyped.common.feature_key import FeatureKey
from hyped.data.processors.base import BaseDataProcessor
from hyped.data.processors.templates.jinja2 import Jinja2, Jinja2Config
from tests.hyped.data.processors.base import BaseTestDataProcessor


class TestJinja2(BaseTestDataProcessor):
    @pytest.fixture(
        params=[
            # simple access to values
            (
                "{{ FeatureKey('A') | index_example }}",
                "valA",
                {FeatureKey("A")},
            ),
            (
                "{{ FeatureKey('B', 0) | index_example }}",
                "valB.0",
                {FeatureKey("B", 0)},
            ),
            (
                "{{ FeatureKey('B', 1) | index_example }}",
                "valB.1",
                {FeatureKey("B", 1)},
            ),
            (
                "{{ FeatureKey('B', 2) | index_example }}",
                "valB.2",
                {FeatureKey("B", 2)},
            ),
            (
                "{{ FeatureKey('B') | index_example }}",
                str(["valB.0", "valB.1", "valB.2"]),
                {FeatureKey("B")},
            ),
            ("{{ FeatureKey('C') | index_example }}", "0", {FeatureKey("C")}),
            (
                "{{ FeatureKey('D', 'X') | index_example }}",
                "valX",
                {FeatureKey("D", "X")},
            ),
            (
                "{{ FeatureKey('D', 'Y') | index_example }}",
                "valY",
                {FeatureKey("D", "Y")},
            ),
            # simple access to features
            (
                "{{ FeatureKey('A') | index_features }}",
                str(Value("string")),
                {FeatureKey("A")},
            ),
            (
                "{{ FeatureKey('B', 0) | index_features }}",
                str(Value("string")),
                {FeatureKey("B", 0)},
            ),
            (
                "{{ FeatureKey('B', 1) | index_features }}",
                str(Value("string")),
                {FeatureKey("B", 1)},
            ),
            (
                "{{ FeatureKey('B', 2) | index_features }}",
                str(Value("string")),
                {FeatureKey("B", 2)},
            ),
            (
                "{{ FeatureKey('B') | index_features }}",
                str(Sequence(Value("string"), length=3)),
                {FeatureKey("B")},
            ),
            (
                "{{ FeatureKey('C') | index_features }}",
                str(ClassLabel(names=["label"])),
                {FeatureKey("C")},
            ),
            (
                "{{ FeatureKey('D', 'X') | index_features }}",
                str(Value("string")),
                {FeatureKey("D", "X")},
            ),
            (
                "{{ FeatureKey('D', 'Y') | index_features }}",
                str(Value("string")),
                {FeatureKey("D", "Y")},
            ),
            # get class label from label id
            (
                """{{
                    (FeatureKey('C') | index_features).names[
                        FeatureKey('C') | index_example
                    ]
                }}""",
                "label",
                {FeatureKey("C")},
            ),
            # loop over sequence
            (
                "{% for i in range((FeatureKey('B') | index_features).length) %}"
                "{{ FeatureKey('B', i) | index_example }} "
                "{% endfor %}",
                "valB.0 valB.1 valB.2 ",
                {
                    FeatureKey("B"),
                    FeatureKey("B", 0),
                    FeatureKey("B", 1),
                    FeatureKey("B", 2),
                },
            ),
        ]
    )
    def template_target_keys(self, request):
        return request.param

    @pytest.fixture
    def template(self, template_target_keys):
        return template_target_keys[0]

    @pytest.fixture
    def target(self, template_target_keys):
        return template_target_keys[1]

    @pytest.fixture
    def feature_keys(self, template_target_keys):
        return template_target_keys[2]

    @pytest.fixture
    def in_features(self):
        return Features(
            {
                "A": Value("string"),
                "B": Sequence(Value("string"), length=3),
                "C": ClassLabel(names=["label"]),
                "D": {"X": Value("string"), "Y": Value("string")},
            }
        )

    @pytest.fixture
    def in_batch(self):
        return {
            "A": ["valA"],
            "B": [["valB.0", "valB.1", "valB.2"]],
            "C": [0],  # refers to class label 'label'
            "D": [{"X": "valX", "Y": "valY"}],
        }

    @pytest.fixture
    def processor(self, template):
        return Jinja2(Jinja2Config(template=template, output="out"))

    @pytest.fixture
    def expected_out_features(self):
        return Features({"out": Value("string")})

    @pytest.fixture
    def expected_out_batch(self, target):
        return {"out": [target]}

    def test_required_feature_keys(self, in_features, processor, feature_keys):
        # reset data processor preparation state
        BaseDataProcessor.__init__(processor, config=processor.config)
        # cannot access required feature keys before preparation
        with pytest.raises(RuntimeError):
            processor.required_feature_keys
        # prepare the data processor
        processor.prepare(in_features)
        assert processor.required_feature_keys == feature_keys

    def test_warning_on_parse_error(self, in_features):
        # invalid feature key, 'INVALID' not part of features
        template = "{{ FeatureKey('INVALID') | index_example }}"
        processor = Jinja2(Jinja2Config(template=template, output="out"))
        # should warn about error in template
        with pytest.warns(RuntimeWarning, match="Encountered an expcetion"):
            processor._collect_required_feature_keys(in_features)
