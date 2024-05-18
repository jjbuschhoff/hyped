import pytest
from datasets import Features, Sequence, Value

from hyped.data.ref import FeatureRef
from hyped.common.feature_key import FeatureKey


class TestFeatureRef(object):

    @pytest.mark.parametrize(
        "ref", [
            FeatureRef(
                key_=FeatureKey(),
                feature_=Value("string"),
                node_id_=-1,
                flow_=None
            ),
            FeatureRef(
                key_=FeatureKey(),
                feature_=Value("int32"),
                node_id_=-1,
                flow_=None
            ),
            FeatureRef(
                key_=FeatureKey(),
                feature_=Sequence(Value("int32")),
                node_id_=-1,
                flow_=None
            ),
            FeatureRef(
                key_=FeatureKey(),
                feature_=Sequence(Value("int32"), length=3),
                node_id_=-1,
                flow_=None
            ),
            FeatureRef(
                key_=FeatureKey(),
                feature_=Features({"x": Value("int32")}),
                node_id_=-1,
                flow_=None
            ),
            FeatureRef(
                key_=FeatureKey(),
                feature_=Features(
                    {
                        "x": Value("int32"),
                        "y": Sequence(Value("string")),
                        "z": Sequence(Value("string"), length=3),
                        "a": {
                            "a.x": Value("int32"),
                            "a.y": Sequence(Value("string")),
                            "a.z": Sequence(Value("string"), length=3),
                        }
                    }
                ),
                node_id_=-1,
                flow_=None
            ),
        ]
    )
    def test_serialize(self, ref):
        # serialize and deserialize feature reference
        rec_ref = FeatureRef.model_validate(ref.model_dump())
        rec_ref_json = FeatureRef.model_validate_json(ref.model_dump_json())
        # check reconstructed reference
        assert rec_ref == ref
        assert rec_ref_json == ref

    def test_basics(self):
        
        ref = FeatureRef(
            key_=FeatureKey(),
            feature_=Features(
                {
                    "x": Value("int32"),
                    "y": Sequence(Value("string")),
                    "z": Sequence(Value("string"), length=3),
                    "a": {
                        "a_x": Value("int32"),
                    }
                }
            ),
            node_id_=-1,
            flow_=None
        )

        # invalid sub-feature key
        with pytest.raises(KeyError):
            ref.invalid
        with pytest.raises(KeyError):
            ref["invalid"]

        # index out of range error
        with pytest.raises(IndexError):
            ref.z[4]

        # basic sub-feature access checks
        assert ref["x"] == ref.x == FeatureRef(
            key_=FeatureKey("x"), feature_=Value("int32"), node_id_=-1, flow_=None
        )
        assert ref["y"] == ref.y == FeatureRef(
            key_=FeatureKey("y"), feature_=Sequence(Value("string")), node_id_=-1, flow_=None
        )
        assert ref.y[:] == FeatureRef(
            key_=FeatureKey("y", slice(None)), feature_=Sequence(Value("string")), node_id_=-1, flow_=None
        )
        assert ref.y[0] == FeatureRef(
            key_=FeatureKey("y", 0), feature_=Value("string"), node_id_=-1, flow_=None
        )
        assert ref.z[0] == FeatureRef(
            key_=FeatureKey("z", 0), feature_=Value("string"), node_id_=-1, flow_=None
        )
        assert ref["a"]["a_x"] == ref["a", "a_x"] == ref.a.a_x == FeatureRef(
            key_=FeatureKey("a", "a_x"), feature_=Value("int32"), node_id_=-1, flow_=None
        )

