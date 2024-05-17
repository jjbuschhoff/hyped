from typing import Callable, ClassVar

from hyped.data.ref import FeatureRef
from hyped.common.pydantic import BaseModelWithTypeValidation

from datasets.features.features import FeatureType, Features
from .base import BaseDataProcessorConfig

from .inputs import InputRefs

class LambdaOutputFeature(object):
    def __init__(self, f: Callable[
        [BaseDataProcessorConfig, InputRefs], FeatureType]
    ) -> None:
        self.build_feature_type = f


class OutputFeature(LambdaOutputFeature):
    def __init__(self, feature_type: FeatureType) -> None:
        super(OutputFeature, self).__init__(lambda _, __: feature_type)


class OutputRefs(FeatureRef, BaseModelWithTypeValidation):

    _feature_generators: ClassVar[dict[str, LambdaOutputFeature]]
    _feature_names: ClassVar[set[str]]
    # do not use the dynamic FeatureRef getattr function
    __getattr__ = None
    
    @classmethod
    def type_validator(cls) -> None:

        cls._feature_generators = {}
        cls._feature_names = set()
        # ignore all fields from the feature ref base type
        ignore_fields = FeatureRef.model_fields.keys()

        for name, field in cls.model_fields.items():
            if name in ignore_fields:
                continue
            # each field should be a feature ref with
            # an output feature type annotation
            if not (
                issubclass(field.annotation, FeatureRef)
                and len(field.metadata) == 1
                and isinstance(field.metadata[0], LambdaOutputFeature)
            ):
                raise TypeError(name)

            # add field to feature names and extract generator
            cls._feature_names.add(name)
            cls._feature_generators[name] = field.metadata[0]

    def __init__(
        self,
        config: BaseDataProcessorConfig,
        inputs: InputRefs,
        node_id: int,
    ) -> None:

        features = Features({
            key: gen.build_feature_type(config, inputs)
            for key, gen in type(self)._feature_generators.items()
        })

        flow = inputs.flow
        super(OutputRefs, self).__init__(
            key_=tuple(),
            feature_=features,
            node_id_=node_id,
            flow_=flow,
            **{
                key: FeatureRef(
                    key_=key,
                    feature_=feature,
                    node_id_=node_id,
                    flow_=flow
                )
                for key, feature in features.items()
            }
        )
    
    @property
    def refs(self) -> set[FeatureRef]:
        ignore_fields = FeatureRef.model_fields.keys()
        return set(
            [
                getattr(self, key) for key in self.model_fields.keys()
                if key not in ignore_fields
            ]
        )
