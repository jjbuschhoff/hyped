from __future__ import annotations
from abc import ABC
from itertools import chain
from functools import cached_property
from typing import (
    Any,
    TypeVar,
    Generic,
    ClassVar,
    Callable,
    get_origin,
    get_args
)
import typing
import typing_extensions
import inspect
import asyncio

from pydantic import BaseModel
from pydantic._internal._model_construction import ModelMetaclass
from datasets import Features
from datasets.features.features import FeatureType

from hyped.common.feature_ref import Feature, FeatureRef, FeatureCollection, FeatureKey
from hyped.common.feature_validators import FeatureValidator
from hyped.base.config import BaseConfig, BaseConfigurable


class input_refs_meta(ModelMetaclass):
    
    def __new__(cls, name, bases, attrs) -> type:
        # get all annotations and extract the feature generators        
        refs = attrs.get("__annotations__", {}).copy()
        # check all annotations
        for ref in refs.values():
            origin = get_origin(ref)
            args = get_args(ref)
            # all annotations should be annotated
            if origin not in [typing.Annotated, typing_extensions.Annotated]:
                raise TypeError()
            # make sure the type is always a feature reference
            # with a feature validator annotation
            if (
                (len(args) != 2)
                or (set(get_args(Feature)) != set(get_args(args[0])))
                or (not isinstance(args[1], FeatureValidator))
            ):
                raise TypeError()

        # create the type and set the feature generators
        return super().__new__(cls, name, bases, attrs)


class InputRefs(BaseModel, metaclass=input_refs_meta):

    @property
    def refs(self) -> set[Feature]:
        return set([getattr(self, key) for key in self.model_fields.keys()])


class LambdaOutputFeature(object):
    def __init__(self, f: Callable[[BaseDataProcessorConfig], FeatureType]) -> None:
        self.build_feature_type = f


class OutputFeature(LambdaOutputFeature):
    def __init__(self, feature_type: FeatureType) -> None:
        super(OutputFeature, self).__init__(lambda _: feature_type)


class output_refs_meta(ModelMetaclass):
    
    def __new__(cls, name, bases, attrs) -> type:
        # get all annotations and extract the feature generators        
        refs = attrs["__annotations__"].copy()
        gens = refs.pop("_feature_generators", {})
        # check all annotations
        for ref_name, ref in refs.items():
            origin = get_origin(ref)
            args = get_args(ref)
            # all annotations should be annotated
            if origin not in [typing.Annotated, typing_extensions.Annotated]:
                raise TypeError()
            # make sure the type is always a feature reference
            # with a feature type specifying annotation
            if (
                (len(args) != 2)
                or (args[0] is not FeatureRef)
                or (not isinstance(args[1], LambdaOutputFeature))
            ):
                raise TypeError()
            # add feature type specification to generators
            _, feature_gen = args
            gens[ref_name] = feature_gen
        # create the type and set the feature generators
        T = super().__new__(cls, name, bases, attrs)
        T._feature_generators = gens
        
        return T


class OutputRefs(
    FeatureRef, metaclass=output_refs_meta
):

    _feature_generators: ClassVar[dict[str, LambdaOutputFeature]]

    # do not use the dynamic FeatureRef getattr function
    __getattr__ = None

    def __init__(self, config: BaseDataProcessorConfig, node_id: int) -> None:

        features = Features({
            key: gen.build_feature_type(config)
            for key, gen in type(self)._feature_generators.items()
        })

        super(OutputRefs, self).__init__(
            key=FeatureKey(),
            feature=features,
            node_id=node_id,
            **{
                key: FeatureRef(
                    key=FeatureKey(key),
                    feature=feature,
                    node_id=node_id
                )
                for key, feature in features.items()
            }
        )
    
    @property
    def refs(self) -> set[Feature]:
        ignore_fields = FeatureRef.model_fields.keys()
        return set(
            [
                getattr(self, key) for key in self.model_fields.keys()
                if key not in ignore_fields
            ]
        )


I = TypeVar("I", bound=InputRefs)
O = TypeVar("O", bound=OutputRefs)


class BaseDataProcessorConfig(BaseConfig, Generic[I, O]):

    inputs: I
    outputs_: None | O = None
    # node id in the data flow graph
    # set when processor is added to the data flow
    node_id: None | int = None

    def __init__(self, **kwargs) -> None:

        if "outputs_" in kwargs:
            raise TypeError()

        if "node_id" in kwargs:
            raise TypeError()

        super(BaseDataProcessorConfig, self).__init__(**kwargs)


T = TypeVar("T", bound=BaseDataProcessorConfig)

class BaseDataProcessor(BaseConfigurable[T], ABC):
    
    def __init__(self, config: T) -> None:
        self._config = config
        # get the output ref type from the configuration
        out_refs_type = type(self).config_type.model_fields["outputs_"].annotation
        _, self._out_refs_type = get_args(out_refs_type)
        # check whether the process function is a coroutine
        self._is_process_async = inspect.iscoroutinefunction(self.process)

    @classmethod
    def from_config(cls, config: T) -> BaseDataProcessor:
        return cls(config)

    @property
    def config(self) -> T:
        return self._config

    @property
    def inputs(self) -> InputRefs:
        return self._config.inputs

    @property
    def node_id(self) -> int:

        if self.config.node_id is None:
            raise RuntimeError()

        return self.config.node_id

    @node_id.setter
    def node_id(self, node_id: int) -> None:

        if self.config.node_id is not None:
            raise RuntimeError()

        self.config.node_id = node_id

    @cached_property
    def out_features(self) -> U:
        return self._out_refs_type(self.config, self.node_id)

    async def batch_process(
        self, inputs: dict[FeatureRef, list[Any]], index: list[int], rank: int
    ) -> tuple[dict[FeatureRef, list[Any]], list[int]]:

        # apply process function to each sample in the input batch
        keys = inputs.keys()
        outputs = [
            self.process(dict(zip(keys, values)), i, rank)
            for i, values in zip(index, zip(*inputs.values()))
        ]
        # gather all outputs in case the process function
        # is a coroutine
        if self._is_process_async:
            outputs = await asyncio.gather(*outputs)

        # pack output samples to batch format
        return {
            key: [d[key] for d in outputs]
            for key in self.out_features.refs
        }, index

    async def process(
        self, inputs: dict[FeatureRef, Any], index: int, rank: int
    ) -> dict[FeatureRef, Any]:
        ...

    def process(
        self, inputs: dict[FeatureRef, Any], index: int, rank: int
    ) -> dict[FeatureRef, Any]:
        ...

