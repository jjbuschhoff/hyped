from __future__ import annotations

from hyped.data.ref import FeatureRef
from hyped.data.processors.base import (
    Batch,
    BaseDataProcessorConfig,
    BaseDataProcessor,
)
from hyped.data.processors.base.inputs import InputRefs
from hyped.data.processors.base.outputs import OutputRefs, LambdaOutputFeature
from hyped.common.feature_checks import check_feature_equals

from pydantic import (
    Field,
    BaseModel,
    BeforeValidator,
)
from pydantic._internal._model_construction import ModelMetaclass
from datasets.features.features import FeatureType, Features, Sequence
from typing_extensions import Annotated


class FeatureCollection(BaseModel):

    collection: Annotated[
        (
            dict[str, FeatureCollection | FeatureRef]
            | Annotated[
                list[FeatureCollection | FeatureRef],
                Field(min_length=1)
            ]
        ),
        BeforeValidator(
            lambda x: (
                {
                    k: (
                        v if isinstance(v, FeatureRef)
                        else FeatureCollection(collection=v)
                    )
                    for k, v in x.items()
                } if isinstance(x, dict) else
                [
                    (
                        v if isinstance(v, FeatureRef)
                        else FeatureCollection(collection=v)
                    )
                    for v in x
                ] if isinstance(x, list) else
                x
            )
        )
    ]

    @property
    def feature(self) -> FeatureType:
        
        if isinstance(self.collection, dict):
            return Features(
                {
                    k: v.feature_
                    for k, v in self.collection.items()
                }
            )

        if isinstance(self.collection, list):
            # collect all features in specified in the list
            collected_features = (
                item.feature_ for item in self.collection
            )
            f = next(collected_features)
            # make sure the feature types match
            for ff in collected_features:
                if not check_feature_equals(f, ff):
                    raise TypeError(
                        "Expected all items of a sequence to be of the "
                        "same feature type, got %s != %s" % (str(f), str(ff))
                    )
            return Sequence(f, length=len(self.collection))

    @property
    def refs(self) -> set[FeatureRef]:

        feature_refs = set()

        for v in (
            self.collection.values() if isinstance(self.collection, dict) else self.collection
        ):
            if isinstance(v, FeatureCollection):
                feature_refs.update(v.refs)
            elif isinstance(v, FeatureRef):
                feature_refs.add(v)

        return feature_refs


class CollectFeaturesInputRefs(InputRefs):
    collection: FeatureCollection
 
    @classmethod
    def type_validator(cls) -> None:
        pass

    @property
    def named_refs(self) -> dict[str, FeatureRef]:
        return {str(hash(ref)): ref for ref in self.collection.refs}

    @property
    def flow(self) -> "DataFlowGraph":
        return next(iter(self.collection.refs)).flow_


class CollectFeaturesOutputRefs(OutputRefs):
    collected: Annotated[
        FeatureRef,
        LambdaOutputFeature(lambda _, i: i.collection.feature)
    ]

class CollectFeaturesConfig(BaseDataProcessorConfig):
    pass

class CollectFeatures(
    BaseDataProcessor[
        CollectFeaturesConfig,
        CollectFeaturesInputRefs,
        CollectFeaturesOutputRefs
    ]
):

    def __init__(self) -> None:
        self.collection: FeatureCollection | None = None
        super(CollectFeatures, self).__init__(
            config=CollectFeaturesConfig()
        )

    def call(
        self,
        inputs: None | CollectFeaturesInputRefs = None,
        collection: None | FeatureCollection = None,
        **kwargs
    ) -> int:
       
        if self.collection is not None:
            # feature collector is already in use
            # create a new one for this call
            return CollectFeatures().call(
                inputs=inputs, collection=collection, **kwargs
            )

        # check inputs
        if sum(
            [
                inputs is not None,
                collection is not None,
                len(kwargs) > 0
            ]
        ) > 1:
            raise ValueError(
                "Please specify either 'inputs', 'collection' or keyword "
                "arguments, but not multiple."
            )

        if collection is not None:
            # build collection from collection argument
            collection = (
                collection if isinstance(collection, FeatureCollection)
                else FeatureCollection(collection=collection)
            )
        elif len(kwargs) > 0:
            # build collection from keyword arguments
            collection = FeatureCollection(collection=kwargs)

        if inputs is None:
            # build input refs from collection
            inputs = CollectFeaturesInputRefs(collection=collection)

        # save collection
        self.collection = inputs.collection
        # call the processor
        return super(CollectFeatures, self).call(inputs=inputs)

    def collect_values(
        self,
        inputs: Batch,
        col: FeatureCollection | FeatureRef
    ) -> list[Any]:
      
        if isinstance(col, FeatureRef):
            return inputs[str(hash(col))]

        if isinstance(col.collection, dict):
            # collect values from sub-collection and
            # convert from dict of lists to list of dicts
            data = {
                k: self.collect_values(inputs, v)
                for k, v in col.collection.items()
            }
            return [
                dict(zip(data.keys(), values))
                for values in zip(*data.values())
            ]

        if isinstance(col.collection, list):
            # collect values from sub-collection and transpose
            data = (
                self.collect_values(inputs, v)
                for v in col.collection
            )
            return [list(row) for row in zip(*data)]

    async def batch_process(
        self,
        inputs: Batch,
        index: list[int],
        rank: int
    ) -> Batch:

        out = {
            "collected": self.collect_values(
                inputs=inputs,
                col=self.collection
            )
        }

        return out, index
