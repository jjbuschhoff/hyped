from hyped.data.ref import FeatureRef
from hyped.data.processors.base import BaseDataProcessorConfig, BaseDataProcessor, Batch
from hyped.data.processors.base.inputs import InputRefs, FeatureValidator
from hyped.data.processors.base.outputs import OutputRefs, LambdaOutputFeature
from typing import Annotated, Any


class NoOpInputRefs(InputRefs):
    x: Annotated[
        FeatureRef,
        FeatureValidator(lambda *args: None)
    ]


class NoOpOutputRefs(OutputRefs):
    y: Annotated[
        FeatureRef,
        LambdaOutputFeature(lambda _, i: i.x.feature_)
    ]


class NoOpConfig(BaseDataProcessorConfig):
    pass


class NoOp(BaseDataProcessor[NoOpConfig, NoOpInputRefs, NoOpOutputRefs]):

    def __init__(self) -> None:
        super(NoOp, self).__init__(config=NoOpConfig())

    async def process(
        self,
        inputs: Batch,
        index: int,
        rank: int
    ) -> Batch:
        return {"y": inputs["x"]}
