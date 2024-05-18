from __future__ import annotations

from typing import Annotated, Any

from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
    Batch,
)
from hyped.data.processors.base.inputs import FeatureValidator, InputRefs
from hyped.data.processors.base.outputs import LambdaOutputFeature, OutputRefs
from hyped.data.ref import FeatureRef


class NoOpInputRefs(InputRefs):
    x: Annotated[FeatureRef, FeatureValidator(lambda *args: None)]


class NoOpOutputRefs(OutputRefs):
    y: Annotated[FeatureRef, LambdaOutputFeature(lambda _, i: i.x.feature_)]


class NoOpConfig(BaseDataProcessorConfig):
    pass


class NoOp(BaseDataProcessor[NoOpConfig, NoOpInputRefs, NoOpOutputRefs]):
    def __init__(self) -> None:
        super(NoOp, self).__init__(config=NoOpConfig())

    @classmethod
    def from_config(cls, config: NoOpConfig) -> NoOp:
        return cls()

    async def process(self, inputs: Batch, index: int, rank: int) -> Batch:
        return {"y": inputs["x"]}
