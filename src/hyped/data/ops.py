from .ref import FeatureRef
from hyped.data.processors.ops.collect import CollectFeatures

def collect(
    collection: None | dict[str, ...] | list[...] = None, **kwargs
) -> FeatureRef:
    return CollectFeatures().call(collection=collection, **kwargs)
