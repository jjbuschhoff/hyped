from hyped.data.processors.ops.collect import CollectFeatures

from .ref import FeatureRef


def collect(
    collection: None | dict[str, ...] | list[...] = None, **kwargs
) -> FeatureRef:
    return CollectFeatures().call(collection=collection, **kwargs)
