"""Jinja2 Template Data Processor."""
import warnings
from functools import partial
from typing import Any, Callable, Iterable

from datasets import ClassLabel, Features, Sequence, Value
from datasets.features.features import FeatureType
from jinja2 import Environment

from hyped.common.feature_checks import (
    get_sequence_feature,
    get_sequence_length,
)
from hyped.common.feature_key import FeatureKey
from hyped.data.processors.base import (
    BaseDataProcessor,
    BaseDataProcessorConfig,
)


def build_sample(feature: FeatureType) -> dict[str, Any]:
    """Build a sample according to the given feature.

    Arguments:
        feature (FeatureType):
            feature type for which to build a sample value for

    Returns:
        sample (dict[str, Any]):
            sample matching the given feature
    """
    if isinstance(feature, (dict, Features)):
        return {key: build_sample(val) for key, val in feature.items()}

    if isinstance(feature, (list, Sequence)):
        f = get_sequence_feature(feature)
        l = get_sequence_length(feature)
        return [build_sample(f) for _ in range(l if (l != -1) else 1)]

    if isinstance(feature, ClassLabel):
        assert feature.num_classes > 0
        return 0

    if isinstance(feature, Value):
        return feature.pa_type.to_pandas_dtype()(0)

    raise TypeError("Unexpected feature type: %s" % feature)


def _map_to_none(*args, **kwargs):
    """Helper function mapping always to none."""
    return None


class _set_filters(object):
    """Context manager setting and resetting environment filters."""

    def __init__(
        self, env: Environment, filters: dict[str, Callable[[Any], Any]]
    ) -> None:
        """Initializer.

        Arguments:
            env (jinja2.Environment):
                Jinja2 environment to set filters for
            filters (dict[str, Callable[[Any], Any]):
                collection of filters to set when entering the context manager
        """
        self.env = env
        self.filters = filters
        self.cached_filters = None

    def __enter__(self) -> None:
        """Set filters."""
        self.cached_filters = self.env.filters.copy()
        self.env.filters.update(self.filters)

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        """Reset filters."""
        self.env.filters = self.cached_filters


class Jinja2Config(BaseDataProcessorConfig):
    """Jinja2 Template Data Processor Config.

    Creates a new feature by applying a jinja2 template
    to the given datapoints.

    Access to the specific example values and their features is given
    through `FeatureKey` instances and jinja2 filters. Specifically see
    the following example on how to access the respective values:

    .. highlight:: python
    .. code-block:: python

        valA = "The value of A is {{ FeatureKey('A') | index_example }}."
        featA = "The feature of A is {{ FeatureKey('A') | index_features }}."

    Attributes:
        template (str):
            string template to apply
        output (str):
            output column to store the rendered template at
    """

    template: str
    output: str

    @property
    def required_feature_keys(self) -> Iterable[FeatureKey]:
        """Required Feature Keys.

        Not computed on configuration level, please use the
        `required_feature_keys` property of the respective `Jinja2`
        processor.

        Raises TypeError
        """
        raise TypeError(
            "`required_feature_keys` are not collected on configuration "
            "level. Please refer to the corresponding property of the "
            "`Jinja2` processor instance."
        )


class Jinja2(BaseDataProcessor[Jinja2Config]):
    """Jinja2 Template Data Processor Config.

    Creates a new feature by applying a jinja2 template
    to the given datapoints.

    Access to the specific example values and their features is given
    through `FeatureKey` instances and jinja2 filters. Specifically see
    the following example on how to access the respective values:

    .. highlight:: python
    .. code-block:: python

        valA = "The value of A is {{ FeatureKey('A') | index_example }}."
        featA = "The feature of A is {{ FeatureKey('A') | index_features }}."
    """

    def __init__(self, config: Jinja2Config) -> None:
        """Instantiate a new Jinja2 Template Data Processor.

        Arguments:
            config (Jinja2Config):
                config of the data processor
        """
        super(Jinja2, self).__init__(config)
        # set up the jinja environment
        self.env = Environment()
        self.env.filters = {
            "index_example": _map_to_none,
            "index_features": _map_to_none,
        }
        # create template
        self.template = self.env.from_string(self.config.template)
        # collect feature keys mentioned in template
        self.feature_keys: None | set[FeatureKey] = None

    def _collect_required_feature_keys(self, features: Features) -> None:
        """Collect all feature keys referenced in the template.

        Clears the current feature keys set and adds all feature keys
        referenced in the template, given a specific dataset feature
        mapping.

        Arguments:
            features (Features):
                input dataset features
        """
        self.feature_keys = set()
        example = build_sample(features)

        def _index_example(key):
            """Collect feature key and build value matching feature."""
            self.feature_keys.add(key)
            return key.index_example(example)

        def _index_features(key):
            """Collect feature key and index features."""
            self.feature_keys.add(key)
            return key.index_features(features)

        with _set_filters(
            self.env,
            {
                "index_example": _index_example,
                "index_features": _index_features,
            },
        ):
            try:
                # render template with collection features
                self.template.render(FeatureKey=FeatureKey)
            except Exception as e:
                warnings.warn(
                    "Encountered an expcetion while collecting referenced "
                    "feature keys in the template: %s" % str(e),
                    RuntimeWarning,
                )

    @property
    def required_feature_keys(self) -> set[FeatureKey]:
        """Input dataset feature keys required for execution of the processor.

        In this case the required feature keys are not inferred from the
        configuration of the data processor directly, but instead are
        extracted from the template.

        Returns:
            feature_keys (set[FeatureKey]):
                set of required feature keys
        """
        if not self.is_prepared:
            raise RuntimeError(
                "Jinja2 Data Processor must be prepared before accessing "
                "the required feature keys."
            )
        # feature keys should be set after preparation
        assert self.feature_keys is not None
        return self.feature_keys

    def map_features(self, features: Features) -> Features:
        """Map features.

        Checks whether the feature keys referenced in the template are valid
        and returns the output feature.

        Arguments:
            features (datasets.Features):
                input dataset features

        Returns:
            out_features: (datasets.Features):
                output dataset features
        """
        self._collect_required_feature_keys(features)
        # check if all features exist
        for key in self.feature_keys:
            key.index_features(features)
        # create output feature
        return {self.config.output: Value("string")}

    def process(
        self, example: dict[str, Any], index: int, rank: int
    ) -> dict[str, Any]:
        """Process example.

        Renders the template based on the given example and it's
        features.

        Arguments:
            example (dict[str, Any]):
                example to process
            index (int):
                dataset index of the example
            rank (int):
                execution process rank

        Returns:
            out (dict[str, Any]):
                output exmaple containing the rendered template
        """
        # set filters to use actual values
        with _set_filters(
            self.env,
            {
                "index_example": partial(
                    FeatureKey.index_example, example=example
                ),
                "index_features": partial(
                    FeatureKey.index_features, features=self.in_features
                ),
            },
        ):
            return {
                self.config.output: self.template.render(FeatureKey=FeatureKey)
            }
