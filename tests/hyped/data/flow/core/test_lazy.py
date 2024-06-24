from types import MappingProxyType
from unittest.mock import AsyncMock, MagicMock

from hyped.data.flow.core.lazy import LazyFlowOutput


def old_test_lazy_flow():
    input_proxy = {"x": 0}
    executor = MagicMock()
    executor.execute = AsyncMock(return_value={"y": [1]})
    executor.collect.feature_.keys = MagicMock()

    obj = LazyFlowOutput(input_proxy, executor)
    assert obj.keys() == executor.collect.feature_.keys()


def test_lazy_flow_initialization():
    input_proxy = MappingProxyType({"x": 0})
    executor = MagicMock()

    obj = LazyFlowOutput(input_proxy, executor)

    assert obj._proxy == input_proxy
    assert obj._executor == executor
    assert obj._proxy_snapshot is None
    assert obj._out_snapshot is None


def test_lazy_flow_keys():
    input_proxy = MappingProxyType({"x": 0})
    executor = MagicMock()
    executor.collect.feature_.keys = MagicMock(return_value=["y"])

    obj = LazyFlowOutput(input_proxy, executor)

    assert list(obj.keys()) == ["y"]
    executor.collect.feature_.keys.assert_called_once()


def test_lazy_flow_getitem():
    input_dict = {"x": 0}
    input_proxy = MappingProxyType(input_dict)
    executor = MagicMock()
    executor.execute = AsyncMock(return_value={"y": [1]})
    executor.collect.feature_.keys = MagicMock(return_value=["y"])

    obj = LazyFlowOutput(input_proxy, executor)
    prev_snapshot = obj._proxy_snapshot

    value = obj["y"]

    assert value == 1
    executor.execute.assert_called_once()
    assert obj._out_snapshot == {"y": 1}
    assert obj._proxy_snapshot != prev_snapshot

    # change the input and make sure the flow is executed again
    executor.reset_mock()
    input_dict["x"] = 1
    prev_snapshot = obj._proxy_snapshot

    value = obj["y"]

    executor.execute.assert_called_once()
    assert obj._proxy_snapshot != prev_snapshot


def test_lazy_flow_getitem_keyerror():
    input_proxy = MappingProxyType({"x": 0})
    executor = MagicMock()
    executor.collect.feature_.keys = MagicMock(return_value=["y"])

    obj = LazyFlowOutput(input_proxy, executor)

    try:
        obj["z"]
    except KeyError as e:
        assert str(e) == "'z'"


def test_lazy_flow_iteration():
    input_proxy = MappingProxyType({"x": 0})
    executor = MagicMock()
    executor.collect.feature_.keys = MagicMock(return_value=["y"])

    obj = LazyFlowOutput(input_proxy, executor)
    keys = list(iter(obj))

    assert keys == ["y"]


def test_lazy_flow_length():
    input_proxy = MappingProxyType({"x": 0})
    executor = MagicMock()
    executor.collect.feature_.keys = MagicMock(return_value=["y"])

    obj = LazyFlowOutput(input_proxy, executor)
    assert len(obj) == 1


def test_lazy_flow_str():
    input_proxy = MappingProxyType({"x": 0})
    executor = MagicMock()
    executor.collect.feature_.keys = MagicMock(return_value=["y"])
    executor.execute = AsyncMock(return_value={"y": [1]})

    obj = LazyFlowOutput(input_proxy, executor)
    _ = obj["y"]

    assert str(obj) == str({"y": 1})


def test_lazy_flow_repr():
    input_proxy = MappingProxyType({"x": 0})
    executor = MagicMock()

    obj = LazyFlowOutput(input_proxy, executor)

    assert (
        repr(obj)
        == f"LazyFlowOutput(input_proxy={input_proxy}, executor={executor})"
    )
