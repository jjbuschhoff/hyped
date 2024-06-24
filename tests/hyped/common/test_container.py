from typing import Tuple, Union

import pytest
from pydantic import ValidationError

from hyped.common.container import NestedContainer


@pytest.fixture
def setup_data():
    simple_data = 42
    dict_data = {"a": 1, "b": {"c": 3, "d": 4}}
    list_data = [1, [2, 3], 4]
    return simple_data, dict_data, list_data


def test_simple_data(setup_data):
    simple_data, _, _ = setup_data
    container = NestedContainer[int](data=simple_data)
    assert container.data == simple_data
    assert container.unpack() == simple_data
    assert container.flatten() == {(): simple_data}


def test_dict_data(setup_data):
    _, dict_data, _ = setup_data
    container = NestedContainer[int](data=dict_data)
    assert container.data["a"].data == 1
    assert container.data["b"].data["c"].data == 3
    assert container.data["b"].data["d"].data == 4
    assert container.unpack() == dict_data
    assert container.flatten() == {("a",): 1, ("b", "c"): 3, ("b", "d"): 4}


def test_list_data(setup_data):
    _, _, list_data = setup_data
    container = NestedContainer[int](data=list_data)
    assert container.data[0].data == 1
    assert container.data[1].data[0].data == 2
    assert container.data[1].data[1].data == 3
    assert container.data[2].data == 4
    assert container.unpack() == list_data
    assert container.flatten() == {(0,): 1, (1, 0): 2, (1, 1): 3, (2,): 4}


def test_map_function(setup_data):
    _, dict_data, _ = setup_data

    container = NestedContainer[int](data=dict_data)

    def increment(path: Tuple[Union[str, int]], value: int) -> int:
        return value + 1

    mapped_container = container.map(increment, int)
    assert mapped_container.data["a"].data == 2
    assert mapped_container.data["b"].data["c"].data == 4
    assert mapped_container.data["b"].data["d"].data == 5


def test_invalid_data():
    with pytest.raises(ValidationError):
        NestedContainer[str](data=[1, 2, 3])


def test_complex_structure():
    complex_data = {
        "key1": [1, 2, {"nested_key1": 3, "nested_key2": [4, 5]}],
        "key2": {"key3": [6, 7]},
    }
    container = NestedContainer[int](data=complex_data)
    expected_flattened = {
        ("key1", 0): 1,
        ("key1", 1): 2,
        ("key1", 2, "nested_key1"): 3,
        ("key1", 2, "nested_key2", 0): 4,
        ("key1", 2, "nested_key2", 1): 5,
        ("key2", "key3", 0): 6,
        ("key2", "key3", 1): 7,
    }
    assert container.flatten() == expected_flattened
    assert container.unpack() == complex_data
