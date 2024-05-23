import asyncio
import multiprocessing as mp
import os
import pickle
from contextlib import nullcontext

import pytest

from hyped.common.lazy import (
    LazyInstance,
    LazySharedInstance,
    LazyStaticInstance,
)


def factory(pid=None):
    return "INSTANCE(%s)" % (pid or os.getpid())


def test_getitem():
    assert LazyStaticInstance(factory)[:4] == "INST"


@pytest.mark.asyncio
async def test_context_manager():
    with LazyStaticInstance(nullcontext):
        pass
    async with LazyStaticInstance(nullcontext):
        pass


class TestStaticLazyInstance(object):
    @pytest.fixture
    def obj(self):
        return LazyStaticInstance[str](factory)

    def test_pickel(self, obj):
        val = obj.lower()
        reconstructed = pickle.loads(pickle.dumps(obj))
        assert val == reconstructed.lower()

    def test_case(self, obj):
        assert not obj._is_instantiated()
        # interact with the object and check the instance
        assert obj.lower() == factory().lower()
        assert obj._is_instantiated()


class TestLazyInstance(TestStaticLazyInstance):
    @pytest.fixture
    def obj(self):
        return LazyInstance[str](factory)

    def _mp_worker(self, obj, value):
        # should have a different pid since the obj instance
        # is created within the worker process and the value
        # is coming from outside
        assert obj.lower() != value.lower()

    def test_same_instance(self):
        # check instance is not recreated when not needed to
        obj = LazyInstance(object)
        assert obj._get_instance() == obj._get_instance()

    def test_case_mp(self, obj):
        p = mp.Process(
            target=self._mp_worker,
            args=(
                obj,
                factory(),
            ),
        )
        p.start()
        p.join()
        # check error in process
        assert p.exitcode == 0

    def test_case_new_loop(self, obj):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # create instance
        obj.lower()
        loop_hash_A = object.__getattribute__(obj, "_loop_hash")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # create instance
        obj.lower()
        loop_hash_B = object.__getattribute__(obj, "_loop_hash")

        assert loop_hash_A != loop_hash_B


class TestLazySharedInstance(TestStaticLazyInstance):
    def obj_factory(self):
        return LazySharedInstance[str]("test_shared_instance", factory)

    @pytest.fixture(scope="function")
    def obj(self):
        # save environment before creation of shared instance
        env = os.environ.copy()
        yield self.obj_factory()
        # reset the environment to the previous state
        os.environ.clear()
        os.environ.update(env)

    def _mp_worker(self, expected_val=None):
        # create expected value
        expected_val = expected_val or factory()

        obj = self.obj_factory()
        # interact with the object and check the instance
        assert obj.lower() == expected_val.lower()
        assert obj._is_instantiated()

    def test_case_mp(self, obj):
        # lazy shared instance is created but not instantiated
        assert not obj._is_instantiated()
        # interact with the object and check the instance
        assert obj.lower() == factory().lower()
        assert obj._is_instantiated()
        # expected value contains process id of parent process
        # and not the one of the child process which would be
        # the case when factory function would be called from
        # within the child process
        p = mp.Process(target=self._mp_worker, args=(factory(),))
        p.start()
        p.join()
        # check error in process
        assert p.exitcode == 0

    def test_case_mp_2(self, obj):
        # lazy shared instance is created but not instantiated
        assert not obj._is_instantiated()
        p = mp.Process(target=self._mp_worker)
        p.start()
        p.join()
        # check error in process
        assert p.exitcode == 0

        assert obj.lower() == factory(p.pid).lower()
