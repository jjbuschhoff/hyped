"""Lazy instance utilities."""
import asyncio
import os
import pickle
import tempfile
import warnings
from functools import partial
from time import sleep
from typing import Any, Callable, Generic, TypeVar

T = TypeVar("T")


class LazyStaticInstance(Generic[T]):
    """Lazy instance that is instantiated on first interaction.

    This class delays the creation of an instance until it is accessed or used.
    """

    __slots__ = ("_factory", "_instance")

    def __init__(self, factory: Callable[[], T]) -> None:
        """Initialize a LazyStaticInstance.

        Args:
            factory (Callable[[], T]): A factory function to create the instance.
        """
        self._factory = factory
        self._instance: None | T = None

    def _callback(self) -> None:
        """Instantiate the object if it has not been created yet."""
        if object.__getattribute__(self, "_instance") is None:
            self._instance = object.__getattribute__(self, "_factory")()

    def _get_instance(self) -> T:
        """Get the instance, creating it if necessary.

        Returns:
            T: The instantiated object.
        """
        self._callback()
        return object.__getattribute__(self, "_instance")

    def _is_instantiated(self) -> bool:
        """Check if the instance has been created.

        Returns:
            bool: True if the instance is created, False otherwise.
        """
        return object.__getattribute__(self, "_instance") is not None

    def __getstate__(self) -> dict[str, Any]:
        """Get the state of the instance for pickling.

        Returns:
            dict[str, Any]: The state of the instance.
        """
        return {
            "_factory": object.__getattribute__(self, "_factory"),
            "_instance": object.__getattribute__(self, "_instance"),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the instance for pickling.

        Args:
            state (dict[str, Any]): The state to set.
        """
        self._factory = state["_factory"]
        self._instance = state["_instance"]

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the instance.

        Args:
            name (str): The attribute name.

        Returns:
            Any: The attribute value.
        """
        return getattr(self._get_instance(), name)

    def __getitem__(self, *args) -> Any:
        """Forward item access to the instance.

        Args:
            *args: The item access arguments.

        Returns:
            Any: The item value.
        """
        return self._get_instance().__getitem__(*args)

    def __enter__(self) -> Any:
        """Enter the context of the instance.

        Returns:
            Any: The result of entering the context.
        """
        return self._get_instance().__enter__()

    def __exit__(self, *args) -> None:
        """Exit the context of the instance.

        Args:
            *args: The context exit arguments.
        """
        return self._get_instance().__exit__(*args)

    def __aenter__(self) -> Any:
        """Enter the asynchronous context of the instance.

        Returns:
            Any: The result of entering the async context.
        """
        return self._get_instance().__aenter__()

    def __aexit__(self, *args) -> None:
        """Exit the asynchronous context of the instance.

        Args:
            *args: The async context exit arguments.
        """
        return self._get_instance().__aexit__(*args)

    def __await__(self):
        """Await the instance.

        Returns:
            Any: The result of awaiting the instance.
        """
        return self._get_instance().__await__()


T = TypeVar("T")


class LazyInstance(LazyStaticInstance[T]):
    """Lazy instance that is recreated for new processes or event loops.

    This class ensures that the instance is recreated for every new process
    and whenever a new event loop is active.
    """

    __slots__ = ("_loop_hash", "_on_new_event_loop", "_on_new_process")

    def __init__(
        self,
        factory: Callable[[], T],
    ) -> None:
        """Initialize a LazyInstance.

        Args:
            factory (Callable[[], T]): A factory function to create the instance.
        """
        super(LazyInstance, self).__init__(factory)
        self._loop_hash: None | int = None

    def __getstate__(self) -> dict[str, Any]:
        """Get the state for pickling.

        Returns:
            dict[str, Any]: The state to pickle.
        """
        # don't pass instance and loop hash to processes
        return {"_factory": object.__getattribute__(self, "_factory")}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Set the state of the instance for pickling.

        Args:
            state (dict[str, Any]): The state to set.
        """
        # recreate the instance in a new sub-process
        self._factory = state["_factory"]
        self._instance = None
        self._loop_hash = None

    def _callback(self) -> None:
        """Instantiate the object if it has not been created or the event loop has changed."""
        with warnings.catch_warnings():
            # DeprecationWarning: There is no current event loop
            warnings.simplefilter("ignore", category=DeprecationWarning)

            loop = asyncio.get_event_loop()
            loop_hash = hash(loop)

        if (object.__getattribute__(self, "_instance") is None) or (
            object.__getattribute__(self, "_loop_hash") != loop_hash
        ):
            self._instance = self._factory()
            self._loop_hash = loop_hash


T = TypeVar("T")


def _load_from_shared_factory(
    tmp_file_name: str, factory: Callable[[], T]
) -> T:
    """Create or load an instance from a shared factory.

    This function handles the creation and sharing of an instance
    across multiple processes. If the instance is not yet created,
    it will be instantiated and saved to a file. If the instance
    is already being created by another process, it will wait until
    the instance is ready and then load it from the file.

    Args:
        tmp_file_name (str): The temporary file name used to store the instance.
        factory (Callable[[], T]): A factory function to create the instance.

    Returns:
        T: The shared instance.
    """
    if os.path.isfile("%s.registered" % tmp_file_name):
        # mark instance as pending
        os.rename(
            "%s.registered" % tmp_file_name,
            "%s.pending" % tmp_file_name,
        )
        # create instance
        instance = factory()
        # write instance to file
        with open("%s.pending" % tmp_file_name, "wb+") as f:
            f.write(pickle.dumps(instance))
        # mark instance as ready to use
        os.rename("%s.pending" % tmp_file_name, tmp_file_name)
        # return the instance
        return instance

    else:
        # wait for the instance to be ready
        while os.path.isfile("%s.pending" % tmp_file_name):
            sleep(0.1)
        assert os.path.isfile(tmp_file_name)
        # load instance from file
        with open(tmp_file_name, "rb") as f:
            return pickle.loads(f.read())


class LazySharedInstance(LazyStaticInstance[T]):
    """Lazy shared instance that is shared across subprocesses.

    This class ensures that an instance is created only once and
    shared across multiple subprocesses. The instance will be shared
    only with processes spawned after creating the lazy object, but
    the underlying instance can be created at a later time.
    """

    def __init__(self, identifier: str, factory: Callable[[], T]) -> None:
        """Initialize a LazySharedInstance.

        Args:
            identifier (str): The identifier used to track the instance across processes.
            factory (Callable[[], T]): A factory function to create the instance.
        """
        # environment keys used to share the object
        env_key = "__HYPED_SHARED_INSTANCE_%s" % identifier

        # that is executed in the parent process only
        if env_key not in os.environ:
            # file name storing object
            # mark as registered but not instantiated yet
            tmp_file_name = tempfile.NamedTemporaryFile().name
            os.environ[env_key] = tmp_file_name
            # create registered instance file
            if not os.path.isfile("%s.registered" % tmp_file_name):
                open("%s.registered" % tmp_file_name, "w").close()

        else:
            tmp_file_name = os.environ[env_key]

        wrapped_factory = partial(
            _load_from_shared_factory,
            tmp_file_name=tmp_file_name,
            factory=factory,
        )
        super(LazySharedInstance, self).__init__(wrapped_factory)
