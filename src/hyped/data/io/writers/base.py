"""Base Dataset Writer."""
import json
import os
from abc import ABC, abstractmethod
from typing import Any

import _io
from torch.utils.data._utils.worker import get_worker_info

from hyped.common.consumer import BaseDatasetConsumer


class BaseDatasetWriter(BaseDatasetConsumer, ABC):
    """Base Dataset Writer.

    Implements the `BaseDatasetConsumer` class to write a dataset
    to the disk in json-line format.
    """

    def __init__(
        self,
        save_dir: str,
        exist_ok: bool = False,
        num_proc: None | int = None,
        tqdm_kwargs: dict[str, Any] = {},
        tqdm_update_interval: float = 0.2,
    ) -> None:
        """Initialize dataset writer object.

        Arguments:
            save_dir (str): the directory to save the dataset in
            exist_ok (bool):
                whether it is ok to write to the directory if it
                already exists. Defaults to False.
            num_proc (None | int):
                The number of processes to use. Defaults to the number of
                cpu cores available.
            tqdm_kwargs (dict[str, Any]):
                extra keyword arguments passed to the tqdm progress bar
            tqdm_update_interval (float):
                the update interval in seconds in which the tqdm bar
                is updated
        """
        # initialize consumer
        super(BaseDatasetWriter, self).__init__(
            num_proc=num_proc,
            tqdm_kwargs=tqdm_kwargs,
            tqdm_update_interval=tqdm_update_interval,
        )
        # create save directory if needed
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=exist_ok)

    @abstractmethod
    def worker_shard_file_obj(
        self, path: str, worker_id: int
    ) -> _io.TextIOWrapper:
        """Worker Shard File Object.

        Return the file object used to store the data consumed by the
        worker of a given id.

        Arguments:
            path (str): path to store the file in
            worker_id (int): worker id

        Returns:
            file (_io.TextIOWrapper): file used by the worker of the given id
        """
        ...

    def initialize_worker(self) -> None:
        """Open the save file for the worker."""
        worker_info = get_worker_info()
        # open data save file
        worker_info.args.save_file = self.worker_shard_file_obj(
            self.save_dir, worker_info.id
        )
        # store file paths
        worker_info.args.save_file_path = worker_info.args.save_file.name
        worker_info.args.features_file_path = os.path.join(
            self.save_dir, "features.json"
        )

        if worker_info.id == 0:
            # save the datasets features
            with open(worker_info.args.features_file_path, "w+") as f:
                f.write(json.dumps(worker_info.dataset.features.to_dict()))

    def finalize_worker(self) -> None:
        """Cleanup and close the save file."""
        worker_info = get_worker_info()
        # check if the file is empty
        worker_info.args.save_file.seek(0, os.SEEK_END)
        is_empty = worker_info.args.save_file.tell() == 0
        # close the file
        worker_info.args.save_file.close()

        # delete the file if it is empty
        if is_empty:
            os.remove(worker_info.args.save_file_path)
