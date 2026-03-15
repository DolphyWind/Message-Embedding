from threading import Thread
from typing import Any, Optional, TypeVar, Iterator
import mlflow
from contextlib import nullcontext
from queue import Queue


T = TypeVar("T")


def repeat_last(lst: list[T]) -> Iterator[T]:
    if not lst:
        return
    for item in lst:
        yield item
    while True:
        yield lst[-1]


class MLFlowLogger:
    def __init__(self, run_id: Optional[str]):
        self.run_id = run_id
        self.__queue: Queue = Queue()
        self.__main_thread = Thread(target=self.__mainloop)
        self.__main_thread.start()

    def log_param(self, name: str, value: Any, **kwargs):
        self.__queue.put(lambda: mlflow.log_param(name, value, **kwargs))

    def log_metric(self, key: str, value: float, **kwargs):
        self.__queue.put(lambda: mlflow.log_metric(key=key, value=value, **kwargs))

    def __mainloop(self):
        while True:
            task = self.__queue.get()
            if task is None:
                break
            active = mlflow.active_run()
            if active is None:
                mlflow.start_run(run_id=self.run_id)

            for timeout in repeat_last([0.1, 1, 3, 9, 30, 90]):
                try:
                    task()
                except Exception as e:
                    print(f"{e}\nAn exception occured when logging. Trying again in {timeout} seconds.")
                else:
                    break
            self.__queue.task_done()

    def stop(self):
        self.__queue.put(None)
        self.__main_thread.join()


class NullLogger:
    def __init__(self, _=None):
        self.run_id: Optional[str] = None

    def log_param(self, *args, **kwargs):
        pass

    def log_metric(self, *args, **kwargs):
        pass

    def stop(self):
        pass
