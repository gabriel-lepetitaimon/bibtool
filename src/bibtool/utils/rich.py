from time import time

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)


class RichProgress:
    def __init__(self, name, total, done_message=None, parent=None) -> None:
        if parent is None:
            self.progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                transient=False,
            )
        else:
            self.progress = parent.progress
        self.nested = parent is not None
        self.name = name
        self.total = total
        self.done_message = done_message
        self.task = None
        self.t0 = None

    def __enter__(self):
        self.task = self.progress.add_task(self.name, total=self.total)
        self.t0 = time()

        if not self.nested:
            self.progress.start()

        return self

    def update(self, advance=None, completed=None, message=None, visible=True, total=None):
        if completed is not None:
            advance = None

        self.progress.update(
            self.task,
            advance=advance,
            description=message,
            completed=completed,
            visible=visible,
            total=total,
            refresh=True,
        )

    def __exit__(self, exc_type, exc_value, traceback):
        global _progress
        elapsed = time() - self.t0
        if exc_type is None:
            self.progress.update(self.task, visible=False)
            self.progress.remove_task(self.task)
            if self.done_message is not None:
                self.progress.console.print(self.done_message.replace("{t}", f"{elapsed:.1f}"))

        if not self.nested:
            self.progress.stop()

    @staticmethod
    def iteration(name, total, done_message=None):
        return RichProgress(name, total, done_message)
