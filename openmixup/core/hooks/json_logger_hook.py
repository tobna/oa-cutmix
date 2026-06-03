import json

from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook


@HOOKS.register_module
class JsonLoggerHook(LoggerHook):
    """Log epoch metrics as JSON to stdout.

    Combines train and val metrics into single JSON line per epoch.
    Handles cases where validation is skipped (e.g., not every epoch).
    """

    def __init__(self, interval=1, ignore_last=True, reset_flag=False, by_epoch=True):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self._train_metrics = None

    def _progress_percent(self, runner):
        max_epochs = getattr(runner, "max_epochs", None)
        if max_epochs and max_epochs > 0:
            progress = (runner.epoch + 1) / max_epochs
        else:
            max_iters = getattr(runner, "max_iters", None)
            if max_iters and max_iters > 0:
                progress = (runner.iter + 1) / max_iters
            else:
                return None
        return max(0.0, min(round(progress * 100, 2), 100.0))

    def log(self, runner):
        if not self.by_epoch:
            return
        if not self.end_of_epoch(runner):
            return

        tags = self.get_loggable_tags(runner)
        if not tags:
            return

        mode = self.get_mode(runner)
        if mode == "train":
            self._train_metrics = tags.copy()

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        tags = self.get_loggable_tags(runner)

        if tags and self.get_mode(runner) == "val" and self._train_metrics is not None:
            combined = self._train_metrics.copy()
            combined.update(tags)
            combined["epoch"] = runner.epoch + 1
            combined["iter"] = runner.iter
            progress = self._progress_percent(runner)
            if progress is not None:
                combined["progress"] = progress
            print("\n" + json.dumps(combined) + "\n", flush=True)
        elif self._train_metrics is not None:
            log_entry = self._train_metrics.copy()
            log_entry["epoch"] = runner.epoch + 1
            log_entry["iter"] = runner.iter
            progress = self._progress_percent(runner)
            if progress is not None:
                log_entry["progress"] = progress
            print("\n" + json.dumps(log_entry, sort_keys=True), flush=True)

        self._train_metrics = None
