import os
import os.path as osp
from datetime import datetime

from mmcv.runner.hooks import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook

from openmixup.utils import get_root_logger
from openmixup.utils.dist_utils import master_only


@HOOKS.register_module
class OMPWandbLoggerHook(LoggerHook):
    """Log metrics to wandb.

    Combines train and val metrics into single log per epoch.
    Handles cases where validation is skipped (e.g., not every epoch).

    Args:
        api_key_file (str): File containing wandb API key in project root.
            Default: '.wandb.apikey'. Set to None to ignore.
        init_kwargs (dict): Arguments passed to wandb.init().
            Default: None.
        interval (int): Checking interval. Default: 1.
        ignore_last (bool): Ignore last log if less than interval. Default: True.
        reset_flag (bool): Clear buffer after logging. Default: False.
        by_epoch (bool): Use EpochBasedRunner. Default: True.
    """

    def __init__(
        self,
        api_key_file=".wandb.apikey",
        init_kwargs=None,
        interval=1,
        ignore_last=True,
        reset_flag=False,
        by_epoch=True,
    ):
        super().__init__(interval, ignore_last, reset_flag, by_epoch)
        self.api_key_file = api_key_file
        self.init_kwargs = init_kwargs or {}
        self._train_metrics = None
        self._skip = False
        self.wandb = None

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

    def _get_api_key(self):
        api_key = os.environ.get("WANDB_API_KEY")
        if api_key:
            return api_key
        if self.api_key_file is None:
            return None
        key_path = osp.join(
            osp.dirname(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))),
            self.api_key_file,
        )
        if osp.exists(key_path):
            with open(key_path) as f:
                return f.read().strip()
        return None

    def import_wandb(self):
        try:
            import wandb

            self.wandb = wandb
        except ImportError:
            self.wandb = None

    @master_only
    def before_run(self, runner):
        self.import_wandb()

        logger = getattr(runner, "logger", None) or get_root_logger()

        if self.wandb is None:
            logger.warning("WandbLoggerHook: wandb is not installed. Skipping wandb logging.")
            self._skip = True
            return

        api_key = self._get_api_key()
        if api_key is None:
            logger.warning(
                "WandbLoggerHook: No API key found (checked WANDB_API_KEY "
                f"env var and {self.api_key_file}). Skipping wandb init."
            )
            self._skip = True
            return

        self._skip = False

        init_kwargs = self.init_kwargs.copy()

        cfg = getattr(runner, "cfg", None)
        dataset_name = self._get_dataset_from_path(cfg)
        exp_name = runner.meta.get("exp_name", "unknown").split(".py")[0]
        timestamp = getattr(runner, "timestamp", None)
        if not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name_parts = [part for part in (dataset_name, exp_name, timestamp) if part]
        if run_name_parts and "name" not in init_kwargs:
            init_kwargs["name"] = "-".join(run_name_parts)

        self.wandb.login(key=api_key, relogin=True)

        self.wandb.init(**init_kwargs)

        self.wandb.config.update(
            {
                "exp_name": runner.meta.get("exp_name", "unknown"),
                "seed": runner.meta.get("seed", 0),
            }
        )
        if dataset_name:
            self.wandb.config.update({"dataset": dataset_name})

        if cfg is None:
            logger.warning("WandbLoggerHook: runner has no cfg attribute; skipping config upload.")
        else:
            cfg_filename = getattr(cfg, "filename", None)
            if cfg_filename:
                try:
                    self.wandb.config.update({"config_file": cfg_filename})
                except Exception:
                    pass
            self._log_config(cfg)

    def _log_config(self, cfg, prefix=""):
        for key, value in cfg.items():
            if isinstance(value, dict):
                self._log_config(value, prefix=f"{prefix}{key}.")
            else:
                try:
                    self.wandb.config.update({f"{prefix}{key}": value})
                except Exception:
                    pass

    def _get_dataset_from_path(self, cfg):
        if cfg is None:
            return None
        cfg_path = getattr(cfg, "filename", None) or getattr(cfg, "_filename", None)
        if not cfg_path:
            return None
        parts = cfg_path.split(os.sep)
        try:
            configs_idx = parts.index("configs")
        except ValueError:
            return None
        if len(parts) <= configs_idx + 2:
            return None
        return parts[configs_idx + 2]

    def log(self, runner):
        if self._skip:
            return
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

    @master_only
    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        tags = self.get_loggable_tags(runner)
        logger = get_root_logger()

        if tags and self.get_mode(runner) == "val" and self._train_metrics is not None:
            combined = self._train_metrics.copy()
            combined.update(tags)
            combined["epoch"] = runner.epoch + 1
            combined["iter"] = runner.iter
            progress = self._progress_percent(runner)
            if progress is not None:
                combined["progress"] = progress
            self.wandb.log(combined, step=combined["epoch"] if self.by_epoch else combined["iter"])
            logger.info("Logged combined metrics to wandb")
        elif self._train_metrics is not None:
            log_entry = self._train_metrics.copy()
            log_entry["epoch"] = runner.epoch + 1
            log_entry["iter"] = runner.iter
            progress = self._progress_percent(runner)
            if progress is not None:
                log_entry["progress"] = progress
            self.wandb.log(log_entry, step=log_entry["epoch"] if self.by_epoch else log_entry["iter"])
            logger.info("Logged train metrics to wandb")
        else:
            logger.info(f"No metrics to log. skip is: {self._skip}")

    @master_only
    def after_run(self, runner):
        if self._skip or self.wandb is None:
            return
        self.wandb.finish()
