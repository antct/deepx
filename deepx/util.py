import functools
import gc
import os
import random

import numpy as np
import torch


def seed_everything(seed, workers=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def find_batch_size(func):
    _statements = [
        "CUDA out of memory",
        "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED",
        "DefaultCPUAllocator: can't allocate memory",
        "CUDA error: out of memory"
    ]

    @functools.wraps(func)
    def wrapper_sender(self, *args, **kwargs):
        batch_size = self.config.per_device_train_batch_size
        gc.collect()
        torch.cuda.empty_cache()
        while True:
            if batch_size == 0:
                raise RuntimeError("batch size must > 0")
            try:
                return func(self, *args, **kwargs)
            except RuntimeError as exception:
                oom_flag = False
                if isinstance(exception, RuntimeError) and len(exception.args) == 1:
                    oom_flag = any(err in exception.args[0] for err in _statements)
                if oom_flag:
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.config.per_device_train_batch_size //= 2
                    self.log("oom, batch size {} -> {}".format(batch_size, self.config.per_device_train_batch_size))
                    batch_size = self.config.per_device_train_batch_size
                    if batch_size == 0:
                        raise RuntimeError("batch size must > 0")
                    self.load_dataloader()
                    self.load_optimizer_and_scheduler()
                else:
                    raise exception
            except Exception:
                raise
    return wrapper_sender


class EarlyStopping:
    def __init__(self, monitor_key='acc', mode='max', min_delta=0, patience=10, percentage=False):
        self.monitor_key = monitor_key
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best_metric = None
        self.best_metrics = None
        self.bad_epochs = 0
        self.is_better = None
        self.is_stop = False
        self.init_is_better(mode, min_delta, percentage)

    def step(self, metrics, steps=-1):
        if self.monitor_key not in metrics:
            raise RuntimeError('monitor key {} not exists'.format(self.monitor_key))

        metrics['steps'] = steps
        metric = metrics[self.monitor_key]

        if self.best_metric is None:
            self.is_best = True
            self.best_metric = metric
            self.best_metrics = metrics
            return

        if np.isnan(metric):
            self.is_stop = True
            return

        if self.is_better(metric, self.best_metric):
            self.is_best = True
            self.bad_epochs = 0
            self.best_metric = metric
            self.best_metrics = metrics
        else:
            self.is_best = False
            self.bad_epochs += 1

        if self.bad_epochs >= self.patience:
            self.is_stop = True

    def init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise RuntimeError('better mode {} is unknown'.format(mode))
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (best * min_delta / 100)


def on_main_process(func):
    """
    Run func on main process
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.accelerator.is_main_process:
            return func(self, *args, **kwargs)
    return wrapper


def on_local_main_process(func):
    """
    Run func on local main process
    """
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.accelerator.is_local_main_process:
            return func(self, *args, **kwargs)
    return wrapper


def ensure_prepared(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        self.prepare()
        return func(self, *args, **kwargs)
    return wrapper
