from collections import defaultdict
import logging
import numpy as np
import torch


class Logger:
    def __init__(self, console_logger):
        self.console_logger: logging.Logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False

        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name):
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value
        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict):
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key, value, t, to_sacred=True):
        if isinstance(value, torch.Tensor):
            value = value.cpu()
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

    def print_recent_stats(self):
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1])
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            item = "{:.4f}".format(
                np.mean([x[1] for x in self.stats[k][-window:]]))
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)


# set up a custom logger
def get_logger():
    """
    Loggerの初期設定
    """
    logger = logging.getLogger()
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    logger.setLevel(logging.DEBUG)

    # コンソール出力用
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    # コンソールにはINFOレベル以上のみ出力する
    stream_handler.setLevel(logging.INFO)

    # ファイル出力用
    logFilePath = "results/run_full.log"
    # w+にしないと上書きされない
    file_handler = logging.FileHandler(filename=logFilePath, mode="w+")
    file_handler.setFormatter(formatter)
    # ファイルにはすべて残す
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger
