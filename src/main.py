# ここの__main__関数が一番最初に実行される

import numpy as np
import os
# import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch as th
from utils.logging import get_logger
import yaml
import argparse

from run import run

# set to "no" if you want to see stdout/stderr in console
# Sacredの標準出力設定 "sys"(Windows), "fd"(Linux), "no"
SETTINGS["CAPTURE_MODE"] = "sys"

logger = get_logger()

# Sacredで実験を定義
ex = Experiment("foodbank_qmix")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.main
def my_main(_run, _config, _log):
    """
    Sacredで実験を実行すると最初に呼び出される
    """
    # Setting the random seed throughout the modules
    config = deepcopy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # run the framework
    run(_run, config, _log)


def _get_config_from_yaml(file_name, subfolder):
    """
    指定されたYAMLからパラメータを読み込む
    """
    with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(file_name)), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(file_name, exc)
    return config_dict


if __name__ == "__main__":
    # Get the defaults from default.yaml
    # default.yamlからデフォルトのパラメータを読み込む
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    # 使用するアルゴリズムと環境設定を引数から読み込む
    parser = argparse.ArgumentParser()
    # アルゴリズム用
    parser.add_argument("--algo", default="qmix",
                        help="The algorithm to train the agent")
    # 環境設定用
    parser.add_argument("--env", default="sc2",
                        help="Which environment settings to load")
    # 引数を解析して取得
    args = parser.parse_args()

    # 対応するYAMLファイルから読み込み
    algo_config = _get_config_from_yaml(args.algo, "algorithms")
    env_config = _get_config_from_yaml(args.env, "environments")

    # 最初に読み込んだdefaultにアルゴリズムと環境設定のパラメーターを結合
    config_dict = {**config_dict, **env_config, **algo_config}

    print(yaml.dump(config_dict))

    # now add all the config to sacred
    # Scaredにパラメータを設定
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    # Sacredのデータベース
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    # 実験を実行
    ex.run()
