import datetime
import os
import pprint
import time
import threading
import torch as th
import logging as lg
from types import SimpleNamespace

import wandb
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath

from runners.episode_runner import EpisodeRunner
from controllers.basic_controller import BasicMAC
from learners.q_learner import QLearner
from components.episode_buffer import EpisodeBatch, ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log: lg.Logger):
    '''
    プログラム開始～終了まで
    '''
    # check args sanity
    # 引数をチェック
    _config = args_sanity_check(_config, _log)

    # 名前空間
    # args["###"]ではなく、args.### の形でパラメータへアクセスできる
    args = SimpleNamespace(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(
        args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs")
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    # 学習開始
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    # 学習が終わったので、プログラムを終了させる
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")


def run_sequential(args, logger: Logger):
    '''
    学習プロセスの一番のメイン関数
    '''

    # Init runner so we can get env info
    # 環境情報にアクセスするためのRunner
    runner = EpisodeRunner(args=args, logger=logger)

    # Set up schemes and groups here
    # 環境やエージェントの情報を取得
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    # Default/Base scheme
    # 環境やエージェントに関するスキーマを定義
    scheme = {
        # グローバル状態の次元数
        "state": {"vshape": env_info["state_shape"]},
        # 各エージェントの部分観測
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        # 行動
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        # 選択可能な行動
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        # 報酬
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    # groups: エージェント数分存在する情報（観測、行動）を管理するためのもの
    groups = {
        "agents": args.n_agents
    }
    # EpisodeBatchの初期化に使われる
    # OneHotベクトルで表す数を設定
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    pprint.pprint("scheme: {}".format(scheme))
    pprint.pprint("groups: {}".format(groups))
    pprint.pprint("preprocess: {}".format(preprocess))

    # 経験再生用バッファ
    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # Setup multiagent controller here
    # マルチエージェントを制御するコントローラー
    mac = BasicMAC(buffer.scheme, groups, args)

    # Give runner the scheme
    # 先ほど定義したスキーマとMACをRunnerに渡して初期化
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    # エージェントたち
    learner = QLearner(mac, buffer.scheme, logger, args)

    # グラボ用
    if args.use_cuda:
        learner.cuda()

    # チェックポイントモード用
    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info(
                "Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(
                timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # start training
    # ----------------- トレーニング開始！！！ -----------------

    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info(
        "Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # Run for a whole episode at a time
        # １つのエピソード全体を実行してバッチを取得
        episode_batch = runner.run(test_mode=False)
        # 経験再生バッファにエピソードを保存
        buffer.insert_episode_batch(episode_batch)

        # バッファに十分溜まったら
        if buffer.can_sample(args.batch_size):
            # バッチをサンプリング
            episode_sample: EpisodeBatch = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            # はみ出たタイムステップは切り捨てる
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)

            # バッチを用いてエージェントに学習させる
            learner.train(episode_sample, runner.t_env, episode)

        # Execute test runs once in a while
        # 定期的にテストでgreedyに実行する
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info(
                "t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(
                args.local_results_path, "models", args.unique_token, str(runner.t_env))
            # "results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += 1

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            wandb.log({"episode": episode})
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # ----------------- トレーニング終了 -----------------
    runner.close_env()
    logger.console_logger.info("Finished Training")


def evaluate_sequential(args, runner: EpisodeRunner):
    '''
    テストモードでエピソードを実行して評価する
    '''

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()


def args_sanity_check(config, _log):
    """
    引数が正常かどうか確認
    """

    # set CUDA flags
    # CUDAを使用するかどうか

    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning(
            "CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (
            config["test_nepisode"] // config["batch_size_run"]) * config["batch_size_run"]

    return config
