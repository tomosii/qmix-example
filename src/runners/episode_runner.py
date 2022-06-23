# from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from controllers.basic_controller import BasicMAC

from smac.env import StarCraft2Env
from utils.logging import Logger


class EpisodeRunner:

    def __init__(self, args, logger: Logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        # 環境
        self.env = StarCraft2Env(**self.args.env_args)
        # 最大タイムステップ
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        # ログ用
        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac: BasicMAC):
        """
        RunnerにMACを設定する
        """
        # EpisodeBatchの引数を固定したものをここで作っておく（初期化のたびに再利用するため）
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        """
        環境の情報を取得（辞書型）
        """
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        """
        環境など諸々を初期化
        """
        # 新しいバッチを用意
        self.batch = self.new_batch()
        # 環境をリセット
        self.env.reset()
        # タイムステップを0に
        self.t = 0

    def run(self, test_mode=False):
        """
        1エピソードを実行
        """

        # 環境を初期化
        self.reset()

        terminated = False
        episode_return = 0

        # 隠れ状態を初期化
        self.mac.init_hidden(batch_size=self.batch_size)

        # ---------------- エピソード開始！！！ ----------------

        while not terminated:

            # 遷移前の情報を環境から取得
            pre_transition_data = {
                # グローバル状態
                "state": [self.env.get_state()],
                # 各エージェントの選択可能な行動
                "avail_actions": [self.env.get_avail_actions()],
                # エージェントの部分観測
                "obs": [self.env.get_obs()]
            }

            # バッチに遷移前の情報を追加
            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            # 現時点のバッチ（エピソードの最初から今までの遷移情報が含まれている）を渡して、行動を決定
            actions = self.mac.select_actions(
                self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # 行動を出力して、環境からフィードバックを得る
            # 報酬，エピソードが終了したか，環境情報
            reward, terminated, env_info = self.env.step(actions[0])

            # このエピソードの総収益
            episode_return += reward

            # 遷移後の情報
            post_transition_data = {
                # 選択した行動
                "actions": actions,
                # 獲得した報酬
                "reward": [(reward,)],
                # エピソード終了の原因が目的達成による時のみTrue（最大回数を超えて終了した場合はFalse）
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            # 遷移後の情報もバッチに追加
            self.batch.update(post_transition_data, ts=self.t)

            # タイムステップを進める
            self.t += 1

        # ---------------- エピソード終了 ----------------

        # 終端状態における情報を取得
        last_data = {
            # 終端のグローバル状態
            "state": [self.env.get_state()],
            # 選択可能な行動
            "avail_actions": [self.env.get_avail_actions()],
            # 部分観測
            "obs": [self.env.get_obs()]
        }

        # バッチに終端状態の情報を追加
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        # 終端状態における行動を決定（？）
        actions = self.mac.select_actions(
            self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        # バッチに最後の行動を追加
        self.batch.update({"actions": actions}, ts=self.t)

        # ------ ログをまとめる ------
        # 使われていないので空？
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        # ファイル名用
        log_prefix = "test_" if test_mode else ""

        # cur_statsにはenv_infoがコピーされる
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0)
                          for k in set(cur_stats) | set(env_info)})

        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        # 累計タイムステップを蓄積（テスト時以外）
        if not test_mode:
            self.t_env += self.t

        # エピソード総収益の履歴
        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            # テストの際
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            # 時々epsilonのログをとる
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat(
                    "epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        # バッチを返す
        return self.batch

    def _log(self, returns: list, stats: dict, prefix):
        """
        前回記録時から今までの総収益の平均・分散を記録
        """
        # 総収益の平均・分散
        self.logger.log_stat(prefix + "return_mean",
                             np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std",
                             np.std(returns), self.t_env)
        # 総収益の履歴をクリア
        returns.clear()

        # env_infoに含まれるデータの平均
        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean",
                                     v / stats["n_episodes"], self.t_env)

        # env_infoの履歴をクリア
        stats.clear()
