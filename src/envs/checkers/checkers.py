import copy
import logging

import gym
import numpy as np
# from gym import spaces
from gym.utils import seeding

# from ..utils.action_space import MultiAgentActionSpace
from ..utils.draw import draw_grid, fill_cell, draw_circle, write_cell_text
# from ..utils.observation_space import MultiAgentObservationSpace

logger = logging.getLogger(__name__)


class Checkers(gym.Env):
    """
    The map contains apples and lemons. The first player (red) is very sensitive and scores 10 for
    the team for an apple (green square) and −10 for a lemon (orange square). The second (blue), less sensitive
    player scores 1 for the team for an apple and −1 for a lemon. There is a wall of lemons between the
    players and the apples. Apples and lemons disappear when collected, and the environment resets
    when all apples are eaten. It is important that the sensitive agent eats the apples while the less sensitive
    agent should leave them to its team mate but clear the way by eating obstructing lemons.

    Reference Paper : Value-Decomposition Networks For Cooperative Multi-Agent Learning ( Section 4.2)
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, full_observable=False, step_cost=-0.01, max_steps=100, clock=False, debug=False, seed=None):
        self._grid_shape = (3, 8)
        self.n_agents = 2
        self.n_actions = len(ACTION_MEANING)
        self._max_steps = max_steps
        self._step_count = None
        self._step_cost = step_cost
        self.full_observable = full_observable
        self._add_clock = clock
        self.debug = debug
        self.episode_limit = max_steps
        self._episode_count = 0
        self.timeouts = 0

        # self.action_space = MultiAgentActionSpace(
        #     [spaces.Discrete(5) for _ in range(self.n_agents)])
        # self._obs_high = np.ones(2 + (3 * 3 * 5) + (1 if clock else 0))
        # self._obs_low = np.zeros(2 + (3 * 3 * 5) + (1 if clock else 0))
        # if self.full_observable:
        #     self._obs_high = np.tile(self._obs_high, self.n_agents)
        #     self._obs_low = np.tile(self._obs_low, self.n_agents)
        # self.observation_space = MultiAgentObservationSpace([spaces.Box(self._obs_low, self._obs_high)
        #                                                      for _ in range(self.n_agents)])

        self.init_agent_pos = {
            0: [0, self._grid_shape[1] - 2], 1: [2, self._grid_shape[1] - 2]}
        self.agent_reward = {0: {'lemon': -10, 'apple': 10},
                             1: {'lemon': -1, 'apple': 1}}

        self.agent_prev_pos = None
        self._base_grid = None
        self._full_obs = None
        self._agent_dones = None
        self.viewer = None
        self._food_count = None
        self._total_episode_reward = None
        self.steps_beyond_done = None
        self.seed()

    def reset(self):
        """
        環境を初期化
        エージェントの観測を返す
        """
        # グローバル状態を初期化
        self.__init_full_obs()

        # タイムステップをリセット
        self._step_count = 0
        # 報酬の総和
        self._total_episode_reward = [0 for _ in range(self.n_agents)]
        # 残っている果物の数
        self._food_count = {'lemon': ((self._grid_shape[1] - 2) // 2) * self._grid_shape[0],
                            'apple': ((self._grid_shape[1] - 2) // 2) * self._grid_shape[0]}
        # 終了したか
        self._agent_dones = [False for _ in range(self.n_agents)]
        # 最大ステップを超えたか
        self.steps_beyond_done = None

        if self.debug:
            logging.debug(
                "Started Episode {}".format(self._episode_count).center(
                    60, "*"
                )
            )

        return self.get_obs(), self.get_state()

    def step(self, agents_action):
        """
        行動を環境に出力してタイムステップを1つ進める

        -> [ 報酬, 終了フラグ, 追加情報(残り個数など) ]
        """
        # 人数分の行動が入力されているかチェック
        assert len(agents_action) == self.n_agents
        # タイムステップを進める
        self._step_count += 1

        if self.debug:
            logging.debug("Actions".center(60, "-"))

        # 時間経過によるマイナスの報酬
        rewards = [0 for _ in range(self.n_agents)]
        # 各エージェントが得た報酬の総和
        personal_total_rewards = {}

        # エージェントごと

        for agent_i, action in enumerate(agents_action):
            # 行動をとって位置を移動
            self.__update_agent_pos(agent_i, action)

            # 前回と位置が変わっていたら
            if self.agent_pos[agent_i] != self.agent_prev_pos[agent_i]:
                for food in ['lemon', 'apple']:
                    # 移動先に果物があるかを調べる
                    if PRE_IDS[food] in self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i][1]]:
                        # 果物をGETしたことによる報酬を獲得
                        rewards[agent_i] += self.agent_reward[agent_i][food]
                        # 果物の残数を更新
                        self._food_count[food] -= 1
                        break
                # エージェントの移動をグリッドに反映
                self.__update_agent_view(agent_i)

            # 報酬の総和を更新
            self._total_episode_reward[agent_i] += rewards[agent_i]
            personal_total_rewards.update({
                "agent{}_total_score".format(agent_i + 1): self._total_episode_reward[agent_i]
            })

        terminated = False
        reward = sum(rewards) + self._step_cost
        info = {
            'apples_left': self._food_count["apple"],
            'lemons_left': self._food_count["lemon"],
            **personal_total_rewards  # 結合
        }

        # 終了時
        if self._food_count['apple'] == 0:  # 全てのリンゴをGETした
            terminated = True
            # 全エージェントの終了フラグを立てる
            # for i in range(self.n_agents):
            #     self._agent_dones[i] = True
        elif self._step_count >= self._max_steps:  # 最大ステップ数を超えた
            terminated = True
            info["timeout"] = True
            self.timeouts += 1

        # 終了したにも関わらずstep()を呼ばれた際のエラー用
        if self.steps_beyond_done is None and terminated:
            self.steps_beyond_done = 0
        elif self.steps_beyond_done is not None:
            if self.steps_beyond_done == 0:
                logger.warning(
                    "You are calling 'step()' even though this environment has already returned all(dones) = True for "
                    "all agents. You should always call 'reset()' once you receive 'all(dones) = True' -- any further"
                    " steps are undefined behavior.")
            self.steps_beyond_done += 1

        if self.debug:
            logging.debug("Reward = {}, Personal rewards = {}".format(
                reward, rewards).center(60, "-"))

        if terminated:
            self._episode_count += 1

        # [報酬, 終了フラグ, 追加情報（残り個数）]
        # return self.get_agent_obs(), rewards, self._agent_dones, {'food_count': self._food_count}
        return reward, terminated, info

    def get_env_info(self):
        """
        環境のパラメータ
        """
        # env_info = super().get_env_info()
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": 2,
            "episode_limit": 100,
        }
        # env_info["agent_features"] = self.ally_state_attr_names
        # env_info["enemy_features"] = self.enemy_state_attr_names
        return env_info

    def get_obs_size(self):
        """Returns the size of the observation."""
        return 2 + (3 * 3 * 5)
        # own_feats = self.get_obs_own_feats_size()
        # move_feats = self.get_obs_move_feats_size()

        # n_enemies, n_enemy_feats = self.get_obs_enemy_feats_size()
        # n_allies, n_ally_feats = self.get_obs_ally_feats_size()

        # enemy_feats = n_enemies * n_enemy_feats
        # ally_feats = n_allies * n_ally_feats

        # return move_feats + enemy_feats + ally_feats + own_feats

    def get_state_size(self):
        """Returns the size of the global state."""
        return self.get_obs_size() * self.n_agents

        # if self.obs_instead_of_state:
        #     return self.get_obs_size() * self.n_agents

        # nf_al = 4 + self.shield_bits_ally + self.unit_type_bits
        # nf_en = 3 + self.shield_bits_enemy + self.unit_type_bits

        # enemy_state = self.n_enemies * nf_en
        # ally_state = self.n_agents * nf_al

        # size = enemy_state + ally_state

        # if self.state_last_action:
        #     size += self.n_agents * self.n_actions
        # if self.state_timestep_number:
        #     size += 1

        # return size

    def get_total_actions(self):
        """Returns the total number of actions an agent could ever take."""
        return 5

    def get_action_meanings(self, agent_i=None):
        if agent_i is not None:
            assert agent_i <= self.n_agents
            return [ACTION_MEANING[i] for i in range(self.action_space[agent_i].n)]
        else:
            return [[ACTION_MEANING[i] for i in range(ac.n)] for ac in self.action_space]

    def get_avail_actions(self):
        """
        全エージェントの選択可能な行動をリストで返す
        """
        avail_actions = []
        for agent_i in range(self.n_agents):
            avail_agent = [1] * self.n_actions
            avail_actions.append(avail_agent)
        return avail_actions

    def __draw_base_img(self):
        self._base_img = draw_grid(
            self._grid_shape[0], self._grid_shape[1], cell_size=CELL_SIZE, fill='white')
        for row in range(self._grid_shape[0]):
            for col in range(self._grid_shape[1]):
                if PRE_IDS['wall'] in self._full_obs[row][col]:
                    fill_cell(self._base_img, (row, col),
                              cell_size=CELL_SIZE, fill=WALL_COLOR, margin=0.05)
                elif PRE_IDS['lemon'] in self._full_obs[row][col]:
                    fill_cell(self._base_img, (row, col),
                              cell_size=CELL_SIZE, fill=LEMON_COLOR, margin=0.05)
                elif PRE_IDS['apple'] in self._full_obs[row][col]:
                    fill_cell(self._base_img, (row, col),
                              cell_size=CELL_SIZE, fill=APPLE_COLOR, margin=0.05)

    def __create_grid(self):
        """create grid and fill in lemon and apple locations. This grid doesn't fill agents location"""
        _grid = []
        for row in range(self._grid_shape[0]):
            if row % 2 == 0:
                _grid.append([PRE_IDS['apple'] if (c % 2 == 0) else PRE_IDS['lemon']
                              for c in range(self._grid_shape[1] - 2)] + [PRE_IDS['empty'], PRE_IDS['empty']])
            else:
                _grid.append([PRE_IDS['apple'] if (c % 2 != 0) else PRE_IDS['lemon']
                              for c in range(self._grid_shape[1] - 2)] + [PRE_IDS['empty'], PRE_IDS['empty']])

        return _grid

    def __init_full_obs(self):
        self.agent_pos = copy.copy(self.init_agent_pos)
        self.agent_prev_pos = copy.copy(self.init_agent_pos)
        self._full_obs = self.__create_grid()
        for agent_i in range(self.n_agents):
            self.__update_agent_view(agent_i)
        self.__draw_base_img()

    def get_obs(self):
        """
        全てのエージェントの観測を1つのリストで返す
        NOTE: 分散実行時はエージェントは自分自身の観測のみ用いるようにする
        """
        _obs = []
        for agent_i in range(self.n_agents):
            pos = self.agent_pos[agent_i]

            # add coordinates
            _agent_i_obs = [round(pos[0] / self._grid_shape[0], 2),
                            round(pos[1] / (self._grid_shape[1] - 1), 2)]

            # add 3 x3 mask around the agent current location and share neighbours
            # ( in practice: this information may not be so critical since the map never changes)
            _agent_i_neighbour = np.zeros((3, 3, 5))
            for r in range(pos[0] - 1, pos[0] + 2):
                for c in range(pos[1] - 1, pos[1] + 2):
                    if self.is_valid((r, c)):
                        item = [0, 0, 0, 0, 0]
                        if PRE_IDS['lemon'] in self._full_obs[r][c]:
                            item[ITEM_ONE_HOT_INDEX['lemon']] = 1
                        elif PRE_IDS['apple'] in self._full_obs[r][c]:
                            item[ITEM_ONE_HOT_INDEX['apple']] = 1
                        elif PRE_IDS['agent'] in self._full_obs[r][c]:
                            item[ITEM_ONE_HOT_INDEX[self._full_obs[r][c]]] = 1
                        elif PRE_IDS['wall'] in self._full_obs[r][c]:
                            item[ITEM_ONE_HOT_INDEX['wall']] = 1
                        _agent_i_neighbour[r -
                                           (pos[0] - 1)][c - (pos[1] - 1)] = item
            _agent_i_obs += _agent_i_neighbour.flatten().tolist()

            # adding time
            if self._add_clock:
                _agent_i_obs += [self._step_count / self._max_steps]
            _obs.append(_agent_i_obs)

            if self.debug:
                logging.debug("Obs Agent: {}".format(agent_i).center(60, "-"))
                # logging.debug(
                #     "Avail. actions {}".format(
                #         self.get_avail_agent_actions(agent_id)
                #     )
                # )
                # logging.debug("Move feats {}".format(move_feats))
                # logging.debug("Enemy feats {}".format(enemy_feats))
                # logging.debug("Ally feats {}".format(ally_feats))
                # logging.debug("Own feats {}".format(own_feats))
                logging.debug(_obs)

        if self.full_observable:
            _obs = np.array(_obs).flatten().tolist()
            _obs = [_obs for _ in range(self.n_agents)]
        return _obs

    def get_state(self):
        """
        グローバル状態を返す
        NOTE: この関数は分散実行時は用いないこと
        """
        # 各エージェントの観測を結合したものをグローバル状態とする
        obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
            np.float32
        )
        return obs_concat

        # if self.obs_instead_of_state:
        #     obs_concat = np.concatenate(self.get_obs(), axis=0).astype(
        #         np.float32
        #     )
        #     return obs_concat

        # state_dict = self.get_state_dict()

        # state = np.append(
        #     state_dict["allies"].flatten(), state_dict["enemies"].flatten()
        # )
        # if "last_action" in state_dict:
        #     state = np.append(state, state_dict["last_action"].flatten())
        # if "timestep" in state_dict:
        #     state = np.append(state, state_dict["timestep"])

        # state = state.astype(dtype=np.float32)

        # if self.debug:
        #     logging.debug("STATE".center(60, "-"))
        #     logging.debug("Ally state {}".format(state_dict["allies"]))
        #     logging.debug("Enemy state {}".format(state_dict["enemies"]))
        #     if self.state_last_action:
        #         logging.debug("Last actions {}".format(self.last_action))

        # return state

    def is_valid(self, pos):
        return (0 <= pos[0] < self._grid_shape[0]) and (0 <= pos[1] < self._grid_shape[1])

    def _has_no_agent(self, pos):
        return self.is_valid(pos) and (PRE_IDS['agent'] not in self._full_obs[pos[0]][pos[1]])

    def __update_agent_pos(self, agent_i, move):

        curr_pos = copy.copy(self.agent_pos[agent_i])
        if move == 0:  # down
            next_pos = [curr_pos[0] + 1, curr_pos[1]]
        elif move == 1:  # left
            next_pos = [curr_pos[0], curr_pos[1] - 1]
        elif move == 2:  # up
            next_pos = [curr_pos[0] - 1, curr_pos[1]]
        elif move == 3:  # right
            next_pos = [curr_pos[0], curr_pos[1] + 1]
        elif move == 4:  # no-op
            next_pos = None
        else:
            raise Exception('Action Not found!')

        self.agent_prev_pos[agent_i] = self.agent_pos[agent_i]
        if next_pos is not None and self._has_no_agent(next_pos):
            self.agent_pos[agent_i] = next_pos

    def __update_agent_view(self, agent_i):
        self._full_obs[self.agent_prev_pos[agent_i][0]
                       ][self.agent_prev_pos[agent_i][1]] = PRE_IDS['empty']
        self._full_obs[self.agent_pos[agent_i][0]][self.agent_pos[agent_i]
                                                   [1]] = PRE_IDS['agent'] + str(agent_i + 1)

    def render(self, mode='human'):
        for agent_i in range(self.n_agents):
            fill_cell(
                self._base_img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill='white', margin=0.05)
            fill_cell(
                self._base_img, self.agent_prev_pos[agent_i], cell_size=CELL_SIZE, fill='white', margin=0.05)
            draw_circle(
                self._base_img, self.agent_pos[agent_i], cell_size=CELL_SIZE, fill=AGENT_COLORS[agent_i])
            write_cell_text(self._base_img, text=str(agent_i + 1), pos=self.agent_pos[agent_i], cell_size=CELL_SIZE,
                            fill='white', margin=0.4)

        # adds a score board on top of the image
        # img = draw_score_board(self._base_img, score=self._total_episode_reward)
        # img = np.asarray(img)

        img = np.asarray(self._base_img)
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)
            return self.viewer.isopen

    def seed(self, n=None):
        self.np_random, seed = seeding.np_random(n)
        return [seed]

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


CELL_SIZE = 30

ACTION_MEANING = {
    0: "DOWN",
    1: "LEFT",
    2: "UP",
    3: "RIGHT",
    4: "NOOP",
}

OBSERVATION_MEANING = {
    0: 'empty',
    1: 'lemon',
    2: 'apple',
    3: 'agent',
    -1: 'wall'
}

# each pre-id should be unique and single char
PRE_IDS = {
    'agent': 'A',
    'wall': 'W',
    'empty': '0',
    'lemon': 'Y',  # yellow color
    'apple': 'R',  # red color
}

AGENT_COLORS = {
    0: 'red',
    1: 'blue'
}
ITEM_ONE_HOT_INDEX = {
    'lemon': 0,
    'apple': 1,
    'A1': 2,
    'A2': 3,
    'wall': 4,
}
WALL_COLOR = 'black'
LEMON_COLOR = 'yellow'
APPLE_COLOR = 'green'
