# qmix-example

マルチエージェント深層強化学習アルゴリズム QMIX のシンプルな問題への適用

## ライブラリ

- 深層学習フレームワーク : PyTorch
- 実験管理ツール : Sacred

## 論文

[QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) (
オックスフォード大学, 2018)

## 参考

論文の著者が実装したレポジトリ [oxwhirl / **pymarl**](https://github.com/oxwhirl/pymarl)

OpenAI Gym 形式のマルチエージェント環境 [koulanurag / **ma-gym**](https://github.com/koulanurag/ma-gym)

## 使い方

`python3 src/main.py --algo qmix --env checkers`

`--algo`: 使用するアルゴリズムを指定

`--env`: 使用する環境を指定

### ハイパーパラメータ

config フォルダの YAML ファイル で管理

ベースは default.yaml

アルゴリズム特有のものは config/algorithms

環境特有のものは config/environments

## 全体の構造＆流れ

## main.py

Sacred(実験管理ツール) の `ex.run()` で `@ex.main my_main()` が呼び出される

run.py の `run()`

## run.py

`run()` でログまわりの設定

`run_sequential()` → 学習プロセスのメイン関数

諸々のクラスを初期化

最大ステップ数を超えるまで、エピソードごとのループ

========================================================

`EpisodeRunner.run()` でエピソードを実行

`ReplayBuffer`にバッチを保存した後、サンプリング

バッチを用いて `QLearner.train()` で学習

定期的にテストモード(greedy)で `EpisodeRunner.run()` を実行

========================================================

## runners/ EpisodeRunner

`run()` → 1 エピソード全体を実行してバッチを返す

## components/ ReplayBuffer

経験再生用メモリ（`EpisodeBatch`を継承）

## learners/ QLearner

ネットワークの学習処理を担当

Agent Network & Mixing Network をまとめている

`train()` → バッチデータをもとに学習

## controllers/ BasicMAC

マルチエージェントコントローラー（MAC）

Agent Network（`RNNAgent`）の入出力を制御

## modules/agents/ RNNAgent

RNN（GRU）を用いた Agent Network (`torch.nn`)

## modules/mixers/ QMixer

HyperNetwork を用いた Mixing Network (`torch.nn`)
