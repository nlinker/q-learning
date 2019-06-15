# environment
import argparse
import datetime
import dateutil
import numpy as np
import matplotlib.pyplot as plt
import pickle

from time import sleep
from typing import Tuple, Dict, Any, List, Optional
from collections import defaultdict

from .stats import Stats
from .discrete import Discrete
from .plotting import build_stats_plot

# ------------------------------------------------------------


class State:
    def __init__(self, view: Tuple[int, ...]):
        self.view = view

    def __hash__(self) -> int:
        return hash(self.view)

    def __eq__(self, other):
        return self.view == other.view


# ------------------------------------------------------------
# noinspection PyShadowingNames
class Environment:
    available_actions = [(0, 0), (-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    action_space = Discrete(len(available_actions))

    def __init__(self, n: int, seed: int = None):
        self.random_seed = seed
        self.n = n
        self.enemies: List[Enemy] = []
        self.player: Optional[Player] = None
        self.world_map = list([[0 for _ in range(self.n)] for _ in range(self.n)])
        self.reset()

    def reset(self) -> State:
        if self.random_seed is not None:
            Environment.action_space.seed(self.random_seed)
        n = self.n
        # do the same as
        for i in range(n):
            for j in range(n):
                self.world_map[i][j] = 0
        self.player = Player(n, 1, 1)
        # self.enemies = [Enemy(n, n - 1, n - 1, self.player)]
        self.enemies = [Enemy(n, n - 2, n - 1, self.player), Enemy(n, n - 1, n - 2, self.player)]
        for e in self.enemies:
            self.world_map[e.x][e.y] = 2
        self.world_map[self.player.x][self.player.y] = 1
        return self.extract_state()

    def step(self, action) -> (State, float, bool, str):
        """ one simulation step, the new position of the agent and the enemy """
        for e in self.enemies:
            self.world_map[e.x][e.y] = 0
        self.world_map[self.player.x][self.player.y] = 0
        # move everything
        self.player.move(action)
        for e in self.enemies:
            e.move()
        # update world_map
        self.world_map[self.player.x][self.player.y] = 1
        for e in self.enemies:
            self.world_map[e.x][e.y] = 2
        return self.extract_state(), self.get_reward(), self.is_finished(), ""

    def close(self):
        pass

    def extract_state(self) -> State:
        features = [self.player.x, self.player.y]
        for e in self.enemies:
            features.append(e.x)
            features.append(e.y)
        return State(tuple(features))

    def render(self):
        """ visualize the environment """
        print("\033[F" + ("\033[A" * self.n))

        def conv(x):
            if x == 0:
                return '.'
            elif x == 1:
                return 'A'
            else:
                return 'E'

        for e in self.world_map:
            print(' '.join(map(conv, e)))

    def is_finished(self):
        px, py = self.player.x, self.player.y
        # if any of the enemies caught the player, then the game is finished
        for e in self.enemies:
            if px == e.x and py == e.y:
                return True
        return False

    def get_reward(self):
        return -1 if self.is_finished() else 1


# ------------------------------------------------------------
class Player:
    def __init__(self, n, x: int, y: int):
        self.n = n
        self.x = x
        self.y = y

    # noinspection PyShadowingNames
    def move(self, action: int):
        dx, dy = Environment.available_actions[action]
        x = self.x + dx
        y = self.y + dy
        if (0 <= x < self.n) and (0 <= y < self.n):
            self.x = x
            self.y = y


# ------------------------------------------------------------
class Enemy:

    def __init__(self, n: int, x: int, y: int, p: Player):
        self.n = n
        self.x = x
        self.y = y
        self.p = p

    # noinspection PyShadowingNames
    def move(self):
        # make a step towards the player, enemy can move only in 4 directions
        dx, dy, do_move = self.direction(self.p.x, self.p.y)
        if do_move:
            x = self.x + dx
            y = self.y + dy
            if (0 <= x < self.n) and (0 <= y < self.n):
                self.x = x
                self.y = y
            return
        # choose random move towards the (dx, dy)
        is_valid = False
        while not is_valid:
            x_or_y = Environment.action_space.np_random.randint(2)
            if x_or_y == 0:
                (dx2, dy2) = (dx, 0)
            else:
                (dx2, dy2) = (0, dy)
            x = self.x + dx2
            y = self.y + dy2
            is_valid = ((0 <= x < self.n) and (0 <= y < self.n))
            if is_valid:
                self.x = x
                self.y = y

    def direction(self, dst_x: int, dst_y: int) -> (int, int, bool):
        src_x = self.x
        src_y = self.y
        # x is actually the first index
        # y is actually the second index
        if (src_x == dst_x) and (src_y > dst_y):
            return 0, -1, True  # left
        elif (src_x == dst_x) and (src_y < dst_y):
            return 0, +1, True  # right
        elif (src_y == dst_y) and (src_x < dst_x):
            return +1, 0, True  # down
        elif (src_y == dst_y) and (src_x > dst_x):
            return -1, 0, True  # up
        elif (src_x < dst_x) and (src_y < dst_y):
            return +1, +1, False  # stay still
        elif (src_x < dst_x) and (src_y > dst_y):
            return +1, -1, False  # stay still
        elif (src_x > dst_x) and (src_y < dst_y):
            return -1, +1, False  # stay still
        elif (src_x > dst_x) and (src_y > dst_y):
            return -1, -1, False  # stay still
        else:
            return 0, 0, False


# ------------------------------------------------------------
class QModel:
    """ Q-learning model """

    # noinspection PyShadowingNames
    def __init__(self, env: Environment):
        self.gamma = 0.95
        self.alpha = 0.05
        self.epsilon = 0.8
        self.decay_factor = 0.999
        self.state = {}
        self.env = env
        self.prev_state = env.extract_state()
        self.curr_state = env.extract_state()

    # noinspection PyMethodMayBeStatic,PyShadowingNames
    def make_epsilon_greedy_policy(self, q_func, epsilon, action_count):
        def policy_fn(observation):
            actions = np.ones(action_count, dtype=float) * epsilon / action_count
            best_action = np.argmax(q_func[observation])
            actions[best_action] += (1.0 - epsilon)
            return actions

        return policy_fn

    def play(self, q_fun: Dict[State, Any], verbose: bool = False) -> int:
        self.env.reset()
        print("\n" * self.env.n)
        t = 0
        for t in range(1000):
            arr = q_fun[self.env.extract_state()]
            # if np.sum(arr) == 0:
            #     action = env.action_space.sample()
            # else:
            action = np.argmax(arr)
            _, _, done, _ = self.env.step(action)
            if verbose:
                self.env.render()
                sleep(0.2)
            if done:
                print("Episode finished after {} steps".format(t + 1))
                break
        return t

    def default_zeros(self):
        # this was passed to a separate function instead of
        # defaultdict(lambda: np.zeros(self.env.action_space.n))
        # to make pickle work https://stackoverflow.com/a/16439720/5066426
        return np.zeros(self.env.action_space.n)

    # noinspection PyShadowingNames
    def q_learning(self, num_episodes: int) -> Tuple[Dict[State, Any], Stats]:
        gamma = self.gamma
        alpha = self.alpha
        epsilon = self.epsilon
        decay_factor = self.decay_factor
        q_func = defaultdict(self.default_zeros)

        stats = Stats(
            episode_lengths=np.zeros(num_episodes),
            episode_rewards=np.zeros(num_episodes))

        for i_episode in range(num_episodes):
            epsilon *= decay_factor
            s = self.env.reset()
            for t in range(2000):
                if np.random.random() < epsilon or np.sum(q_func[s]) == 0:
                    a = np.random.randint(0, self.env.action_space.n)
                else:
                    a = np.argmax(q_func[s])
                s_next, r, done, _ = self.env.step(a)
                # Update statistics
                stats.episode_rewards[i_episode] += r
                stats.episode_lengths[i_episode] = t

                q_func[s][a] += alpha * (r + gamma * np.max(q_func[s_next]) - q_func[s][a])
                s = s_next
                if done:
                    break
            if (i_episode + 1) % 500 == 0:
                print("\rEpisode {}/{}.".format(i_episode + 1, num_episodes), end="")
        return q_func, stats


def run():
    """Main entry point for command line invocation."""

    parser = argparse.ArgumentParser(description='Perform the training and running Q-learning model')
    parser.add_argument('-e', '--eval-only', dest='eval_only', action='store_true',
                        help='Evaluate only, without training the model (uses `current-q_func` saved before)')
    parser.set_defaults(feature=False)
    eval_only: bool = parser.parse_args().eval_only

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    # file_prefix = now.strftime('%Y%m%d_%H_%M_%S')
    file_prefix = 'current'

    env = Environment(10)
    model = QModel(env)

    if not eval_only:
        # train the model
        q_func, stats = model.q_learning(100_000)
        with open(file_prefix + '-qfunc.pkl', 'wb') as file:
            pickle.dump(q_func, file)
        # save the graph
        _fig = build_stats_plot(stats)
        plt.savefig(file_prefix + '-graph.png')

    # play with the model trained
    with open(file_prefix + '-qfunc.pkl', 'rb') as file:
        q_func = pickle.load(file)
    model.play(q_func, verbose=True)
