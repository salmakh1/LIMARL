from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
import torch as th
import random
import os

# Three states problem
# payoff_values = [[[4, 0, 0],
#                   [0, -2, -2],
#                   [0, -2, -2]],
#
#                  [[0, 0, 0],
#                   [0, 4, 0],
#                   [0, 0, 0]],
#
#                  [[-2, -2, 0],
#                   [-2, -2, 0],
#                   [0, 0, 4]]]


#
payoff_values = [[[4, 0, 0],
                  [0, -2, 0],
                  [0, 0, 0]],

                 [[-1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 0]],

                 [[0, 0, 0],
                  [0, -2, 0],
                  [0, 0, 4]]]

episode_limit = 10
partial_observability = True


class ThreeStepMatrixGame(MultiAgentEnv):
    def __init__(self, batch_size=None, **kwargs):
        # Define the agents
        self.n_agents = 2

        # Define the internal state
        self.steps = 0
        self.n_actions = len(payoff_values[0])
        self.episode_limit = episode_limit - 1
        self.state = random.randint(0, 2)  # inital random state

    def reset(self):
        """ Returns initial observations and states"""
        self.steps = 0

        # self.state = (self.state + 1) % len(payoff_values)
        # print(self.state)
        return self.get_obs(), self.get_state()

    def step(self, actions):
        """ Returns reward, terminated, info """
        # print('new step')
        reward = payoff_values[self.state][actions[0]][actions[1]]
        self.steps += 1
        self.state = (self.state + 1) % len(payoff_values)
        # print(self.state)
        terminated = False
        if self.steps >= self.episode_limit:
            # reward = payoff_values[self.state][actions[0]][actions[1]]
            terminated = True

        info = {}
        return reward, terminated, info

    def get_obs(self):
        """ Returns all agent observations in a list """
        # print('step', self.steps)

        one_hot = self.get_state()
        # print('state', one_hot)

        if partial_observability:
            one_hot[0] = 0
            one_hot[1] = 0
            # one_hot[2] = 0

        # print(one_hot)
        return [np.copy(one_hot[2:]) for _ in range(self.n_agents)]

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        return self.get_obs()[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return len(self.get_obs_agent(0))

    def get_state(self):
        one_hot = np.zeros(len(payoff_values))
        one_hot[self.state] = 1
        return one_hot

    def get_state_size(self):
        """ Returns the shape of the state"""
        return len(self.get_state())

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        return np.ones(self.n_actions)

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_stats(self):
        return None

    def render(self):
        raise NotImplementedError

    def close(self):
        pass

    def seed(self):
        raise NotImplementedError


def print_matrix_status(batch, mixer, mac_out, algorithm_name='any'):
    batch_size = batch.batch_size
    # print('batch_size', batch_size)

    matrix_size = len(payoff_values[0])

    num_states = len(payoff_values)
    # print('num_states', num_states)

    results = th.zeros((num_states, matrix_size, matrix_size))
    # print(results)

    # Precompute one-hot encodings
    one_hot_encodings = [np.eye(num_states)[s] for s in range(num_states)]
    # print('one_hot_encodings', one_hot_encodings)

    with th.no_grad():
        for s in range(results.shape[0]):

            # print(batch["state"].shape)
            # One-hot encoding for state 's'
            one_hot = one_hot_encodings[s]
            one_hot_tensor = th.tensor(one_hot, device=mac_out.device, dtype=th.float)

            state_tensor = batch["state"].to(mac_out.device)
            state_tensor = state_tensor.reshape(batch_size * episode_limit, -1)

            mac_out = mac_out.reshape(batch_size * episode_limit, 2, -1)
            # print("state_tensor ", state_tensor.shape)
            # print("mac_out ", mac_out.shape)

            indexes_of_state_s = []

            for i, state in enumerate(state_tensor):
                # Check if the state matches the one_hot_tensor
                is_equal = (state == one_hot_tensor)
                if th.all(is_equal):
                    # If equal, add the index to the list
                    indexes_of_state_s.append(i)

            # Convert the list to a tensor if needed
            indexes_of_state_s = th.tensor(indexes_of_state_s, device=mac_out.device)
            # print("indexes_of_state_s ", indexes_of_state_s.shape)
            for i in range(results.shape[1]):
                for j in range(results.shape[2]):
                    # Create actions tensor
                    actions = th.LongTensor([[[i], [j]]]).to(mac_out.device).repeat(
                        indexes_of_state_s.shape[0], 1, 1, 1)

                    # print("actions", actions.shape)

                    outputs = mac_out[indexes_of_state_s].unsqueeze(1)
                    # print("output", outputs.shape)

                    # print('HERE')
                    # print("ac", actions.shape)
                    # print("macc ", outputs.shape)
                    # Gather Q-values
                    # qvals = th.gather(mac_out[indexes_of_state_s, 0:1], dim=1, index=actions).squeeze(3)

                    # Perform gather operation safely
                    # print('outputs', outputs.shape)
                    qvals = th.gather(outputs, dim=-1, index=actions)
                    # print('qvals', qvals.shape)

                    qvals = qvals.squeeze(-1)  # Adjust squeezing for lower dimensions
                    # if qvals.dim() == 2:  # If qvals is 2D
                    #     qvals = qvals.unsqueeze(1)  # Add a singleton dimension for time

                    # Compute global Q-value
                    # print("qvals shape after adjustment:", qvals.shape)
                    # print("one_hot_tensor shape:", one_hot_tensor.shape)
                    global_q = mixer(qvals, one_hot_tensor.repeat(qvals.shape[0], 1)).mean()
                    results[s][i][j] = global_q.item()

        # Print results (optional, as they are already written to file)
        th.set_printoptions(1, sci_mode=False)
        print('payoffs')
        print(results)

    directory = "matrix_game"

    # Check if the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")
    else:
        print(f"Directory '{directory}' already exists.")

    file_path = directory + '/' +algorithm_name + "results.txt"

    # Write results to a file
    with open(file_path, "a") as f:
        # f.write("payoffs\n")
        # f.write(str(results.tolist()) + "\n")

        f.write("Payoff Values:\n")
        np_results = results.detach().cpu().numpy()  # Convert to numpy array for easy formatting
        np_results = np.round(np_results, 1)
        for layer in np_results:
            f.write("[\n")
            for row in layer:
                f.write(f"  {row.tolist()},\n")
            f.write("]\n\n")

        for s in range(results.shape[0]):
            f.write(f"state number {s}\n")
            one_hot = one_hot_encodings[s]
            one_hot_tensor = th.tensor(one_hot, device=mac_out.device, dtype=th.float)

            state_tensor = batch["state"].to(mac_out.device)
            state_tensor = state_tensor.reshape(batch_size * episode_limit, -1)
            mac_out = mac_out.reshape(batch_size * episode_limit, 2, -1)

            indexes_of_state_s = []
            for i, state in enumerate(state_tensor):
                is_equal = (state == one_hot_tensor)
                if th.all(is_equal):
                    indexes_of_state_s.append(i)

            indexes_of_state_s = th.tensor(indexes_of_state_s, device=mac_out.device)

            mac_out_values = mac_out[indexes_of_state_s].mean(dim=0).detach().cpu().numpy()
            mac_out_values = np.round(mac_out_values, 1)

            # f.write(f"mac_out shape: {mac_out[indexes_of_state_s].shape}\n")
            f.write(f"{mac_out_values.tolist()}\n")