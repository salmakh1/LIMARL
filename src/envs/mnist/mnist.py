from typing import Optional
from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import torch as th
import random
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split
import numpy as np

class ImageClassificationGame(MultiAgentEnv):
    """
    A harder partially–observable multi-agent MNIST environment.

    • Each episode a single 28×28 MNIST digit is sampled.
    • The image is divided into *patch_split* vertical strips (≡ number of agents by default).
    • At time-step *t* agent *i* only sees strip (i+t) mod patch_split.
      Hence every agent observes **all** strips once per episode but in a different order.
      Good memory and latent-state inference are therefore required.
    • A small configurable Gaussian noise can be injected into the visible strip.
    • A single cooperative reward is provided **only at the final step** (dense rewards confuse
      the credit assignment and make the task too easy for value-factorisation baselines).

    The public interface is identical to the original environment so downstream code
    (QMIX, MAE, State-RVAE learners, etc.) stays unchanged.
    """

    def __init__(self,
                 n_agents: int = 4,
                 episode_limit: int = 8,
                 patch_split: Optional[int] = None,
                 noise_std: float = 0.0,
                 dataset_fraction: float = 0.20,
                 partial_observability: bool = True,
                 **kwargs):
        super().__init__()

        # Core parameters
        self.n_agents: int = n_agents
        self.episode_limit: int = episode_limit
        self.partial: bool = partial_observability
        self.noise_std: float = noise_std
        self.patch_split: int = patch_split or n_agents   # default: one strip per agent

        # === Dataset ===
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        full = MNIST(root='./data', train=True, download=True, transform=tfm)
        subset_len = int(len(full) * dataset_fraction)
        self.dataset, _ = random_split(full, [subset_len, len(full) - subset_len])

        # === Geometry ===
        self._crop_regions = self._make_crop_regions()

        # Episode state
        self.steps: Optional[int] = None
        self._image: Optional[th.Tensor] = None     # [1,28,28]
        self._label: Optional[int] = None

        self.reset()

    # ------------------------------------------------------------------ #
    #                           Helper functions                         #
    # ------------------------------------------------------------------ #
    def _make_crop_regions(self):
        """Return list of (x0,y0,x1,y1) crop boxes that tile the image."""
        H = W = 28
        w = W // self.patch_split
        boxes = []
        for k in range(self.patch_split):
            x0 = k * w
            x1 = W if k == self.patch_split - 1 else (k + 1) * w
            boxes.append((x0, 0, x1, H))
        return boxes

    def _apply_crop(self, img: th.Tensor, box):
        """Mask everything except the crop *box*."""
        x0, y0, x1, y1 = box
        masked = th.zeros_like(img)
        masked[:, y0:y1, x0:x1] = img[:, y0:y1, x0:x1]

        if self.noise_std > 0:
            masked = masked + self.noise_std * th.randn_like(masked)

        return masked

    # ------------------------------------------------------------------ #
    #                        Multi-agent API methods                     #
    # ------------------------------------------------------------------ #
    def reset(self):
        self.steps = 0
        idx = random.randrange(len(self.dataset))
        self._image, self._label = self.dataset[idx]
        self._image = self._image.view(1, 28, 28).float()  # [C,H,W]

        return self.get_obs(), self.get_state()

    def step(self, actions):
        """
        :param actions: list[int] of length n_agents (each ∈ {0,…,9})
        :returns: (reward, terminated, info)
        """
        self.steps += 1
        terminated = self.steps >= self.episode_limit

        reward = 0.0
        if terminated:                               # sparse final reward
            correct = [int(a == self._label) for a in actions]  # type: ignore
            reward = float(sum(correct)) / self.n_agents

        info = {"step": self.steps, "label": self._label}
        return reward, terminated, info

    # --------------- Observation helpers --------------- #
    def _region_for(self, agent_id: int):
        """
        Deterministic ‘rotating-strip’ schedule:
        agent i observes strip (i + t) mod patch_split at time-step t.
        """
        k = (agent_id + self.steps) % self.patch_split  # type: ignore
        return self._crop_regions[k]

    # --------------- Interface required by PyMARL --------------- #
    def get_obs(self):
        if not self.partial:
            return [self._image.flatten().numpy()] * self.n_agents  # type: ignore

        obs = [
            self._apply_crop(self._image, self._region_for(i)).flatten().numpy()  # type: ignore
            for i in range(self.n_agents)
        ]
        return obs

    def get_obs_agent(self, agent_id: int):
        return self.get_obs()[agent_id]

    def get_obs_size(self) -> int:
        return 28 * 28   # flattened length

    def get_state(self):
        """Centralised critic / mixer sees the full image."""
        return self._image.flatten().numpy()  # type: ignore

    def get_state_size(self) -> int:
        return 28 * 28

    def get_avail_actions(self):
        return [np.ones(10, dtype=np.int32)] * self.n_agents

    def get_avail_agent_actions(self, agent_id: int):
        return np.ones(10, dtype=np.int32)

    def get_total_actions(self) -> int:
        return 10

    def render(self):
        pass  # not implemented

    def close(self):
        pass

    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            th.manual_seed(seed)

    def get_stats(self):
        return None


# from envs.multiagentenv import MultiAgentEnv
# from utils.dict2namedtuple import convert
# import torch as th
# import random
# from torchvision import transforms
# from torchvision.datasets import MNIST
# from torch.utils.data import random_split
# import numpy as np
# # th._six = type("six", (), {"string_classes": (str, bytes)})
#
#
#
# class ImageClassificationGame(MultiAgentEnv):
#     def __init__(self, n_agents=2, episode_limit=10, partial_observability=True, dataset_fraction=0.2, **kwargs):
#         super().__init__()
#
#         self.n_agents = n_agents
#         self.episode_limit = episode_limit
#         self.partial_observability = partial_observability
#
#         # Prepare dataset
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#         full_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
#         subset_size = int(len(full_dataset) * dataset_fraction)
#         self.dataset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])
#
#         # Environment state
#         self.steps = 0
#         self.state = None
#         self.current_label = None
#
#         # Setup for agent-specific crops (patch-based partial observability)
#         self.crop_params = self.generate_crop_params()
#
#         self.reset()
#
#     def generate_crop_params(self):
#         crop_regions = []
#         height, width = 28, 28
#         patch_w = width // self.n_agents
#         for i in range(self.n_agents):
#             x_start = i * patch_w
#             x_end = (i + 1) * patch_w if i < self.n_agents - 1 else width
#             crop_regions.append((x_start, 0, x_end, height))
#         return crop_regions
#
#     def apply_crop(self, image, region):
#         x0, y0, x1, y1 = region
#         cropped = th.zeros_like(image)
#         cropped[:, y0:y1, x0:x1] = image[:, y0:y1, x0:x1]
#         return cropped
#
#     def reset(self):
#         self.steps = 0
#         index = random.randint(0, len(self.dataset) - 1)
#         self.state, self.current_label = self.dataset[index]
#         self.state = self.state.view(1, 28, 28).float()
#         return self.get_obs(), self.get_state()
#
#     def step(self, actions):
#         self.steps += 1
#         correct = [int(act == self.current_label) for act in actions]
#         reward = sum(correct) / self.n_agents
#         terminated = self.steps >= self.episode_limit
#         return reward, terminated, {}
#
#     def get_obs(self):
#         if self.partial_observability:
#             return [self.apply_crop(self.state, region).flatten().numpy() for region in self.crop_params]
#         return [self.state.flatten().numpy() for _ in range(self.n_agents)]
#
#     def get_obs_agent(self, agent_id):
#         return self.get_obs()[agent_id]
#
#     def get_obs_size(self):
#         return self.get_obs()[0].shape[0]
#
#     def get_state(self):
#         return self.state.flatten().numpy()
#
#     def get_state_size(self):
#         return self.state.numel()
#
#     def get_avail_actions(self):
#         return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]
#
#     def get_avail_agent_actions(self, agent_id):
#         return np.ones(10)
#
#     def get_total_actions(self):
#         return 10
#
#     def render(self):
#         raise NotImplementedError
#
#     def close(self):
#         pass
#
#     def seed(self, seed_value=None):
#         if seed_value is not None:
#             th.manual_seed(seed_value)
#             np.random.seed(seed_value)
#             random.seed(seed_value)
#
#     def get_stats(self):
#         return None



# from envs.multiagentenv import MultiAgentEnv
# from utils.dict2namedtuple import convert
# import torch as th
# import random
# from torchvision import transforms
# from torchvision.datasets import MNIST
# from torch.utils.data import random_split, Subset
# import numpy as np
# th._six = type("six", (), {"string_classes": (str, bytes)})
#
# #
# #
# class ImageClassificationGame(MultiAgentEnv):
#     def __init__(self, n_agents=2, episode_limit=10, partial_observability=True, dataset_fraction=0.2, **kwargs):
#         """
#         Multi-agent environment for collaborative image classification.
#
#         Args:
#             n_agents (int): Number of agents (default: 2).
#             episode_limit (int): Maximum steps per episode (default: 10).
#             partial_observability (bool): If True, agents receive distorted observations.
#             dataset_fraction (float): Fraction of the dataset to use for training (default: 0.1).
#         """
#         super().__init__()
#
#         self.hard = False
#
#         # Dataset preparation
#         transform = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Normalize((0.5,), (0.5,))
#         ])
#         full_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
#         subset_size = int(len(full_dataset) * dataset_fraction)
#         print("subset_size", subset_size)
#
#         self.dataset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])
#
#         # Environment parameters
#         self.n_agents = n_agents
#         self.episode_limit = episode_limit
#         self.partial_observability = partial_observability
#
#         # State variables
#         self.features = 28
#         self.steps = 0
#         self.state = None
#         self.observations = None
#         self.observations_ = None
#         self.current_label = None
#
#         # Noisy transformation for partial observability
#         # self.transform_noisy = transforms.Compose([
#         #     transforms.RandomRotation(15),
#         #     transforms.RandomAffine(degrees=0, translate=(0.5, 0.5))
#         #     # transforms.GaussianBlur(3, sigma=(0.1, 2.0))
#         # ])
#         self.reset()  # Ensure the state is initialized
#
#     def mask_dynamic_crop(self, tensor):
#         """
#         Dynamically masks uncropped regions based on actual crop bounds.
#
#         Args:
#             tensor (Tensor): Input image tensor (B, C, H, W).
#
#         Returns:
#             Tensor: Image tensor with uncropped regions masked.
#         """
#         C, H, W = tensor.shape
#
#         # Randomly determine crop bounds
#         crop_x = random.randint(0, W // 2)
#         crop_y = random.randint(0, H // 2)
#         # crop_width = random.randint(W // 5, W // 2)  # Wider range
#         # crop_height = random.randint(H // 5, H // 2)
#         crop_width = random.randint(W // 8, W // 4)
#         crop_height = random.randint(H // 8, H // 4)
#
#         x2 = min(crop_x + crop_width, W)
#         y2 = min(crop_y + crop_height, H)
#
#         # Create a black mask and overlay the cropped area
#         mask = th.zeros_like(tensor)
#         mask[:, crop_y:y2, crop_x:x2] = tensor[:, crop_y:y2, crop_x:x2]
#
#         # Debugging: Print crop bounds
#         return mask
#
#     def reset(self):
#         """Resets the environment and initializes a new episode."""
#         self.steps = 0
#
#         # Sample a random image from the dataset
#         index = random.randint(0, len(self.dataset) - 1)
#         self.state, self.current_label = self.dataset[index]
#
#         # Ensure state is flattened
#         self.state = self.state.view(-1).float()
#
#         # Generate observations
#         # observation_ = self.transform_noisy(self.state.view(1, 28, 28))
#         # observation_ = self.transform_noisy(self.state.view(1, 28, 28))
#         self.observations_ = [self.mask_dynamic_crop(self.state.view(1, 28, 28)) for _ in range(self.n_agents)]
#         self.observations = [obs.view(-1).float() for obs in self.observations_]
#
#         return self.get_obs(), self.get_state()
#
#     def step(self, actions):
#         """
#         Executes a step in the environment.
#
#         Args:
#             actions (list): Class predictions from all agents.
#
#         Returns:
#             tuple: reward (int), terminated (bool), info (dict).
#         """
#
#         self.steps += 1
#
#         # Reward logic
#         if self.hard:
#             if len(set(actions)) == 1 and actions[0] == self.current_label:
#                 reward = 1
#             else:
#                 reward = 0
#         else:
#             reward = (float(actions[0] == self.current_label) + float(actions[1] == self.current_label))/2
#
#         terminated = self.steps >= self.episode_limit
#         # print("self.state is ", self.state)
#
#         # print('at step', self.steps-1, 'label', self.current_label, 'actions', actions)
#         info = {}
#         return reward, terminated, info
#
#     def get_obs(self):
#         """Returns observations for all agents."""
#         if self.partial_observability:
#             # observation_ = self.transform_noisy(self.state.view(1, 28, 28))
#             # observation_ = self.transform_noisy(self.state.view(1, 28, 28))
#             # print(self.state.shape)
#             observation_list = [self.mask_dynamic_crop(self.state.view(1, 28, 28)) for _ in range(self.n_agents)]
#             return [obs.flatten().numpy() for obs in observation_list]
#         return [self.state.flatten().numpy() for _ in range(self.n_agents)]
#
#     def get_obs_agent(self, agent_id):
#         """Returns the observation for a specific agent."""
#         if agent_id < 0 or agent_id >= self.n_agents:
#             raise ValueError(f"Invalid agent_id: {agent_id}")
#         return self.get_obs()[agent_id]
#
#     def get_obs_size(self):
#         """Returns the shape of an observation."""
#         return self.state.shape[0]
#
#     # def get_state(self):
#     #     """Returns the current global state."""
#     #     return self.state
#
#     def get_state(self):
#         """Returns the current global state."""
#         state = self.state.flatten()
#         return state.numpy()  # Convert to NumPy array
#
#     def get_state_size(self):
#         """Returns the shape of the state."""
#         # if self.state is None:
#         #     raise ValueError("State is not initialized. Call reset() before accessing state size.")
#         return self.state.shape[0]
#
#     def get_avail_actions(self):
#         """Returns available actions for all agents."""
#         return [self.get_avail_agent_actions(agent_id) for agent_id in range(self.n_agents)]
#
#     def get_avail_agent_actions(self, agent_id):
#         """Returns available actions for a specific agent."""
#         if agent_id < 0 or agent_id >= self.n_agents:
#             raise ValueError(f"Invalid agent_id: {agent_id}")
#         return np.ones(10)
#
#     def get_total_actions(self):
#         """Returns the total number of actions available."""
#         return 10  # Number of classes in MNIST
#
#     def render(self):
#         """Renders the environment (optional, not implemented)."""
#         raise NotImplementedError
#
#     def close(self):
#         """Cleans up resources (optional)."""
#         pass
#
#     def seed(self, seed_value=None):
#         """Sets the random seed for reproducibility (optional)."""
#         if seed_value is not None:
#             th.manual_seed(seed_value)
#             np.random.seed(seed_value)
#             random.seed(seed_value)
#
#     def get_stats(self):
#         return None
#




#
# from envs.multiagentenv import MultiAgentEnv
# from utils.dict2namedtuple import convert
# import torch as th
# import random
# import numpy as np
# from torchvision import transforms
# from torchvision.datasets import MNIST
#
# class ImageClassificationGame(MultiAgentEnv):
#     # share one dataset across all envs
#     _shared_dataset = None
#
#     def __init__(
#         self,
#         n_agents=2,
#         episode_limit=10,
#         partial_observability=True,
#         dataset_fraction=0.2,
#         noise_std=0.0,
#         step_penalty= -0.01,
#         **kwargs
#     ):
#         super().__init__()
#         self.n_agents = n_agents
#         self.episode_limit = episode_limit
#         self.partial_observability = partial_observability
#         self.noise_std = noise_std
#         self.step_penalty = step_penalty
#
#         # prepare shared dataset once
#         if ImageClassificationGame._shared_dataset is None:
#             transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.5,), (0.5,))
#             ])
#             full = MNIST(root='./data', train=True, download=True, transform=transform)
#             subset_size = int(len(full) * dataset_fraction)
#             ImageClassificationGame._shared_dataset, _ = th.utils.data.random_split(
#                 full, [subset_size, len(full) - subset_size]
#             )
#         self.dataset = ImageClassificationGame._shared_dataset
#
#         # placeholders
#         self.steps = 0
#         self.state = None            # (1,28,28) tensor
#         self.current_label = None
#         self.crop_regions = None     # will be reset each episode
#
#         # initialize
#         self.reset()
#
#     def _sample_crop_regions(self):
#         """Randomly partition the 28×28 image into n_agents vertical strips."""
#         width = 28
#         # choose random breakpoints
#         splits = sorted(random.sample(range(1, width), self.n_agents - 1))
#         edges = [0] + splits + [width]
#         return [(edges[i], 0, edges[i+1], 28) for i in range(self.n_agents)]
#
#     def apply_crop(self, image, region):
#         x0, y0, x1, y1 = region
#         cropped = th.zeros_like(image)
#         cropped[:, y0:y1, x0:x1] = image[:, y0:y1, x0:x1]
#         if self.noise_std > 0:
#             cropped += th.randn_like(cropped) * self.noise_std
#         return cropped
#
#     def reset(self):
#         # reset counters & randomize crops
#         self.steps = 0
#         self.crop_regions = self._sample_crop_regions()
#
#         # sample a new image & label
#         idx = random.randrange(len(self.dataset))
#         img, lbl = self.dataset[idx]
#         self.state = img.view(1, 28, 28).float()
#         self.current_label = int(lbl)
#
#         return self.get_obs(), self.get_state()
#
#     def step(self, actions):
#         """
#         actions: list of integers (one per agent)
#         returns: reward (float), terminated (bool), info (dict)
#         """
#         self.steps += 1
#
#         # majority‐vote at final step, else just step penalty
#         if self.steps < self.episode_limit:
#             reward = self.step_penalty
#         else:
#             # count how many agents guessed correctly
#             correct_votes = sum(int(a == self.current_label) for a in actions)
#             # normalized majority reward in [–1, +1]
#             vote_reward = (correct_votes / self.n_agents) * 2 - 1
#             reward = vote_reward
#
#         done = self.steps >= self.episode_limit
#         info = {
#             "true_label": self.current_label,
#             "step": self.steps
#         }
#         return reward, done, info
#
#     def get_obs(self):
#         """Return a list of length n_agents, each a flattened numpy array."""
#         obs = []
#         for region in self.crop_regions:
#             patch = (self.apply_crop(self.state, region)
#                      if self.partial_observability else self.state)
#             obs.append(patch.flatten().numpy())
#         return obs
#
#     def get_obs_agent(self, agent_id):
#         return self.get_obs()[agent_id]
#
#     def get_obs_size(self):
#         return self.get_obs()[0].shape[0]
#
#     def get_state(self):
#         return self.state.flatten().numpy()
#
#     def get_state_size(self):
#         return int(self.state.numel())
#
#     def get_avail_actions(self):
#         # every agent always has all 10 classes available
#         return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]
#
#     def get_avail_agent_actions(self, agent_id):
#         return np.ones(10, dtype=np.int32)
#
#     def get_total_actions(self):
#         return 10
#
#     def render(self):
#         # optional: visualize the current state and crop regions
#         raise NotImplementedError
#
#     def close(self):
#         pass
#
#     def seed(self, seed_value=None):
#         if seed_value is not None:
#             random.seed(seed_value)
#             np.random.seed(seed_value)
#             th.manual_seed(seed_value)
#
#     def get_stats(self):
#         return {
#             "steps": self.steps,
#             "current_label": self.current_label
#         }
