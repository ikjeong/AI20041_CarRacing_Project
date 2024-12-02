import torch
import torch
import torch.nn as nn
import torch.optim as optim
from collections import namedtuple, deque
import random
import math
import torch.multiprocessing as mp

class CNN(nn.Module):
  def __init__(self, in_channels, out_channels, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._n_features = 32 * 9 * 9

    self.conv = nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2),
        nn.ReLU(),
    )

    self.fc = nn.Sequential(
        nn.Linear(self._n_features, 256),
        nn.ReLU(),
        nn.Linear(256, out_channels),
    )


  def forward(self, x):
    x = self.conv(x)
    x = x.view((-1, self._n_features))
    x = self.fc(x)
    return x

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN:
  def __init__(self, action_space, batch_size=256, gamma=0.99, eps_start=0.9, eps_end=0.05, eps_decay=1000, lr=0.001):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._n_observation = 4
    self._n_actions = 5
    self._action_space = action_space
    self._batch_size = batch_size
    self._gamma = gamma
    self._eps_start = eps_start
    self._eps_end = eps_end
    self._eps_decay = eps_decay
    self._lr = lr
    self._total_steps = 0
    self._evaluate_loss = []
    self.lock_model = mp.Lock()   # MRSW lock으로 바꾸어야 함
    self.network = CNN(self._n_observation, self._n_actions).to(self.device).share_memory()
    self.target_network = CNN(self._n_observation, self._n_actions).to(self.device).share_memory()
    self.target_network.load_state_dict(self.network.state_dict())
    self.optimizer = optim.AdamW(self.network.parameters(), lr=self._lr, amsgrad=True)
    self.lock_memory = mp.Lock()
    self._shared_memory = mp.Manager().Queue()
    self._memory = ReplayMemory(10000)
    self.training_count = 0

  """
  This function is called during training & evaluation phase when the agent
  interact with the environment and needs to select an action.

  (1) Exploitation: This function feeds the neural network a state
  and then it selects the action with the highest Q-value.
  (2) Evaluation mode: This function feeds the neural network a state
  and then it selects the action with the highest Q'-value.
  (3) Exploration mode: It randomly selects an action through sampling

  Q -> network (policy)
  Q'-> target network (best policy)
  """
  def select_action(self, state, evaluation_phase=False):
    with self.lock_model:
      # Generating a random number for eploration vs exploitation
      sample = random.random()

      # Calculating the threshold - the more steps the less exploration we do
      eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * self._total_steps / self._eps_decay)
      self._total_steps += 1

      if evaluation_phase:
        with torch.no_grad():
          return self.target_network(state).max(1).indices.view(1, 1)
      elif sample > eps_threshold:
        with torch.no_grad():
          return self.network(state).max(1).indices.view(1, 1)
      else:
        return torch.tensor([[self._action_space.sample()]], device=self.device, dtype=torch.long)

  def copy_memory(self):
      # Move the shared memory to the memory
      with self.lock_memory:
          size = self._shared_memory.qsize()
          for i in range(size):
              item = self._shared_memory.get()
              self._memory.push(item[0].to(self.device), item[1].to(self.device), item[2].to(self.device), item[3].to(self.device))

  def train(self):

    if len(self._memory) < self._batch_size:
        return
    
    self.training_count += 1

    # Initializing our memory
    transitions = self._memory.sample(self._batch_size)

    # Initializing our batch
    batch = Transition(*zip(*transitions))

    # Saving in a new tensor all the indices of the states that are non terminal
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)

    # Saving in a new tensor all the non final next states
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Feeding our Q network the batch with states and then we gather the Q values of the selected actions
    state_action_values = self.network(state_batch).gather(1, action_batch)

    # We then, for every state in the batch that is NOT final, we pass it in the target network to get the Q'-values and choose the max one
    next_state_values = torch.zeros(self._batch_size, device=self.device)
    with torch.no_grad():
        next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values

    # Computing the expecting values with: reward + gamma * max(Q')
    expected_state_action_values = (next_state_values * self._gamma) + reward_batch

    # Defining our loss criterion
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Updating with back propagation
    with self.lock_model:
      self.optimizer.zero_grad()
      loss.backward()

      torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
      self.optimizer.step()

      self._evaluate_loss.append(loss.item())

    return

  def copy_weights(self):
    with self.lock_model:
      self.target_network.load_state_dict(self.network.state_dict())

  def get_loss(self):
    return self._evaluate_loss

  def save_model(self, i):
    torch.save(self.target_network.state_dict(), f'model_weights_{i}.pth')

  def load_model(self, i):
    with self.lock_model:
      self.target_network.load_state_dict(torch.load(f'model_weights_{i}.pth', map_location=self.device))
