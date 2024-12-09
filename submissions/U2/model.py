import torch
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
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

"""
cpu memory를 사용함에 유의
"""
class ReplayMemory(object):

  def __init__(self, capacity):
    self.capacity = capacity
    self.index = mp.Manager().Value('i', 0)
    self.memory = mp.Manager().list()
    self.lock_memory = mp.Lock()
    self.size = mp.Manager().Value('i', 0)
    
    self.local_memory = deque(maxlen=self.capacity)

  def push(self, *args):
    with self.lock_memory:
      if len(self.memory) < self.capacity:
        self.memory.append(Transition(*args))
        self.size.value += 1
      else:
        # 버려지지 않도록 프로세스 수 조절
        self.memory[self.index.value] = Transition(*args)
        self.index.value = (self.index.value + 1) % self.capacity

  def sample(self, batch_size):
    return random.sample(self.local_memory, batch_size)
  
  def load(self):
    with self.lock_memory:
      # index로부터 size수만큼 local_memory에 push
      while self.size.value > 0:
        self.local_memory.append(self.memory[self.index.value])
        self.index.value = (self.index.value + 1) % self.capacity
        self.size.value -= 1
        
  def __len__(self):
    return self.local_memory.__len__()

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
    self._total_step_lock = mp.Lock()
    self._total_steps = mp.Manager().Value('i', 0)
    self._evaluate_loss = []
    self.lock_model = mp.Lock()   # MRSW lock으로 바꾸어야 함
    self.network = CNN(self._n_observation, self._n_actions).to(self.device)                    # 학습을 위한 네트워크
    self.shared_network = CNN(self._n_observation, self._n_actions).to("cpu").share_memory()    # 프로세스 간 공유 메모리, action sampling을 위함
    self.shared_network.load_state_dict(self.network.state_dict())
    self.target_network = CNN(self._n_observation, self._n_actions).to(self.device)             # 타겟 네트워크
    self.target_network.load_state_dict(self.network.state_dict())
    self.optimizer = optim.AdamW(self.network.parameters(), lr=self._lr, amsgrad=True)
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
    # Generating a random number for exploration vs exploitation
    sample = random.random()

    # Calculating the threshold - the more steps the less exploration we do
    with self._total_step_lock:
      total_step = self._total_steps.value
      self._total_steps.value += 1
    eps_threshold = self._eps_end + (self._eps_start - self._eps_end) * math.exp(-1. * total_step / self._eps_decay)
    
    '''
    Evaluation 단계에서만 cuda memory를 사용함에 유의
    '''
    if evaluation_phase:
      with torch.no_grad():
        return self.target_network(state).max(1).indices.view(1, 1)
    elif sample > eps_threshold:
      with self.lock_model:
        with torch.no_grad():
          return self.shared_network(state).max(1).indices.view(1, 1)
    else:
      return torch.tensor([[self._action_space.sample()]], device="cpu", dtype=torch.long)

  def train(self):
    if len(self._memory) < self._batch_size:
      return
    
    self.training_count += 1

    # Initializing our memory
    transitions = self._memory.sample(self._batch_size)

    # Initializing our batch
    batch = Transition(*zip(*transitions))

    '''
    Memory sampling 결과는 cpu memory
    cuda memory로 변환해야 함에 유의
    '''
    # Saving in a new tensor all the indices of the states that are non terminal
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)

    # Saving in a new tensor all the non final next states
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(self.device)

    state_batch = torch.cat(batch.state).to(self.device)
    action_batch = torch.cat(batch.action).to(self.device)
    reward_batch = torch.cat(batch.reward).to(self.device)

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
    self.optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(self.network.parameters(), 100)
    self.optimizer.step()

    self._evaluate_loss.append(loss.item())
      
    # Copying the weights to the shared network
    
    with self.lock_model:
      self.shared_network.load_state_dict(self.network.state_dict())
    
    return

  def copy_weights(self):
    self.target_network.load_state_dict(self.network.state_dict())

  def get_loss(self):
    return self._evaluate_loss

  def save_model(self, i):
    torch.save(self.target_network.state_dict(), f'model_weights_{i}.pth')

  def load_model(self, i):
    self.target_network.load_state_dict(torch.load(f'model_weights_{i}.pth', map_location=self.device))
