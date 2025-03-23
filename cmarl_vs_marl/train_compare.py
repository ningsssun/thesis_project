import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import pandas as pd
import os

# 1. Environment definition
class GridWorldEnv:
    """
    A simple grid-world environment with multiple agents, obstacles, and collisions.
    
    Attributes:
        grid_size   : width & height of the grid
        n_agents    : number of agents
        obstacle_map: 2D array indicating obstacle locations (1 = obstacle, 0 = free)
        agent_positions: list of tuples representing each agent's x,y location
        goals       : list of tuples for each agent's goal (x,y)
        done        : boolean indicating if the episode ended
        step_count  : current timestep in this episode
        max_steps   : maximum timesteps allowed before termination
    """
    def __init__(self, grid_size=(10,10), n_agents=2, max_steps=100):
        self.width, self.height = grid_size
        self.n_agents = n_agents
        self.max_steps = max_steps
        
        # Simple obstacle map: randomly place 20% obstacles
        self.obstacle_map = np.zeros(grid_size, dtype=int)
        obstacle_prob = 0.2
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < obstacle_prob:
                    self.obstacle_map[x,y] = 1
        
        # Initialize agent positions
        self.agent_positions = []
        self.goals = []
        
        self.done = False
        self.step_count = 0
        
    def reset(self):
        """
        Resets the environment for a new episode:
        - Clears the done boolean
        - Places each agent at a valid free cell
        - Sets a distinct goal for each agent
        - Resets step counter
        - Returns a list of states for each agent
        """
        self.done = False
        self.step_count = 0
        self.agent_positions = []
        self.goals = []
        
        # Randomly sample distinct start positions and distinct goals
        free_cells = [(x,y) for x in range(self.width) 
                               for y in range(self.height) 
                                 if self.obstacle_map[x,y] == 0]
        random.shuffle(free_cells)
        
        # Place agents
        for i in range(self.n_agents):
            self.agent_positions.append(free_cells.pop())
        # Place goals
        for i in range(self.n_agents):
            self.goals.append(free_cells.pop())
        
        # Return initial observations
        return self._get_observations()
    
    def step(self, actions):
        """
        Each agent takes an action (0=up, 1=down, 2=left, 3=right, 4=no-op)
        - Check collisions or invalid moves
        - Update agent positions
        - Compute rewards
        - Return next_state, rewards, done, info
        """
        rewards = [0.0]*self.n_agents
        
        # 1- Apply actions
        next_positions = []
        for i, act in enumerate(actions):
            x, y = self.agent_positions[i]
            nx, ny = x, y
            
            if act == 0:   # up
                ny -= 1
            elif act == 1: # down
                ny += 1
            elif act == 2: # left
                nx -= 1
            elif act == 3: # right
                nx += 1
            else:
                pass       # no-op
            
            # Check boundaries
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                # invalid move - remain in place or big penalty
                nx, ny = x, y
            
            # Check obstacles
            if self.obstacle_map[nx,ny] == 1:
                # cannot move into obstacle
                nx, ny = x, y
            
            next_positions.append((nx, ny))
        
        # 2- Check collisions among agents
        # If two or more next_positions are same -> collision
        unique_positions = set(next_positions)
        if len(unique_positions) < self.n_agents:
            # At least one collision
            # Option A: immediate termination with big penalty
            self.done = True
            for i in range(self.n_agents):
                rewards[i] -= 100.0  # severe penalty
        else:
            # No direct collision
            self.agent_positions = next_positions
        
        # 3- Compute goal rewards
        # If agent i reaches its goal => +100
        for i, (ax, ay) in enumerate(self.agent_positions):
            if (ax, ay) == self.goals[i]:
                rewards[i] += 100.0
        # Check if all reached goals
        all_done = True
        for i in range(self.n_agents):
            if self.agent_positions[i] != self.goals[i]:
                all_done = False
                break
        if all_done:
            self.done = True
        
        # 4- Time penalty for each step
        for i in range(self.n_agents):
            rewards[i] -= 1.0
        
        # 5- Increment step_count, check max steps
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
        
        # Observations
        next_obs = self._get_observations()
        
        return next_obs, rewards, self.done, {}
    
    def _get_observations(self):
        """
        Return the state for each agent (global or partial)
        """
        obs = []
        for i in range(self.n_agents):
            # Flatten obstacle map, agent position, goal position
            obs_map = self.obstacle_map.flatten().astype(float)
            ax, ay = self.agent_positions[i]
            gx, gy = self.goals[i]
            extended = np.concatenate([
                obs_map,
                np.array([ax, ay, gx, gy], dtype=float)
            ])
            obs.append(extended)
        return obs

# 2. Deep Q-Network for each agent
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5):
        # output_dim=5 -> up,down,left,right,no-op
        super(DQNNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

# 3. Modified base agent class
class BaseQLearningAgent:
    def __init__(self, state_dim, action_dim=5, lr=1e-3, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay = 1e-4
        self.lr = lr
        
        self.q_network = DQNNetwork(state_dim, 128, action_dim)
        self.target_network = DQNNetwork(state_dim, 128, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        self.buffer = deque(maxlen=50000)
        self.batch_size = 64
        self.update_count = 0
        self.target_update_freq = 200

    def select_action(self, state, valid_actions):
        if np.random.rand() < self.eps:
            return np.random.choice(valid_actions)
        else:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                qvals = self.q_network(state_t)[0].cpu().numpy()
            masked_qvals = np.full_like(qvals, -1e9)
            for a in valid_actions:
                masked_qvals[a] = qvals[a]
            return np.argmax(masked_qvals)

    def store_transition(self, s, a, r, s_next, done, valid_next_actions):
        self.buffer.append((s, a, r, s_next, done, valid_next_actions))

    def train_step(self, use_constraints=True):
        if len(self.buffer) < self.batch_size:
            return
        
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, states_next, dones, valid_next_acts = zip(*batch)

        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards_t = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(states_next))
        dones_t = torch.FloatTensor(np.array(dones)).unsqueeze(1)
        
        # Current Q
        qvals = self.q_network(states_t)
        gathered_q = qvals.gather(1, actions_t)
        
        # Target Q
        with torch.no_grad():
            next_qvals = self.target_network(next_states_t)
        
        max_next_q = []
        for i in range(self.batch_size):
            if dones[i]:
                max_next_q.append(0.0)
            else:
                qrow = next_qvals[i].cpu().numpy()
                if use_constraints:
                    valid_acts = valid_next_acts[i]
                    masked_q = np.full(len(qrow), -1e9)
                    for a in valid_acts:
                        masked_q[a] = qrow[a]
                    max_next_q.append(np.max(masked_q))
                else:
                    max_next_q.append(np.max(qrow))
        
        max_next_q = torch.FloatTensor(max_next_q).unsqueeze(1)
        target = rewards_t + (1 - dones_t) * self.gamma * max_next_q
        
        loss = self.criterion(gathered_q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.eps > self.eps_min:
            self.eps -= self.eps_decay

# 4. CMARL Agent
class CMARLAgent(BaseQLearningAgent):
    def get_valid_actions(self, env, agent_index):
        x, y = env.agent_positions[agent_index]
        valid_acts = []
        
        for action in [0, 1, 2, 3, 4]:
            nx, ny = x, y
            if action == 0: ny -= 1
            elif action == 1: ny += 1
            elif action == 2: nx -= 1
            elif action == 3: nx += 1
            
            if (0 <= nx < env.width and 0 <= ny < env.height and
                env.obstacle_map[nx,ny] == 0 and
                (nx, ny) not in env.agent_positions):
                valid_acts.append(action)
        
        return valid_acts if valid_acts else [4]

    def train_step(self):
        super().train_step(use_constraints=True)

# 5. MARL Agent (no constraints)
class MARLAgent(BaseQLearningAgent):
    def get_valid_actions(self, env, agent_index):
        return [0, 1, 2, 3, 4]  # All actions allowed

    def train_step(self):
        super().train_step(use_constraints=False)

# 6. Training function for both algorithms
def train_agents(num_episodes=300, n_agents=2, grid_size=(10,10), 
                log_dir="results", agent_class=CMARLAgent, seed=42):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Initialize environment and agents
    env = GridWorldEnv(grid_size=grid_size, n_agents=n_agents, max_steps=100)
    state_dim = env.width * env.height + 4
    agents = [agent_class(state_dim=state_dim) for _ in range(n_agents)]
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Training metrics storage
    metrics = {
        'episode': [],
        'total_reward': [],
        'collision': [],
        'success': [],
        'steps': []
    }
    
    # Training loop
    for ep in range(num_episodes):
        states = env.reset()
        done = False
        ep_reward = np.zeros(n_agents)
        collision_happened = False
        step_count = 0
        
        while not done:
            # ... (Keep the same training loop logic from previous code) ...
            step_count += 1
        
        # Store metrics
        metrics['episode'].append(ep)
        metrics['total_reward'].append(np.sum(ep_reward))
        metrics['collision'].append(1 if collision_happened else 0)
        metrics['success'].append(1 if success else 0)
        metrics['steps'].append(step_count)
        
        # Save periodic checkpoints
        if (ep % 50 == 0) or (ep == num_episodes-1):
            pd.DataFrame(metrics).to_csv(os.path.join(log_dir, f'progress.csv'), index=False)
    
    # Save final results
    pd.DataFrame(metrics).to_csv(os.path.join(log_dir, 'final_results.csv'), index=False)
    return agents

# 7. Run experiments and print results
def run_experiments():
    seeds = [42, 43, 44]
    algorithms = {'CMARL': CMARLAgent, 'MARL': MARLAgent}
    
    for algo_name, algo_class in algorithms.items():
        for seed in seeds:
            print(f"Training {algo_name} with seed {seed}")
            log_dir = f"results/{algo_name}_seed_{seed}"
            train_agents(
                num_episodes=300,
                n_agents=2,
                grid_size=(10,10),
                log_dir=log_dir,
                agent_class=algo_class,
                seed=seed
            )

if __name__ == "__main__":
    run_experiments()
