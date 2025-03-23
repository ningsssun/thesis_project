#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time
import pandas as pd

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
        
        # Create a random obstacle map (20% chance each cell is an obstacle).
        self.obstacle_map = np.zeros(grid_size, dtype=int)
        obstacle_prob = 0.2
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < obstacle_prob:
                    self.obstacle_map[x,y] = 1
        
        self.agent_positions = []
        self.goals = []
        
        self.done = False
        self.step_count = 0
        
    def reset(self):
        self.done = False
        self.step_count = 0
        self.agent_positions = []
        self.goals = []
        
        # Collect all free cells
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
        
        return self._get_observations()
    
    def step(self, actions):
        """
        Takes a list of actions (one per agent). 0=up,1=down,2=left,3=right,4=no-op.
        Returns (next_states, rewards, done, info)
        """
        rewards = [0.0]*self.n_agents
        
        # Move each agent
        next_positions = []
        for i, act in enumerate(actions):
            x, y = self.agent_positions[i]
            nx, ny = x, y
            if act == 0:  # up
                ny -= 1
            elif act == 1:  # down
                ny += 1
            elif act == 2:  # left
                nx -= 1
            elif act == 3:  # right
                nx += 1
            else:
                pass  # no-op

            # Check boundary
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                nx, ny = x, y  # ignore move

            # Check obstacle
            if self.obstacle_map[nx, ny] == 1:
                nx, ny = x, y  # can't move into obstacle
            
            next_positions.append((nx, ny))
        
        # Check collisions
        unique_positions = set(next_positions)
        if len(unique_positions) < self.n_agents:
            # collision => end episode with penalty
            self.done = True
            for i in range(self.n_agents):
                rewards[i] -= 100.0
        else:
            self.agent_positions = next_positions
        
        # Check if each agent reached goal
        for i, (ax, ay) in enumerate(self.agent_positions):
            if (ax, ay) == self.goals[i]:
                rewards[i] += 100.0
        
        # If all reached, done
        all_done = True
        for i in range(self.n_agents):
            if self.agent_positions[i] != self.goals[i]:
                all_done = False
                break
        if all_done:
            self.done = True
        
        # small step penalty
        for i in range(self.n_agents):
            rewards[i] -= 1.0
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
        
        next_obs = self._get_observations()
        return next_obs, rewards, self.done, {}
    
    def _get_observations(self):
        """
        For each agent, flatten obstacle map + (agent_x, agent_y, goal_x, goal_y).
        """
        obs = []
        obs_map = self.obstacle_map.flatten().astype(float)  # shape=(width*height,)
        for i in range(self.n_agents):
            ax, ay = self.agent_positions[i]
            gx, gy = self.goals[i]
            extra = np.array([ax, ay, gx, gy], dtype=float)
            # concat => shape=(width*height + 4,)
            extended = np.concatenate([obs_map, extra])
            obs.append(extended)
        return obs


##############################################################################
# 2) DQNNetwork
##############################################################################
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


##############################################################################
# 3) AGENT with REPLAY BUFFER
##############################################################################
class ConstrainedQLearningAgent:
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
        
        self.buffer = deque(maxlen=10000)
        self.batch_size = 32
        self.update_count = 0
        self.target_update_freq = 200
    
    def select_action(self, state):
        """
        Epsilon-greedy action selection. 'state' is a 1D numpy array.
        """
        if random.random() < self.eps:
            return random.randint(0, self.action_dim - 1)
        else:
            with torch.no_grad():
                # convert to 2D (batch=1)
                # *** HERE is where we'd get the warning if we pass a list of arrays. ***
                # We must pass one array. Let's do:
                state_t = torch.FloatTensor(state).unsqueeze(0)  # shape=(1,state_dim)
                qvals = self.q_network(state_t)[0].cpu().numpy()
            return np.argmax(qvals)
    
    def store_transition(self, s, a, r, s_next, done):
        self.buffer.append((s, a, r, s_next, done))
    
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # states is now a tuple of numpy arrays, each shape=(state_dim,)
        # => we do np.array(...) on them
        states_array = np.array(states)          # shape=(batch_size, state_dim)
        next_states_array = np.array(next_states)# shape=(batch_size, state_dim)
        
        states_t = torch.FloatTensor(states_array)      # no more warnings
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states_array)
        dones_t = torch.FloatTensor([1.0 if d else 0.0 for d in dones]).unsqueeze(1)
        
        # current Q
        qvals = self.q_network(states_t)
        chosen_q = qvals.gather(1, actions_t)
        
        # next Q
        with torch.no_grad():
            target_qvals = self.target_network(next_states_t)
            max_next_q = target_qvals.max(dim=1, keepdim=True)[0]
        # if done => 0
        target = rewards_t + self.gamma*(1 - dones_t)*max_next_q
        
        loss = self.criterion(chosen_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        if self.eps > self.eps_min:
            self.eps -= self.eps_decay


##############################################################################
# 4) MAIN TRAINING LOOP
##############################################################################
def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    env = GridWorldEnv(grid_size=(10,10), n_agents=2, max_steps=100)

    # We'll create a separate agent for each of the n_agents (for demonstration).
    n_agents = env.n_agents
    # Each observation: flatten obstacle map + 4 => 10*10 + 4 = 104
    state_dim = env.width * env.height + 4
    agents = [ConstrainedQLearningAgent(state_dim=state_dim) for _ in range(n_agents)]
    
    n_episodes = 200
    returns_history = []
    
    for ep in range(n_episodes):
        states_list = env.reset()  # list of (state_dim,) arrays, one per agent
        done = False
        ep_returns = np.zeros(n_agents, dtype=float)
        
        while not done:
            # each agent picks an action
            actions = []
            for i in range(n_agents):
                a = agents[i].select_action(states_list[i])
                actions.append(a)
            
            next_states_list, rewards, done, info = env.step(actions)
            
            # store transitions
            for i in range(n_agents):
                agents[i].store_transition(
                    s=states_list[i],
                    a=actions[i],
                    r=rewards[i],
                    s_next=next_states_list[i],
                    done=done
                )
                ep_returns[i] += rewards[i]
            
            # train step for each agent
            for i in range(n_agents):
                agents[i].train_step()
            
            states_list = next_states_list
        
        returns_history.append(np.sum(ep_returns))
        # optional: print progress
        if (ep+1) % 10 == 0:
            avg_return = np.mean(returns_history[-10:])
            print(f"Episode {ep+1}/{n_episodes}, avg_return(last10)={avg_return:.1f}")
    
    # Summarize
    df = pd.DataFrame({
        "episode": range(1, n_episodes+1),
        "return": returns_history
    })
    print("Last 10 episodes return:", df["return"].tail(10).values)
    print("No slow-tensor warnings if we used np.array(...) properly for batch creation!")

if __name__ == "__main__":
    main()
