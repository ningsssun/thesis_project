import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Q-Network Definition
class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.fc(x)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# Grid World Environment
class GridWorld:
    def __init__(self, size=10, n_agents=2, obstacle_density=0.1):
        self.size = size
        self.n_agents = n_agents
        self.obstacles = self._generate_obstacles(obstacle_density)
        self.agents = []
        self.goals = []
    
    def _generate_obstacles(self, density):
        obstacles = set()
        while len(obstacles) < int(self.size**2 * density):
            obstacles.add((np.random.randint(self.size), np.random.randint(self.size)))
        return obstacles
    
    def reset(self):
        self.agents = []
        self.goals = []
        # Place agents and goals randomly
        for _ in range(self.n_agents):
            while True:
                pos = (np.random.randint(self.size), np.random.randint(self.size))
                if pos not in self.obstacles and pos not in [a for a, _ in self.agents]:
                    self.agents.append(pos)
                    break
            while True:
                goal = (np.random.randint(self.size), np.random.randint(self.size))
                if goal not in self.obstacles and goal != pos and goal not in self.goals:
                    self.goals.append(goal)
                    break
        return self._get_state()
    
    def _get_state(self):
        # State includes agent positions, goals, and obstacles
        return {
            'agents': self.agents.copy(),
            'goals': self.goals.copy(),
            'obstacles': list(self.obstacles)
        }
    
    def get_valid_actions(self, agent_idx):
        x, y = self.agents[agent_idx]
        valid = []
        # Actions: 0=up, 1=down, 2=left, 3=right
        for action in range(4):
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            new_x, new_y = x + dx, y + dy
            if (0 <= new_x < self.size and 0 <= new_y < self.size and
                (new_x, new_y) not in self.obstacles and
                (new_x, new_y) not in self.agents):
                valid.append(action)
        return valid
    
    def step(self, actions):
        new_positions = []
        rewards = [0] * self.n_agents
        done = False
        
        # Move agents and check collisions
        for i, action in enumerate(actions):
            x, y = self.agents[i]
            dx, dy = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
            new_x, new_y = x + dx, y + dy
            # Check if new position is valid and not occupied
            if (new_x, new_y) in self.agents or (new_x, new_y) in new_positions:
                rewards[i] = -100  # Collision penalty
                done = True
            else:
                new_positions.append((new_x, new_y))
        
        # Update positions if no collision
        if not done:
            self.agents = new_positions
            # Check if all agents reached goals
            success = all(pos == goal for pos, goal in zip(self.agents, self.goals))
            for i in range(self.n_agents):
                if self.agents[i] == self.goals[i]:
                    rewards[i] += 100  # Goal reward
            done = success or len(self.agents[0]) == 0  # Max steps
        
        return self._get_state(), rewards, done

# Constrained Q-Learning Agent
class Agent:
    def __init__(self, state_dim, action_dim, gamma=0.95, lr=0.001, epsilon=1.0, epsilon_decay=0.995):
        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.action_dim = action_dim
    
    def act(self, state, valid_actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(valid_actions)
        else:
            # Ensure state is a valid sequence (e.g., list or NumPy array)
            if isinstance(state, (list, np.ndarray)):
                state_tensor = torch.FloatTensor(state)
            else:
                raise ValueError("State must be a sequence (list or NumPy array)")
            
            with torch.no_grad():
                q_values = self.q_net(state_tensor)
            
            # Mask invalid actions
            masked_q = q_values.clone()
            for a in range(self.action_dim):
                if a not in valid_actions:
                    masked_q[a] = -float('inf')
            return torch.argmax(masked_q).item()
    
    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
    
    def train_step(self, batch):
        states, actions, rewards, next_states, dones, next_valids = zip(*batch)
        # Convert to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)
        
        # Compute Q(s,a)
        current_q = self.q_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute max Q'(s',a') over valid actions
        next_q = self.target_net(next_states)
        for i, valid in enumerate(next_valids):
            for a in range(self.action_dim):
                if a not in valid:
                    next_q[i][a] = -float('inf')
        next_q_max = next_q.max(1)[0].detach()
        
        # Target Q-value
        target_q = rewards + (1 - dones) * self.gamma * next_q_max
        
        # Loss and backpropagation
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

# Training Loop
def train():
    env = GridWorld(n_agents=2)
    state_dim = 4  # Example: Agent's position (x, y) and goal's position (x, y)
    action_dim = 4
    agents = [Agent(state_dim, action_dim) for _ in range(env.n_agents)]
    buffer = ReplayBuffer(10000)
    batch_size = 64
    episodes = 1000
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_rewards = [0] * env.n_agents
        
        while not done:
            # Get actions for all agents
            actions = []
            for i in range(env.n_agents):
                # Construct agent_state as a list or NumPy array
                agent_state = [
                    env.agents[i][0],  # Agent's x position
                    env.agents[i][1],  # Agent's y position
                    env.goals[i][0],   # Goal's x position
                    env.goals[i][1]    # Goal's y position
                ]
                
                # Ensure agent_state is a valid sequence
                if not isinstance(agent_state, (list, np.ndarray)):
                    raise ValueError(f"Invalid state format: {agent_state}")
                
                valid_actions = env.get_valid_actions(i)
                action = agents[i].act(agent_state, valid_actions)
                actions.append(action)
            
            # Step environment
            next_state, rewards, done = env.step(actions)
            
            # Store transitions
            for i in range(env.n_agents):
                agent_next_state = [
                    next_state['agents'][i][0],  # Agent's next x position
                    next_state['agents'][i][1],  # Agent's next y position
                    next_state['goals'][i][0],   # Goal's x position
                    next_state['goals'][i][1]    # Goal's y position
                ]
                buffer.push((
                    agent_state,  # Current state
                    actions[i],    # Action taken
                    rewards[i],    # Reward received
                    agent_next_state,  # Next state
                    done,         # Episode done
                    env.get_valid_actions(i)  # Valid actions in next state
                ))
            
            # Train agents
            if len(buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                for agent in agents:
                    loss = agent.train_step(batch)
            
            # Update target networks periodically
            if episode % 10 == 0:
                for agent in agents:
                    agent.update_target()
            
            # Decay epsilon
            for agent in agents:
                agent.epsilon *= agent.epsilon_decay
        
        # Log metrics (collisions, success, steps)
        print(f"Episode {episode}: Rewards {total_rewards}")

if __name__ == "__main__":
    train()
