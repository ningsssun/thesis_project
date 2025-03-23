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

# 3. Constrained Q-Learning Agent
class ConstrainedQLearningAgent:
    """
    Each agent has its own Q-network, optimizer, etc
    """
    def __init__(self, state_dim, action_dim=5, lr=1e-3, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps = 1.0  # exploration
        self.eps_min = 0.05
        self.eps_decay = 1e-4  # decay per step or episode
        self.lr = lr
        
        self.q_network = DQNNetwork(state_dim, 128, action_dim)
        self.target_network = DQNNetwork(state_dim, 128, action_dim)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.buffer = deque(maxlen=50000)
        self.batch_size = 64
        self.update_count = 0
        self.target_update_freq = 200  # how often to sync target net
    
    def select_action(self, state, valid_actions):
        """
        Epsilon-greedy among valid_actions
        """
        if np.random.rand() < self.eps:
            # random among valid actions
            return np.random.choice(valid_actions)
        else:
            # pick best Q among valid actions
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                qvals = self.q_network(state_t)[0].cpu().numpy()
            # set invalid actions Q-values to very negative
            masked_qvals = np.full_like(qvals, -1e9)
            for a in valid_actions:
                masked_qvals[a] = qvals[a]
            return np.argmax(masked_qvals)
    
    def store_transition(self, s, a, r, s_next, done, valid_next_actions):
        # valid_next_actions can be used at training time
        self.buffer.append((s, a, r, s_next, done, valid_next_actions))
    
    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, states_next, dones, valid_next_acts = zip(*batch)
        
        states_t = torch.FloatTensor(states)
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(states_next)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        # Current Q
        qvals = self.q_network(states_t)
        gathered_q = qvals.gather(1, actions_t)  # shape [batch_size,1]
        
        # Target Q
        with torch.no_grad():
            next_qvals = self.target_network(next_states_t)
        
        max_next_q = []
        for i in range(self.batch_size):
            if dones[i]:
                max_next_q.append(0.0)
            else:
                qrow = next_qvals[i].cpu().numpy()
                valid_acts = valid_next_acts[i]
                # mask invalid
                masked_q = np.full(len(qrow), -1e9)
                for a in valid_acts:
                    masked_q[a] = qrow[a]
                max_next_q.append(np.max(masked_q))
        
        max_next_q = torch.FloatTensor(max_next_q).unsqueeze(1)
        target = rewards_t + (1 - dones_t) * self.gamma * max_next_q
        
        loss = self.criterion(gathered_q, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # update target net
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # decay epsilon
        if self.eps > self.eps_min:
            self.eps -= self.eps_decay

    def get_valid_actions(self, env, agent_index):
        """
        Returns the set of valid actions for the agent at agent_index
        """
        all_acts = [0, 1, 2, 3, 4]  # All possible actions
        valid_acts = []

        # Get the agent's current position
        x, y = env.agent_positions[agent_index]

        # Check each action for validity
        for action in all_acts:
            if action == 0:  # Move up
                new_x, new_y = x, y - 1
            elif action == 1:  # Move down
                new_x, new_y = x, y + 1
            elif action == 2:  # Move left
                new_x, new_y = x - 1, y
            elif action == 3:  # Move right
                new_x, new_y = x + 1, y
            elif action == 4:  # Stay in place
                new_x, new_y = x, y

            # Check if the new position is valid
            if (0 <= new_x < env.width and  # Within grid bounds
                0 <= new_y < env.height and
                env.obstacle_map[new_x, new_y] == 0 and  # Not an obstacle
                (new_x, new_y) not in env.agent_positions):  # Not occupied by another agent
                valid_acts.append(action)

        return valid_acts

# 4. Training loop
def train_CMARL(num_episodes=2000, n_agents=2, grid_size=(10,10), log_dir="logs"):
    # Create log directory if it doesn't exist
    import os
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize environment and agents
    env = GridWorldEnv(grid_size=grid_size, n_agents=n_agents, max_steps=100)
    state_dim = env.width * env.height + 4
    agents = [ConstrainedQLearningAgent(state_dim=state_dim, action_dim=5) for _ in range(n_agents)]
    
    # Logging lists
    episode_rewards = []
    collision_counts = []
    success_flags = []
    time_start = time.time()
    
    # Training loop
    for ep in range(num_episodes):
        states = env.reset()
        done = False
        ep_reward = np.zeros(n_agents)
        collision_happened = False
        
        while not done:
            actions = []
            for i in range(n_agents):
                valid_acts = agents[i].get_valid_actions(env, i)
                a = agents[i].select_action(states[i], valid_acts)
                actions.append(a)
            
            next_states, rewards, done, info = env.step(actions)
            
            if done and any(r <= -100 for r in rewards):
                collision_happened = True
            
            for i in range(n_agents):
                valid_next_acts = agents[i].get_valid_actions(env, i) if not done else []
                agents[i].store_transition(
                    s=states[i],
                    a=actions[i],
                    r=rewards[i],
                    s_next=next_states[i],
                    done=done,
                    valid_next_actions=valid_next_acts
                )
                ep_reward[i] += rewards[i]
            
            states = next_states
            
            for i in range(n_agents):
                agents[i].train_step()
        
        # Logging
        episode_rewards.append(np.sum(ep_reward))
        collision_counts.append(1 if collision_happened else 0)
        
        success = 1
        for i in range(n_agents):
            ax, ay = env.agent_positions[i]
            gx, gy = env.goals[i]
            if (ax, ay) != (gx, gy):
                success = 0
                break
        success_flags.append(success)
    
    time_end = time.time()
    training_time = time_end - time_start
    
    # Create a DataFrame for results
    results_df = pd.DataFrame({
        'episode': range(num_episodes),
        'total_reward': episode_rewards,
        'collision': collision_counts,
        'success': success_flags
    })
    
    # Add rolling metrics
    results_df['rolling_reward'] = results_df['total_reward'].rolling(window=20).mean()
    results_df['rolling_collision'] = results_df['collision'].rolling(window=20).mean()
    
    # Save results to a CSV file
    results_file = os.path.join(log_dir, "training_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Training results saved to {results_file}")
    
    # Save metadata (hyperparameters, training time, etc.) to a text file
    metadata_file = os.path.join(log_dir, "metadata.txt")
    with open(metadata_file, "w") as f:
        f.write(f"Training Metadata\n")
        f.write(f"=================\n")
        f.write(f"Number of Episodes: {num_episodes}\n")
        f.write(f"Number of Agents: {n_agents}\n")
        f.write(f"Grid Size: {grid_size}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Final Success Rate: {results_df['success'].mean() * 100:.2f}%\n")
        f.write(f"Total Collisions: {results_df['collision'].sum()}\n")
    print(f"Metadata saved to {metadata_file}")
    
    return agents, results_df, training_time

# 5. Run a simple experiment and print results
if __name__ == "__main__":
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Specify the log directory
    log_directory = "training_logs"
    
    # Run training
    agents, df, ttime = train_CMARL(num_episodes=200, n_agents=2, grid_size=(10,10), log_dir=log_directory)
    
    # Print final stats
    print(f"Finished training. Elapsed time: {ttime:.2f}s")
    final_success_rate = df['success'].mean() * 100
    final_collisions = df['collision'].sum()
    print(f"Success Rate: {final_success_rate:.2f}% over {len(df)} episodes")
    print(f"Total collisions: {final_collisions}")
    
    # Print last 10 lines of the results
    print(df.tail(10))
