import numpy as np
import random
import time
import pandas as pd
from collections import defaultdict

# -----------------------------------------------------
# 1. ENVIRONMENT DEFINITION
# -----------------------------------------------------
class GridWorldEnv:
    """
    A simple grid-world environment with multiple agents, obstacles, and collisions.
    
    Attributes:
        grid_size   : (width, height) of the grid
        n_agents    : number of agents
        obstacle_map: 2D array indicating obstacle locations (1 = obstacle, 0 = free)
        agent_positions: list of (x,y) for each agent
        goals       : list of (x,y) for each agent's goal
        done        : whether the episode is finished
        step_count  : current timestep in this episode
        max_steps   : maximum timesteps allowed before forced termination
    """
    def __init__(self, grid_size=(10,10), n_agents=2, max_steps=100, obstacle_prob=0.2):
        self.width, self.height = grid_size
        self.n_agents = n_agents
        self.max_steps = max_steps
        self.done = False
        self.step_count = 0
        
        # Create obstacle map randomly (1 = obstacle, 0 = free)
        self.obstacle_map = np.zeros(grid_size, dtype=int)
        for x in range(self.width):
            for y in range(self.height):
                if random.random() < obstacle_prob:
                    self.obstacle_map[x, y] = 1
        
        self.agent_positions = []
        self.goals = []
    
    def reset(self):
        """
        Resets the environment for a new episode:
        - Places each agent in a free cell
        - Assigns each agent a distinct goal cell
        - Clears done flag, resets step_count
        - Returns initial observation(s)
        """
        self.done = False
        self.step_count = 0
        self.agent_positions = []
        self.goals = []
        
        # Gather free cells
        free_cells = [(x,y) for x in range(self.width)
                               for y in range(self.height)
                                 if self.obstacle_map[x,y] == 0]
        random.shuffle(free_cells)
        
        # Place agents
        for _ in range(self.n_agents):
            self.agent_positions.append(free_cells.pop())
        
        # Place goals (distinct from agent positions)
        for _ in range(self.n_agents):
            self.goals.append(free_cells.pop())
        
        # Return initial observations if needed
        return self._get_observations()
    
    def step(self, actions):
        """
        Takes a list of actions (one per agent), updates positions,
        checks collisions and goals, returns next observations, rewards, done, info
        """
        rewards = [0.0]*self.n_agents
        
        # 1. propose next positions
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
                # no-op
                pass
            
            # Check boundary
            if nx < 0 or nx >= self.width or ny < 0 or ny >= self.height:
                nx, ny = x, y  # invalid, revert
            
            # Check obstacle
            if self.obstacle_map[nx, ny] == 1:
                nx, ny = x, y  # blocked by obstacle
            
            next_positions.append((nx, ny))
        
        # 2. check collisions among agents
        unique_positions = set(next_positions)
        if len(unique_positions) < self.n_agents:
            # collision occurred
            self.done = True
            for i in range(self.n_agents):
                rewards[i] -= 100.0  # big penalty
        else:
            # update positions
            self.agent_positions = next_positions
        
        # 3. reward for goals
        for i in range(self.n_agents):
            if self.agent_positions[i] == self.goals[i]:
                rewards[i] += 100.0
        
        # check if all agents reached goals
        all_done = True
        for i in range(self.n_agents):
            if self.agent_positions[i] != self.goals[i]:
                all_done = False
                break
        if all_done:
            self.done = True
        
        # step penalty
        for i in range(self.n_agents):
            rewards[i] -= 1.0  # time cost
        
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.done = True
        
        next_obs = self._get_observations()
        return next_obs, rewards, self.done, {}
    
    def _get_observations(self):
        """
        Returns a list of 'observations,' one per agent.
        For a simple tabular approach, we might not need a high-dimensional representation.
        But to keep it consistent, we provide something akin to a global or partial state.
        """
        obs = []
        for i in range(self.n_agents):
            # minimal representation: agent_i pos + goal pos + pos of other agents?
            # you can do more or less. For demonstration, we do a dictionary or tuple.
            agent_pos = self.agent_positions[i]
            goal_pos  = self.goals[i]
            
            # If you want to do a purely tabular approach, you must encode these
            # as an integer or a small discrete representation.
            # We'll do that in the Agent's encode_state function.
            
            # Return raw data for now
            obs.append((agent_pos, goal_pos, tuple(self.agent_positions)))
        return obs

# -----------------------------------------------------
# 2. TABULAR Q-LEARNING AGENT
# -----------------------------------------------------
class TabularQLearningAgent:
    """
    Each agent maintains a dictionary: Q[state][action] -> float
    where state is an encoded representation of the environment from its viewpoint.
    
    We'll do an epsilon-greedy policy, storing transitions and do in-place Q updates.
    This is a simplified approach (no replay buffer).
    """
    def __init__(self, alpha=0.01, gamma=0.95, eps=1.0, eps_min=0.05, eps_decay=1e-4, action_dim=5):
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.action_dim = action_dim
        
        # Q[state] = array of shape (action_dim,)
        self.Q = defaultdict(lambda: np.zeros(self.action_dim, dtype=float))
    
    def encode_state(self, obs):
        """
        obs might be a tuple: (agent_pos, goal_pos, all_agent_positions)
        We'll transform it into a discrete or hashable representation (e.g., a string or a tuple).
        Example:
            agent_pos: (ax, ay)
            goal_pos: (gx, gy)
            all_agent_positions: ((x1, y1), (x2, y2), ...)
        We'll just flatten them into a single big tuple or string.
        """
        # For instance:
        agent_pos, goal_pos, all_poses = obs
        # Convert to a sorted tuple if we want a consistent ordering of all_poses
        # but let's assume the agent index is known, so we keep them as is.
        
        # We'll create a big tuple
        # e.g., ((ax, ay), (gx, gy), (x1, y1), (x2, y2), ...)
        # that is a valid dictionary key
        big_tuple = (agent_pos, goal_pos, all_poses)
        return big_tuple
    
    def select_action(self, state_enc, valid_actions):
        """
        Epsilon-greedy among valid_actions, using self.Q[state_enc].
        """
        if random.random() < self.eps:
            return random.choice(valid_actions)
        else:
            qvals = self.Q[state_enc]
            masked_q = np.full_like(qvals, -1e9, dtype=float)
            for a in valid_actions:
                masked_q[a] = qvals[a]
            return int(np.argmax(masked_q))
    
    def update_q(self, s_enc, a, r, s_next_enc, valid_next_acts, done):
        """
        Performs the in-place Q-update:
          Q[s,a] += alpha * (r + gamma * max_a' Q[s_next,a'] - Q[s,a])
        but only over valid_next_acts if not done
        """
        old_val = self.Q[s_enc][a]
        
        if done:
            target = r
        else:
            next_qvals = self.Q[s_next_enc]
            # mask invalid acts
            masked_next = np.full(len(next_qvals), -1e9)
            for act in valid_next_acts:
                masked_next[act] = next_qvals[act]
            target = r + self.gamma * np.max(masked_next)
        
        self.Q[s_enc][a] = old_val + self.alpha * (target - old_val)
    
    def decay_epsilon(self):
        if self.eps > self.eps_min:
            self.eps -= self.eps_decay
            if self.eps < self.eps_min:
                self.eps = self.eps_min
    
    def get_valid_actions(self, obs, env):
        """
        For demonstration, we'll just return all 5 actions (0..4).
        Optionally you can filter out definitely invalid moves (out-of-bounds or collisions).
        Usually, that's done in env step. We'll keep it simple here.
        """
        return [0,1,2,3,4]

# -----------------------------------------------------
# 3. TRAINING LOOP (MULTI-AGENT)
# -----------------------------------------------------
def train_tabular_CMARL(num_episodes=500, n_agents=2, grid_size=(10,10), max_steps=100):
    """
    Demonstration of training multiple agents in a grid-world using tabular Q-learning with constraints.
    Constraints are enforced in the environment (collisions => large negative reward + done).
    """
    env = GridWorldEnv(grid_size=grid_size, n_agents=n_agents, max_steps=max_steps, obstacle_prob=0.2)
    
    # Create one TabularQLearningAgent per agent
    agents = []
    for _ in range(n_agents):
        agent = TabularQLearningAgent(alpha=0.05, gamma=0.95, eps=1.0, eps_min=0.05, eps_decay=1e-4)
        agents.append(agent)
    
    episode_rewards = []
    collision_counts = []
    success_flags = []
    
    start_time = time.time()
    
    for ep in range(num_episodes):
        # reset environment
        obs_list = env.reset()  # list of obs (one per agent)
        done = False
        
        # track
        ep_rew = np.zeros(n_agents)
        collision_happened = False
        
        while not done:
            actions = []
            for i in range(n_agents):
                # encode state
                s_enc = agents[i].encode_state(obs_list[i])
                valid_acts = agents[i].get_valid_actions(obs_list[i], env)
                
                # pick action
                a = agents[i].select_action(s_enc, valid_acts)
                actions.append(a)
            
            next_obs_list, rewards, done, info = env.step(actions)
            
            # check if collision happened
            if done and any(r <= -100 for r in rewards):
                collision_happened = True
            
            # update Q for each agent
            for i in range(n_agents):
                s_enc = agents[i].encode_state(obs_list[i])
                a = actions[i]
                r = rewards[i]
                if not done:
                    # get next state's valid actions
                    valid_next_acts = agents[i].get_valid_actions(next_obs_list[i], env)
                    s_next_enc = agents[i].encode_state(next_obs_list[i])
                    agents[i].update_q(s_enc, a, r, s_next_enc, valid_next_acts, done=False)
                else:
                    # terminal update
                    agents[i].update_q(s_enc, a, r, None, [], done=True)
                
                ep_rew[i] += r
            
            obs_list = next_obs_list
        
        # end of episode
        total_ep_rew = np.sum(ep_rew)
        episode_rewards.append(total_ep_rew)
        collision_counts.append(1 if collision_happened else 0)
        
        # success if no collision + all agents on goal
        # environment sets done if all agents reached or collision
        # we can check agent positions vs goals
        success = 1
        for i in range(n_agents):
            if env.agent_positions[i] != env.goals[i]:
                success = 0
                break
        success_flags.append(success)
        
        # decay eps
        for agent in agents:
            agent.decay_epsilon()
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # create a DataFrame
    df = pd.DataFrame({
        'episode': range(num_episodes),
        'total_reward': episode_rewards,
        'collision': collision_counts,
        'success': success_flags
    })
    
    return agents, df, training_time

# -----------------------------------------------------
# 4. MAIN: EXAMPLE USAGE
# -----------------------------------------------------
if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    # train with 2 agents in a 10x10 grid, 500 episodes
    agents, results_df, ttime = train_tabular_CMARL(num_episodes=500, n_agents=2, grid_size=(10,10), max_steps=100)
    
    print(f"Training completed in {ttime:.2f} seconds.")
    print(results_df.tail(10))
    
    # We can do basic analysis
    success_rate = results_df['success'].mean() * 100
    collisions = results_df['collision'].sum()
    avg_reward = results_df['total_reward'].mean()
    print(f"Final success rate: {success_rate:.2f}%")
    print(f"Total collisions across all episodes: {collisions}")
    print(f"Average total reward per episode: {avg_reward:.2f}")
    
    # optional: rolling average to see learning curve
    window_size = 20
    results_df['reward_smooth'] = results_df['total_reward'].rolling(window=window_size).mean()
    results_df['collision_smooth'] = results_df['collision'].rolling(window=window_size).mean()
    
    # You can plot in your Jupyter or any environment:
    # import matplotlib.pyplot as plt
    # plt.plot(results_df['reward_smooth'], label='Reward (smooth)')
    # plt.legend()
    # plt.show()
