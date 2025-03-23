#!/usr/bin/env python3
import gym
import numpy as np
from gym import spaces

##############################################################################
# 1) CONSTRAINED QUEUE ENVIRONMENT
#    - Single agent controlling arrival probability & service rate
#    - There's a "safety" or "collision" constraint on average queue length
##############################################################################

class ConstrainedQueueEnv(gym.Env):
    """
    A toy queueing environment with:
      - discrete states = {0, 1, 2, ..., capacity}
      - actions = (flow_prob, service_prob) chosen from discrete sets
      - "collisions" or "safety" violations if average queue length > constraint

    We'll keep it simple:
      - One integer 'queue_length' as the environment state.
      - Each step: agent picks an action index that corresponds to some flow rate, service rate.
      - Then, queue_length evolves based on Bernoulli arrivals, services.

    The agent's main reward is negative queue length (it wants to keep it small),
    but it must also keep the average queue length below a threshold (like a constraint).
    In a real CMDP, you'd incorporate Lagrangians or specialized Q-updates for the constraint.
    Here, we simply demonstrate how to 'mask' or 'penalize' constraint-violating actions.
    """
    metadata = {"render.modes": ["human"]}

    def __init__(self, capacity=10, constraint_threshold=6):
        super().__init__()
        self.capacity = capacity
        self.constraint_threshold = constraint_threshold

        # ACTION SPACE:
        # Let's say we define a handful of discrete (flow_prob, service_prob) pairs:
        self.possible_actions = [
            (0.1, 0.9),
            (0.3, 0.7),
            (0.5, 0.5),
            (0.7, 0.3),
            (0.9, 0.1)
        ]
        self.action_space = spaces.Discrete(len(self.possible_actions))

        # OBSERVATION SPACE:
        # The state is an integer queue_length in [0, capacity].
        self.observation_space = spaces.Discrete(self.capacity + 1)

        # The environment state:
        self.queue_length = 0

        # For demonstration, track average queue length as a 'safety measure'.
        self.avg_queue_length = 0.0
        self.total_steps = 0

    def reset(self):
        self.queue_length = 0
        self.avg_queue_length = 0.0
        self.total_steps = 0
        return self.queue_length

    def step(self, action_idx):
        """
        agent picks an action in [0..4],
        environment evolves with Bernoulli arrivals and services
        """
        flow_prob, service_prob = self.possible_actions[action_idx]
        self.total_steps += 1

        # arrivals:
        arrival = 1 if np.random.rand() < flow_prob else 0
        # services:
        service = 1 if np.random.rand() < service_prob else 0

        # new queue length:
        new_len = self.queue_length + arrival - service
        new_len = max(new_len, 0)
        new_len = min(new_len, self.capacity)

        self.queue_length = new_len

        # update average for constraint checking:
        self.avg_queue_length = ((self.total_steps-1)*self.avg_queue_length + new_len) / self.total_steps

        # define a reward that wants to keep queue short:
        # negative of queue length, e.g.  -queue_length
        reward = -float(self.queue_length)

        # done if we exceed some big limit or if we want an episodic scenario
        done = False
        if self.total_steps >= 100:
            done = True

        # info can hold whether we are violating the constraint
        # e.g. if avg_queue_length > constraint_threshold => not safe
        constraint_violation = (self.avg_queue_length > self.constraint_threshold)
        info = {
            "constraint_ok": not constraint_violation,
            "avg_queue_length": self.avg_queue_length
        }

        return self.queue_length, reward, done, info

    def render(self, mode="human"):
        print(f"Queue length: {self.queue_length}, avg = {self.avg_queue_length:.2f}")


##############################################################################
# 2) CONSTRAINED Q-LEARNING AGENT
#    - We keep a Q-table of shape [n_states, n_actions]
#    - We'll forcibly mask or skip actions that appear to push us "too high"
#      in terms of the constraint. This is a naive approach to constraints.
##############################################################################

class ConstrainedQLearningAgent:
    """
    A simplistic agent that tries to do Q-learning but skip any action that
    leads to 'likely' constraint violation. In practice, you'd have a more
    complex Lagrange-based method or primal-dual Q-learning. Here, we just
    show how you might incorporate constraint logic.

    Steps:
      1) We have Q[state, action].
      2) Epsilon-greedy among actions not predicted to violate constraint
         or to cause huge average queue. (We'll do a naive check with 'obs'.)
      3) Q-learning update as usual.
    """
    def __init__(self, n_states, n_actions, gamma=0.95, alpha=0.1, epsilon=0.1,
                 queue_constraint=6.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.queue_constraint = queue_constraint

        # Q-table:
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state, info):
        """
        A naive approach: skip any action if the queue is already big enough
        that we might be violating constraints. Actually check info["constraint_ok"]?
        We'll illustrate a simpler approach: if state >= queue_constraint => skip
        actions that have small service probability in self.possible_actions
        (i.e. we only choose actions that aggressively reduce queue).
        """
        # If we want to do e-greedy among feasible actions:
        feasible_actions = self._compute_feasible_actions(state)
        if np.random.rand() < self.epsilon:
            # random among feasible
            action = np.random.choice(feasible_actions)
        else:
            # choose best among feasible
            qvals = self.Q[state, feasible_actions]
            best_idx = np.argmax(qvals)
            action = feasible_actions[best_idx]
        return action

    def _compute_feasible_actions(self, state):
        """
        A trivial 'masking' approach: if queue length >= queue_constraint,
        we restrict to the top 2 actions (which presumably have highest service prob).
        Otherwise, any action is fine. Adjust to your constraint logic.
        """
        if state >= self.queue_constraint:
            # let's keep only the actions with service prob >= 0.5
            # recall possible_actions is:  [(0.1,0.9),(0.3,0.7),(0.5,0.5),(0.7,0.3),(0.9,0.1)]
            # i.e. indices: 0->(0.1,0.9),1->(0.3,0.7),2->(0.5,0.5),3->(0.7,0.3),4->(0.9,0.1)
            # we want service prob >= 0.5 => actions 0,1,2
            return [0,1,2]
        else:
            return list(range(self.n_actions))

    def update(self, state, action, reward, next_state, done):
        """
        Standard Q-learning update.
        """
        best_next = np.max(self.Q[next_state]) if not done else 0.0
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.Q[state, action]
        self.Q[state, action] += self.alpha * td_error


##############################################################################
# 3) MAIN TRAINING LOOP (demonstration)
##############################################################################

def main():
    # 1) create environment
    env = ConstrainedQueueEnv(capacity=10, constraint_threshold=6)

    # 2) create agent
    n_states = env.observation_space.n  # 0..10
    n_actions = env.action_space.n      # 5
    agent = ConstrainedQLearningAgent(n_states, n_actions,
                                      gamma=0.95, alpha=0.1,
                                      epsilon=0.1, queue_constraint=6.0)

    n_episodes = 1000
    max_steps_per_episode = 100

    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        for step in range(max_steps_per_episode):
            # at each step, pick action
            # we'll pass an 'info' dict if we want to use env's 'constraint_ok' or so
            # but to keep it simple, we'll do 'state-based' logic in select_action
            action = agent.select_action(obs, info={})

            next_obs, reward, done, info = env.step(action)

            agent.update(obs, action, reward, next_obs, done)

            obs = next_obs
            if done:
                break

    # done training. let's do a quick test run:
    obs = env.reset()
    env.render()
    for t in range(10):
        act = agent.select_action(obs, info={})
        next_obs, rew, done, info = env.step(act)
        env.render()
        obs = next_obs
        if done:
            break

if __name__ == "__main__":
    main()
