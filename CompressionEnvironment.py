import gym
from Environment import Environment



class CompressionEnvironment(gym.Env):
    action_space = gym.spaces.Discrete
    observation_space = gym.spaces.Discrete 

    
    def __init__(self, env:Environment):
        super().__init__()
        self.env = env
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.observation_space = gym.spaces.Discrete(env.num_states)
        self.reset()
        self.pos = self.env.start
        self.terminal = self.env.terminal
        
    
    def reset(self,seed=None, options=None):
        super().reset(seed=seed)    
        self.pos = self.env.start
        return self.pos, {}
    
    def step(self, action):
        next_state, reward, terminated, truncated, running_ppl_hit = self.env.dynamics(self.pos, action)
        self.pos = next_state[0]
        return self.pos, reward[0], terminated[0], truncated[0], running_ppl_hit[0]