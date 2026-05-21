# TrainAgent.py — fixed
import numpy as np
class TrainAgent:
    def __init__(self, env, agent, num_episodes):
        self.env          = env
        self.agent        = agent
        self.num_episodes = num_episodes
        self.action_bits  = {0: 16, 1: 8, 2: 4, 3: 2, 4: 0,5:0,6:0}
        self.alpha        = 0.9
        self.beta         = (1- self.alpha)*100
        self.ppl_budget   = 0.5

    def get_signal_vector(self, layer_idx, budget_remaining, running_ppl_hit):
        """Convert layer index into the full signal vector the Q-network expects."""
        layer = self.env.env.layers[layer_idx]   # access Environment via CompressionEnvironment
        n = len(self.env.env.layers)
        is_protected = 1.0 if (layer_idx == 0 or layer_idx == n - 1) else 0.0
        return np.array([
            layer['importance'],
            layer['sparsity_ratio'],
            budget_remaining,
            min(running_ppl_hit, 1.0),
            is_protected
        ], dtype=np.float32)   # shape (4,) — matches Q-network input

    def train(self):
        all_rewards = []

        for episode in range(self.num_episodes):
            layer_idx, _    = self.env.reset()   # integer layer index
            done            = False
            episode_reward  = 0.0
            running_ppl_hit = 0.0
            budget_remaining= 0.5

            while not done:
                # build signal vector from layer index
                state = self.get_signal_vector(layer_idx, budget_remaining, running_ppl_hit)

                action = self.agent.act(state)   # shape (4,)

                next_layer_idx, ppl_hit, terminated, truncated, _ = self.env.step(action)

                # update running context
                running_ppl_hit  += ppl_hit
                bits_after        = self.action_bits[action]
                budget_remaining -= ((16 - bits_after) / 16) / len(self.env.env.layers)
                

                # reward
                compression_gain = (16 - bits_after) / 16
                ppl_violation    = max(0.0, running_ppl_hit - self.ppl_budget)
                reward           = self.alpha * compression_gain - self.beta * ppl_violation  # small incentive to maintain headroom

                # build next signal vector
                if not terminated:
                    next_state = self.get_signal_vector(next_layer_idx, budget_remaining, running_ppl_hit)
                else:
                    next_state = np.zeros(5, dtype=np.float32)

                if state[4] == True and action in [3, 4, 5, 6]:  # if current layer is protected, override reward to heavily penalize any non-KEEP action
                    reward += -10.0
                
                '''if budget_remaining <= 0.001:  # if budget is exceeded, override reward to heavily penalize
                    terminated = True
                    reward -= 10.0'''

                self.agent.remember(state, action, reward, next_state, terminated)
                self.agent.learn()

                layer_idx      = next_layer_idx
                done           = terminated or truncated
                episode_reward += reward

            all_rewards.append(episode_reward)

            if episode % 10 == 0:
                mean = np.mean(all_rewards[-10:])
                print(f"Episode {episode:4d} | reward {episode_reward:7.3f} | mean(10) {mean:7.3f}")

        return self.agent

'''if __name__ == "__main__":

    from Environment import Environment
    from CompressionEnvironment import CompressionEnvironment
    from DQNAgent import DQNAgent

    env = Environment(num_actions=5, num_states=4, start=0, terminal=3)
    compression_env = CompressionEnvironment(env)
    obs_shape = compression_env.observation_space.n
    num_actions = compression_env.action_space.n
    agent = DQNAgent(obs_shape=4, num_actions=num_actions, lr=0.001, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, batch_size=64, min_experiences=1000, target_update_freq=10)
    trainer = TrainAgent(compression_env, agent, num_episodes=1000)

    trained_agent = trainer.train()'''



if __name__ == "__main__":
    import numpy as np
    from Environment import Environment
    from CompressionEnvironment import CompressionEnvironment
    from DQNAgent import DQNAgent
    from TrainAgent import TrainAgent

    # --- setup ---
    env             = Environment(num_actions=6, num_states=32, start=0, terminal=31)
    compression_env = CompressionEnvironment(env)
    num_actions     = compression_env.action_space.n
    obs_shape       = 5     # [importance, sparsity, budget_remaining, ppl_delta, is_protected]

    agent = DQNAgent(
        obs_shape      = obs_shape,
        num_actions    = num_actions,
        lr             = 0.001,
        gamma          = 0.99,
        epsilon_start  = 1.0,
        epsilon_end    = 0.01,
        epsilon_decay  = 0.995,
        batch_size     = 128,
        min_experiences= 1000,
        target_update_freq = 100
    )

    # --- training ---
    trainer       = TrainAgent(compression_env, agent, num_episodes=500)
    trained_agent = trainer.train()

    # --- inference ---
    for budget in [0.15, 0.10, 0.25,0.5,0.8,1.0,1.15,1.5]:
        print(f"\n=== Inference with PPL budget {budget:.2f} ===")
        trainer.ppl_budget = budget
        print("\n--- Compression Plan ---")
        action_names     = {0:'KEEP', 1:'INT8', 2:'INT4', 3:'PRUNE15', 4:'PRUNE25', 5:'PRUNE50'}
        budget_remaining = budget
        running_ppl_hit  = 0.0
        actions = []
        for i in range(len(env.layers)):
            layer = env.layers[i]

            # build state vector — same structure as training
            state = np.array([
                layer['importance'],
                layer['sparsity_ratio'],
                budget_remaining,
                running_ppl_hit,
                True if (i == 0 or i == len(env.layers) - 1) else False
            ], dtype=np.float32)

            # greedy action — no exploration
            action = trained_agent.act(state)

            # update context for next layer
            agg              = env.action_aggressiveness[action]
            ppl_hit          = agg * layer['importance'] * (1 - layer['sparsity_ratio'] / 2)
            running_ppl_hit += ppl_hit
            bits_after       = {0:16, 1:8, 2:4, 3:2, 4:0, 5:0, 6:0}[action]
            budget_remaining -= (16 - bits_after) / 16 / len(env.layers)
            actions.append(action_names[action])
            print(f"Layer {i:2d} | "
                f"importance={layer['importance']:.2f} | "
                f"sparsity={layer['sparsity_ratio']:.2f} | "
                f"ppl_hit={ppl_hit:.4f} | "
                f"action={action_names[action]}")

        print(f"\nFinal running PPL hit: {running_ppl_hit:.4f}")
        print(f"Budget remaining:      {budget_remaining:.4f}")
        print(actions)