"""FASE 1 - Q-Learning tabular desde cero en FrozenLake-v1 (8x8, slippery).
Ecuacion de Bellman:  Q(s,a) <- Q(s,a) + a*[r + g*max_a' Q(s',a') - Q(s,a)]
Genera q_learning_progress.png (media movil de recompensa) y evalua la politica.
"""
import numpy as np
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA, GAMMA = 0.1, 0.99
EPSILON_START, EPSILON_MIN, EPSILON_DECAY = 1.0, 0.01, 0.9995
MAX_STEPS, SEED = 200, 42


def train_q_learning(episodes=10000, seed=SEED):
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    n_states, n_actions = env.observation_space.n, env.action_space.n  # 64, 4
    q_table = np.zeros((n_states, n_actions))                         # Q-Table 64x4
    rng = np.random.default_rng(seed)
    epsilon, rewards = EPSILON_START, []

    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        total, steps = 0.0, 0
        while not (terminated or truncated) and steps < MAX_STEPS:
            if rng.random() < epsilon:                       # explorar
                action = int(rng.integers(n_actions))
            else:                                            # explotar
                best = np.flatnonzero(q_table[state] == q_table[state].max())
                action = int(rng.choice(best))
            nxt, reward, terminated, truncated, _ = env.step(action)
            td_target = reward + GAMMA * np.max(q_table[nxt]) * (not terminated)
            q_table[state, action] += ALPHA * (td_target - q_table[state, action])
            state, total, steps = nxt, total + reward, steps + 1
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        rewards.append(total)
        if (ep + 1) % 1000 == 0:
            print(f"Episodio {ep+1:>6} | eps={epsilon:.3f} | "
                  f"exito ult.1000 = {np.mean(rewards[-1000:]):.3f}")
    env.close()
    return q_table, rewards


def evaluate(q_table, episodes=1000):
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    wins = 0
    for ep in range(episodes):
        state, _ = env.reset(seed=10000 + ep)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < MAX_STEPS:
            state, reward, terminated, truncated, _ = env.step(int(np.argmax(q_table[state])))
            steps += 1
            if terminated and reward > 0:
                wins += 1
    env.close()
    print(f"Tasa de exito politica greedy: {wins/episodes:.3f} ({wins}/{episodes})")


if __name__ == "__main__":
    q_table, rewards = train_q_learning(10000)
    ma = np.convolve(rewards, np.ones(100) / 100, mode="valid")
    plt.figure(figsize=(10, 6))
    plt.plot(ma)
    plt.title("Progreso Q-Learning (FrozenLake-v1 8x8, slippery)")
    plt.xlabel("Episodio"); plt.ylabel("Tasa de exito (media movil 100)")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("q_learning_progress.png", dpi=120)
    print("Figura guardada: q_learning_progress.png")
    evaluate(q_table)
