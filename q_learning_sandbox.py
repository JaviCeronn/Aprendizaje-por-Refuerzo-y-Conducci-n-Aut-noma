"""
=====================================================================
 FASE 1 – Calentamiento: Q-Learning Tabular desde cero
=====================================================================
Entorno: FrozenLake-v1 (cuadrícula 8x8, resbaladiza / is_slippery=True).

Se implementa Q-Learning SIN librerías de Deep RL, demostrando la
Ecuación de Bellman:

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Política de exploración: epsilon-greedy con decaimiento exponencial.
Salida: gráfico de la media móvil de recompensa sobre 10.000 episodios
        guardado en 'q_learning_progress.png'.

Uso:
    python q_learning_sandbox.py
"""
from __future__ import annotations

import numpy as np
import gymnasium as gym
import matplotlib

matplotlib.use("Agg")  # backend sin ventana (sirve también en Colab/headless)
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Hiperparámetros
# ---------------------------------------------------------------------
EPISODES = 10_000          # nº de episodios de entrenamiento
ALPHA = 0.1                # tasa de aprendizaje
GAMMA = 0.99               # factor de descuento
EPSILON_START = 1.0        # exploración inicial (100 % aleatorio)
EPSILON_MIN = 0.01         # exploración mínima
EPSILON_DECAY = 0.9995     # decaimiento multiplicativo por episodio
MAX_STEPS = 200            # límite de pasos por episodio (anti-bucles)
SEED = 42


def train_q_learning(episodes: int = EPISODES, seed: int = SEED):
    """Entrena Q-Learning tabular en FrozenLake-v1 8x8 resbaladizo.

    Devuelve:
        q_table : np.ndarray de forma (n_estados, n_acciones)
        rewards : lista con la recompensa total por episodio
    """
    # 1) Inicializar el entorno (8x8, resbaladizo)
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)

    n_states = env.observation_space.n   # 64 estados
    n_actions = env.action_space.n       # 4 acciones (izq, abajo, der, arriba)

    # 2) Crear la Q-Table inicializada a ceros (64 x 4)
    q_table = np.zeros((n_states, n_actions), dtype=np.float64)

    # Generador de aleatoriedad reproducible para epsilon-greedy
    rng = np.random.default_rng(seed)

    epsilon = EPSILON_START
    rewards_per_episode: list[float] = []

    # 3) Bucle de entrenamiento
    for ep in range(episodes):
        state, _ = env.reset(seed=seed + ep)
        terminated = truncated = False
        total_reward = 0.0
        steps = 0

        while not (terminated or truncated) and steps < MAX_STEPS:
            # --- Política epsilon-greedy --------------------------------
            if rng.random() < epsilon:
                action = int(rng.integers(n_actions))          # explorar
            else:
                # explotar: romper empates aleatoriamente para no sesgar
                best = np.flatnonzero(q_table[state] == q_table[state].max())
                action = int(rng.choice(best))

            # --- Interacción con el entorno -----------------------------
            next_state, reward, terminated, truncated, _ = env.step(action)

            # --- Actualización de Bellman (Q-Learning, off-policy) ------
            best_next = np.max(q_table[next_state])
            td_target = reward + GAMMA * best_next * (not terminated)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += ALPHA * td_error

            state = next_state
            total_reward += reward
            steps += 1

        # 4) Decaimiento de epsilon (cada vez se explora un poco menos)
        epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)
        rewards_per_episode.append(total_reward)

        if (ep + 1) % 1000 == 0:
            recent = np.mean(rewards_per_episode[-1000:])
            print(f"Episodio {ep + 1:>6} | epsilon={epsilon:.3f} | "
                  f"tasa de éxito (últimos 1000) = {recent:.3f}")

    env.close()
    return q_table, rewards_per_episode


def moving_average(values, window: int = 100):
    """Media móvil simple (para suavizar la curva de recompensa)."""
    return np.convolve(values, np.ones(window) / window, mode="valid")


def plot_progress(rewards, window: int = 100, out_path: str = "q_learning_progress.png"):
    """Dibuja y guarda la media móvil de la recompensa por episodio."""
    ma = moving_average(rewards, window)
    plt.figure(figsize=(10, 6))
    plt.plot(ma, color="#1f77b4")
    plt.title(f"Progreso de Q-Learning en FrozenLake-v1 8x8 "
              f"(media móvil de {window} episodios)")
    plt.xlabel("Episodio")
    plt.ylabel(f"Tasa de éxito media (ventana={window})")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"[plot] Gráfico guardado en {out_path}")


def evaluate_policy(q_table, episodes: int = 1000, seed: int = SEED):
    """Evalúa la política greedy aprendida (sin exploración)."""
    env = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
    successes = 0
    for ep in range(episodes):
        state, _ = env.reset(seed=10_000 + seed + ep)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated) and steps < MAX_STEPS:
            action = int(np.argmax(q_table[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            steps += 1
            if terminated and reward > 0:
                successes += 1
    env.close()
    rate = successes / episodes
    print(f"[eval] Tasa de éxito de la política greedy: {rate:.3f} "
          f"({successes}/{episodes})")
    return rate


if __name__ == "__main__":
    print("=== FASE 1: Q-Learning tabular en FrozenLake-v1 (8x8, slippery) ===")
    q_table, rewards = train_q_learning()
    plot_progress(rewards)
    evaluate_policy(q_table)
    print("Listo. Q-Table aprendida con forma:", q_table.shape)
