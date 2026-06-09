"""
FASE 2 - Baselines DQN y PPO en Duckietown.

Se ejecuta como SCRIPT con Python 3.11 + xvfb-run, NO dentro del kernel de Jupyter:
gym-duckietown mantiene estado OpenGL GLOBAL por proceso y SEGFAULTEA el proceso si se
renderiza dentro del kernel del notebook (en Colab se ve como "restarting kernel"). Al
lanzarlo como proceso independiente con xvfb-run, cada simulador tiene su contexto GL y el
entrenamiento es estable; ademas SubprocVecEnv (spawn) puede reimportar este modulo .py.

Subcomandos (--algo):
    frame  captura un frame REAL de Duckietown y guarda el pipeline de vision
    dqn    entrena DQN (baseline discreto), guarda modelo + curva
    ppo    entrena PPO (baseline continuo), guarda modelo + curva
    eval   carga DQN y PPO ya entrenados, evalua, guarda grafica comparativa y best_agent.zip
    all    dqn + ppo + eval

Uso:  python src/train_fase2.py --algo dqn --timesteps 30000
"""
from __future__ import annotations
import argparse
import logging
import os
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"     # lo heredan los subprocesos 'spawn' de SubprocVecEnv
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)               # corte GLOBAL: el stack daffy no lo revierte

import numpy as np
import gym as old_gym                        # API antigua (gym-duckietown)
import gymnasium as gym                      # API moderna (Stable-Baselines3)
from gymnasium import spaces
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback

IMG_SIZE = 64
N_STACK = 4
OBS_SHAPE = (1, IMG_SIZE, IMG_SIZE)
SEED = 42
RESULTS = os.path.join("results", "Fase_2")
os.makedirs(RESULTS, exist_ok=True)

DQN_PATH = os.path.join(RESULTS, "dqn_duckie")
PPO_PATH = os.path.join(RESULTS, "ppo_duckie")

TRAIN_MAPS = [
    "Duckietown-loop_empty-v0",
    "Duckietown-udem1-v0",
    "Duckietown-zigzag_dists-v0",
    "Duckietown-small_loop-v0",
    "Duckietown-straight_road-v0",
]

DISCRETE_ACTIONS = np.array([
    [0.6, 0.6],    # 0 recto
    [0.35, 0.6],   # 1 giro suave izquierda
    [0.6, 0.35],   # 2 giro suave derecha
    [0.2, 0.6],    # 3 giro fuerte izquierda
    [0.6, 0.2],    # 4 giro fuerte derecha
], dtype=np.float32)


class DuckieWrapper(gym.Env):
    """Adapta gym-duckietown a Gymnasium: recorta cielo, gris, 64x64 -> (1,64,64)."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name="Duckietown-loop_empty-v0", seed=None):
        super().__init__()
        import gym_duckietown                 # registra los entornos al importar
        logging.disable(logging.INFO)         # el stack daffy re-activa sus loggers al importar
        self.env_name = env_name
        self.env = old_gym.make(env_name)
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=OBS_SHAPE, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        return self._process_obs(obs), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), float(reward), bool(done), False, info

    def _process_obs(self, obs):
        obs = obs[obs.shape[0] // 2:, :, :]                                   # recortar cielo
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)                          # escala de grises
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=0).astype(np.uint8)              # (1,64,64)

    def render(self):
        return self.env.render(mode="rgb_array")

    def close(self):
        self.env.close()


class DiscreteWrapper(gym.ActionWrapper):
    """Discretiza el espacio de accion continuo para DQN (5 acciones)."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

    def action(self, action):
        return DISCRETE_ACTIONS[int(action)]


class LaneFollowingReward(gym.Wrapper):
    """Reward shaping (solo entrenamiento): penaliza salirse (-10), premia avanzar (+0.1)."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = reward
        if terminated and reward < 0:
            shaped = -10.0
        shaped += 0.1
        return obs, shaped, terminated, truncated, info


class CustomCNN(BaseFeaturesExtractor):
    """CNN tipo Nature (Mnih 2015) adaptada a (4,64,64) con normalizacion de pixeles."""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations.float() / 255.0))


def make_env(map_name, discrete=False, shaping=True, seed=SEED):
    def _init():
        env = DuckieWrapper(map_name, seed=seed)
        if shaping:
            env = LaneFollowingReward(env)
        if discrete:
            env = DiscreteWrapper(env)
        return env
    return _init


def make_vec_env(maps, discrete=False, shaping=True, n_stack=N_STACK):
    # gym-duckietown guarda estado OpenGL GLOBAL por proceso: tener mas de un simulador en el
    # MISMO proceso (DummyVecEnv) provoca un Segmentation fault al renderizar el segundo
    # contexto. SubprocVecEnv aisla cada entorno en su propio proceso (un contexto GL
    # independiente); 'spawn' evita conflictos con el contexto CUDA del proceso padre.
    # Con un solo mapa basta DummyVecEnv.
    env_fns = [make_env(m, discrete=discrete, shaping=shaping, seed=SEED + i)
               for i, m in enumerate(maps)]
    if len(env_fns) == 1:
        vec = DummyVecEnv(env_fns)
    else:
        vec = SubprocVecEnv(env_fns, start_method="spawn")
    vec = VecFrameStack(vec, n_stack=n_stack)
    vec = VecMonitor(vec)
    return vec


def policy_kwargs(features_dim=256):
    return dict(features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=features_dim))


class RewardLogger(BaseCallback):
    """Recoge la recompensa de cada episodio (VecMonitor la pone en info['episode']['r'])."""
    def __init__(self):
        super().__init__(verbose=0)
        self.ep_rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.ep_rewards.append(info["episode"]["r"])
        return True


def plot_curve(rewards, color, title, filename):
    if not rewards:
        print(f"[aviso] sin episodios completos para {filename}")
        return
    w = min(20, len(rewards))
    ma = np.convolve(rewards, np.ones(w) / w, mode="valid")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ma, color=color)
    ax.axhline(ma[-1], color="tomato", linestyle="--", label=f"Final: {ma[-1]:.1f}")
    ax.set(xlabel="Episodio", ylabel=f"Recompensa (media movil {w})", title=title)
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    path = os.path.join(RESULTS, filename)
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"Guardado: {path}")


# ------------------------------------------------------------------ FRAME REAL
def capture_frame():
    """Captura 4 frames REALES de Duckietown y guarda el pipeline de vision."""
    import gym_duckietown  # noqa: F401
    env = old_gym.make("Duckietown-loop_empty-v0")
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    raw = [np.asarray(obs)]
    for _ in range(3):                                   # avanzar recto para 4 frames distintos
        out = env.step(np.array([0.5, 0.5], dtype=np.float32))
        o = out[0]
        raw.append(np.asarray(o))
    env.close()

    frame = raw[0]
    H = frame.shape[0]
    cropped = frame[H // 2:, :, :]
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    stack = []
    for fr in raw:
        c = fr[fr.shape[0] // 2:, :, :]
        g = cv2.cvtColor(c, cv2.COLOR_RGB2GRAY)
        stack.append(cv2.resize(g, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA))

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle("Pipeline de vision - frame REAL de Duckietown", fontweight="bold", fontsize=12)
    h0, w0 = frame.shape[0], frame.shape[1]
    steps = [f"1. Frame original\n{h0}x{w0}  RGB", f"2. Recorte 50% superior\n{h0//2}x{w0}  RGB",
             f"3. Escala de grises\n{h0//2}x{w0}  1 canal", "4. Resize 64x64\n64x64  1 canal"]
    imgs = [frame, cropped, gray, resized]
    cmaps = [None, None, "gray", "gray"]
    for i, (img, title, cmap) in enumerate(zip(imgs, steps, cmaps), 1):
        ax = fig.add_subplot(2, 5, i)
        ax.imshow(img, cmap=cmap); ax.set_title(title, fontsize=8); ax.axis("off")
    for j, f in enumerate(stack):
        ax = fig.add_subplot(2, 5, 6 + j)
        ax.imshow(f, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"5. Frame t-{3-j}  (stack {j+1}/4)\n64x64  1 canal", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(RESULTS, "vision_pipeline.png")
    plt.savefig(path, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Guardado: {path}")


# ------------------------------------------------------------------ ENTRENAMIENTO
def train_dqn(timesteps, device):
    print("\n=== Entrenando DQN (baseline discreto) ===")
    env = make_vec_env(TRAIN_MAPS, discrete=True, shaping=True)
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs(),
                learning_rate=1e-4, buffer_size=50_000, learning_starts=1_000,
                batch_size=64, gamma=0.99, train_freq=4, target_update_interval=1_000,
                exploration_fraction=0.2, exploration_final_eps=0.05,
                verbose=1, seed=SEED, device=device)
    cb = RewardLogger()
    model.learn(total_timesteps=timesteps, callback=cb)
    model.save(DQN_PATH); env.close()
    plot_curve(cb.ep_rewards, "steelblue", "DQN - curva de entrenamiento en Duckietown",
               "dqn_training.png")
    print(f"Modelo guardado: {DQN_PATH}.zip")


def train_ppo(timesteps, device):
    print("\n=== Entrenando PPO (baseline continuo) ===")
    env = make_vec_env(TRAIN_MAPS, discrete=False, shaping=True)
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs(),
                learning_rate=3e-4, n_steps=2_048, batch_size=256, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                verbose=1, seed=SEED, device=device)
    cb = RewardLogger()
    model.learn(total_timesteps=timesteps, callback=cb)
    model.save(PPO_PATH); env.close()
    plot_curve(cb.ep_rewards, "seagreen", "PPO - curva de entrenamiento en Duckietown",
               "ppo_training.png")
    print(f"Modelo guardado: {PPO_PATH}.zip")


def evaluate_agent(model, eval_maps, n_episodes=5, discrete=False):
    env = make_vec_env(eval_maps, discrete=discrete, shaping=False)
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = np.array([False])
        ep_ret, steps = 0.0, 0
        while not done[0] and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_ret += float(reward[0]); steps += 1
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns)), float(np.std(returns)), returns


def plot_comparison(dqn_stats, ppo_stats):
    dqn_mean, dqn_std, _ = dqn_stats
    ppo_mean, ppo_std, _ = ppo_stats
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["DQN (discreto)", "PPO (continuo)"], [dqn_mean, ppo_mean],
                  yerr=[dqn_std, ppo_std], capsize=8,
                  color=["steelblue", "seagreen"], alpha=0.85)
    for bar, m in zip(bars, [dqn_mean, ppo_mean]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(dqn_std, ppo_std) * 0.15 + 0.5,
                f"{m:.1f}", ha="center", va="bottom", fontweight="bold")
    ax.set(ylabel="Recompensa media (5 episodios)",
           title="DQN vs PPO - evaluacion en los 5 mapas de entrenamiento")
    ax.grid(axis="y", alpha=0.3); plt.tight_layout()
    path = os.path.join(RESULTS, "dqn_vs_ppo.png")
    plt.savefig(path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"Guardado: {path}")


def run_eval(device, n_episodes):
    print("\n=== Evaluacion comparativa (modo determinista) ===")
    dqn_model = DQN.load(DQN_PATH, device=device)
    ppo_model = PPO.load(PPO_PATH, device=device)
    dqn_stats = evaluate_agent(dqn_model, TRAIN_MAPS, n_episodes, discrete=True)
    ppo_stats = evaluate_agent(ppo_model, TRAIN_MAPS, n_episodes, discrete=False)
    plot_comparison(dqn_stats, ppo_stats)

    # mejor agente continuo -> entregable. Guardamos los dos nombres en uso:
    #   best_agent.zip        (requisitos de entrega / diapositiva 9)
    #   best_duckie_agent.zip (el que carga notebooks/eval.ipynb del profesor)
    ppo_model.save("best_agent")
    ppo_model.save("best_duckie_agent")
    print("\nbest_agent.zip y best_duckie_agent.zip guardados.")
    print(f"DQN: {dqn_stats[0]:.2f} +/- {dqn_stats[1]:.2f} | retornos: {[round(x,1) for x in dqn_stats[2]]}")
    print(f"PPO: {ppo_stats[0]:.2f} +/- {ppo_stats[1]:.2f} | retornos: {[round(x,1) for x in ppo_stats[2]]}")


def main():
    parser = argparse.ArgumentParser(description="Fase 2 - DQN y PPO en Duckietown.")
    parser.add_argument("--algo", choices=["frame", "dqn", "ppo", "eval", "all"], default="all")
    parser.add_argument("--timesteps", type=int, default=30_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)

    if args.algo == "frame":
        capture_frame()
        return
    if args.algo in ("dqn", "all"):
        train_dqn(args.timesteps, device)
    if args.algo in ("ppo", "all"):
        train_ppo(args.timesteps, device)
    if args.algo in ("eval", "all"):
        run_eval(device, args.eval_episodes)


if __name__ == "__main__":
    main()
