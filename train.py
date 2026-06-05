"""
FASES 2 y 3 - Pipeline de entrenamiento (DQN, PPO y SAC) en Duckietown.
Define DuckieWrapper, DiscreteWrapper, LaneFollowingReward y CustomCNN;
entrena DQN, PPO y SAC (con curriculum), evalua y guarda best_duckie_agent.zip.

Uso:
    python train.py --algo all --timesteps 200000 --curriculum
    python train.py --algo sac --timesteps 300000 --curriculum
"""
from __future__ import annotations
import argparse
import logging
import os
import warnings

# Silencio del ruido del stack daffy / gym antiguo (NO afecta a la tabla de SB3,
# que se imprime por su propio canal):
#  - PYTHONWARNINGS lo heredan los subprocesos 'spawn' (donde se crean los envs).
#  - logging.disable() es un corte GLOBAL que las libs no revierten con setLevel().
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

import numpy as np
import gym as old_gym                 # API antigua (gym-duckietown)
import gymnasium as gym               # API moderna (Stable-Baselines3)
from gymnasium import spaces
import cv2
import torch
import torch.nn as nn
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# --- Compatibilidad gym 0.25.2 -----------------------------------------
# gym 0.25.2 expone np_random como np.random.Generator (con .integers()/.random()
# nativos), que es justo lo que usa gym-duckietown: no hace falta ningun parche.

IMG_SIZE = 64
N_STACK = 4
OBS_SHAPE = (1, IMG_SIZE, IMG_SIZE)
SEED = 42

TRAIN_MAPS = [
    "Duckietown-loop_empty-v0",
    "Duckietown-udem1-v0",
    "Duckietown-zigzag_dists-v0",
    "Duckietown-small_loop-v0",
    "Duckietown-straight_road-v0",
]

DISCRETE_ACTIONS = np.array([
    [0.6, 0.6],    # 0 recto
    [0.35, 0.6],   # 1 giro suave izq
    [0.6, 0.35],   # 2 giro suave der
    [0.2, 0.6],    # 3 giro fuerte izq
    [0.6, 0.2],    # 4 giro fuerte der
], dtype=np.float32)


class DuckieWrapper(gym.Env):
    """Adapta gym-duckietown a Gymnasium: recorta cielo, gris, 64x64 -> (1,64,64)."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name="Duckietown-loop_empty-v0", seed=None):
        super().__init__()
        import gym_duckietown  # registra los entornos al importar
        logging.disable(logging.INFO)  # el stack daffy re-activa sus loggers al importar
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
        obs = obs[obs.shape[0] // 2:, :, :]
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=0).astype(np.uint8)

    def render(self):
        return self.env.render(mode="rgb_array")

    def close(self):
        self.env.close()


class DiscreteWrapper(gym.ActionWrapper):
    """Discretiza el espacio de accion continuo para DQN."""
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

    def action(self, action):
        return DISCRETE_ACTIONS[int(action)]


class LaneFollowingReward(gym.Wrapper):
    """Reward shaping (solo entrenamiento): penaliza salirse, premia avanzar."""
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = reward
        if terminated and reward < 0:
            shaped = -10.0
        shaped += 0.1
        return obs, shaped, terminated, truncated, info


class CustomCNN(BaseFeaturesExtractor):
    """CNN tipo Nature adaptada a (4,64,64) con normalizacion de pixeles."""
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
        observations = observations.float() / 255.0
        return self.linear(self.cnn(observations))


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
    env_fns = [make_env(m, discrete=discrete, shaping=shaping, seed=SEED + i)
               for i, m in enumerate(maps)]
    # gym-duckietown guarda estado OpenGL GLOBAL por proceso: tener mas de un
    # simulador en el MISMO proceso (DummyVecEnv) provoca un Segmentation fault al
    # renderizar el segundo contexto. SubprocVecEnv aisla cada entorno en su propio
    # proceso (un contexto GL independiente cada uno); 'spawn' evita conflictos con
    # el contexto CUDA del proceso padre. Con un solo mapa basta DummyVecEnv.
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
    return {"mean_reward": float(np.mean(returns)), "std_reward": float(np.std(returns)),
            "returns": returns}


def train_dqn(timesteps, device="auto"):
    print("\n=== Entrenando DQN (baseline discreto) ===")
    env = make_vec_env(TRAIN_MAPS, discrete=True, shaping=True)
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs(),
                learning_rate=1e-4, buffer_size=50_000, learning_starts=1_000,
                batch_size=64, gamma=0.99, train_freq=4, target_update_interval=1_000,
                exploration_fraction=0.2, exploration_final_eps=0.05,
                verbose=1, seed=SEED, device=device)
    model.learn(total_timesteps=timesteps)
    model.save("dqn_duckie_agent"); env.close()
    return model


def train_ppo(timesteps, device="auto"):
    print("\n=== Entrenando PPO (baseline continuo) ===")
    env = make_vec_env(TRAIN_MAPS, discrete=False, shaping=True)
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs(),
                learning_rate=3e-4, n_steps=2_048, batch_size=256, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                verbose=1, seed=SEED, device=device)
    model.learn(total_timesteps=timesteps)
    model.save("ppo_duckie_agent"); env.close()
    return model


def train_sac(timesteps, curriculum=False, device="auto"):
    """FASE 3: SAC off-policy de control continuo, con curriculum opcional."""
    print("\n=== Entrenando SAC (Fase 3) ===")

    def build(env):
        return SAC("CnnPolicy", env, policy_kwargs=policy_kwargs(),
                   learning_rate=3e-4, buffer_size=100_000, learning_starts=1_000,
                   batch_size=256, tau=0.005, gamma=0.99, train_freq=1,
                   gradient_steps=1, ent_coef="auto",
                   verbose=1, seed=SEED, device=device)

    if curriculum:
        stages = [
            ["Duckietown-straight_road-v0"],
            ["Duckietown-small_loop-v0", "Duckietown-loop_empty-v0"],
            TRAIN_MAPS,
        ]
        # nº FIJO de entornos en todas las etapas: SB3 exige el mismo n_envs en
        # set_env entre etapas. Rellenamos las ranuras ciclando los mapas de la etapa.
        n_par = len(TRAIN_MAPS)
        model = None
        per_stage = max(1, timesteps // len(stages))
        for i, maps in enumerate(stages, 1):
            stage_maps = [maps[j % len(maps)] for j in range(n_par)]
            print(f"\n--- Curriculum etapa {i}/{len(stages)}: {stage_maps} ---")
            env = make_vec_env(stage_maps, discrete=False, shaping=True)
            if model is None:
                model = build(env)
            else:
                model.set_env(env)
            model.learn(total_timesteps=per_stage, reset_num_timesteps=False)
            env.close()
    else:
        env = make_vec_env(TRAIN_MAPS, discrete=False, shaping=True)
        model = build(env)
        model.learn(total_timesteps=timesteps)
        env.close()

    model.save("sac_duckie_agent")
    return model


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento RL Duckietown.")
    parser.add_argument("--algo", choices=["dqn", "ppo", "sac", "all"], default="all")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Dispositivo:", device)

    # Pantalla virtual para el render OpenGL headless.
    try:
        from pyvirtualdisplay import Display
        Display(visible=False, size=(1024, 768)).start()
        print("[info] Pantalla virtual iniciada.")
    except Exception as e:
        print(f"[aviso] Sin pantalla virtual: {e}")

    trained, results = {}, {}
    if args.algo in ("dqn", "all"):
        trained["DQN"] = (train_dqn(args.timesteps, device), True)
    if args.algo in ("ppo", "all"):
        trained["PPO"] = (train_ppo(args.timesteps, device), False)
    if args.algo in ("sac", "all"):
        trained["SAC"] = (train_sac(args.timesteps, args.curriculum, device), False)

    print("\n=== Evaluacion comparativa ===")
    for name, (model, disc) in trained.items():
        res = evaluate_agent(model, TRAIN_MAPS, n_episodes=args.eval_episodes, discrete=disc)
        results[name] = res
        print(f"{name:>4}: recompensa media = {res['mean_reward']:.2f} +/- {res['std_reward']:.2f}")

    # Guardar el mejor modelo CONTINUO (compatible con la evaluacion del profesor).
    continuous = {n: r for n, r in results.items() if n in ("PPO", "SAC")}
    best = max(continuous, key=lambda n: continuous[n]["mean_reward"]) if continuous \
        else max(results, key=lambda n: results[n]["mean_reward"])
    trained[best][0].save("best_duckie_agent")
    print(f"\n>>> Mejor agente continuo: {best} -> best_duckie_agent.zip")

    print("\n=== RESUMEN ===")
    for name, r in results.items():
        print(f"{name}: {r['mean_reward']:.2f} +/- {r['std_reward']:.2f} "
              f"| returns={[round(x,1) for x in r['returns']]}")


if __name__ == "__main__":
    main()
