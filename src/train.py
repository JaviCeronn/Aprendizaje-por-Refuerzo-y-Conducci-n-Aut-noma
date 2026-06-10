"""
FASES 2 y 3 - Pipeline de entrenamiento (DQN, PPO y SAC) en Duckietown.

IMPORTANTE sobre el espacio de acciones: los entornos Duckietown-*-v0 de
gym-duckietown@daffy apuntan a DuckietownEnv, cuyo step() interpreta la
accion como [velocidad, angulo_de_giro] (NO como velocidades de rueda).
DISCRETE_ACTIONS y SpeedSteerWrapper estan definidos en esas coordenadas.

Uso:
    python train.py --algo all --timesteps 200000 --n-envs 5 --results out/
"""
from __future__ import annotations
import argparse
import io as _io
import logging
import os
import pathlib
import sys
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

import numpy as np
import gym as old_gym
import gymnasium as gym
from gymnasium import spaces
import cv2
import torch
import torch.nn as nn
from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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

# Accion = [velocidad, angulo de giro] (semantica de DuckietownEnv).
# Set simetrico: recto + giro suave/cerrado a cada lado.
DISCRETE_ACTIONS = np.array([
    [0.44,  0.0],   # recto
    [0.30,  2.0],   # giro izquierda suave
    [0.30, -2.0],   # giro derecha suave
    [0.20,  5.0],   # giro izquierda cerrado
    [0.20, -5.0],   # giro derecha cerrado
], dtype=np.float32)


def short_map(name):
    return name.replace("Duckietown-", "").replace("-v0", "")


class DuckieWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name="Duckietown-loop_empty-v0", seed=None):
        super().__init__()
        # gym_duckietown imprime pyglet.options al importarse; suprimir stdout
        _out, sys.stdout = sys.stdout, _io.StringIO()
        try:
            import gym_duckietown
        finally:
            sys.stdout = _out
        logging.disable(logging.INFO)
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
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

    def action(self, action):
        return DISCRETE_ACTIONS[int(action)]


class SpeedSteerWrapper(gym.ActionWrapper):
    """Para PPO/SAC: a = [v, s] en [-1,1]^2 -> [velocidad, angulo].
    La velocidad queda siempre en [0.10, 0.40] (nunca marcha atras), de modo
    que la politica inicial (media gaussiana en 0) ya avanza recto y solo
    tiene que aprender a dirigir. El angulo en [-5, 5] permite curvas
    cerradas, imposibles con el Box [-1,1] original."""

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32), dtype=np.float32)

    def action(self, a):
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        vel = 0.25 + 0.15 * float(np.clip(a[0], -1.0, 1.0))
        angle = 5.0 * float(np.clip(a[1], -1.0, 1.0))
        return np.array([vel, angle], dtype=np.float32)


class LaneFollowingReward(gym.Wrapper):
    """Recompensa densa basada en la posicion real en el carril, leida del
    simulador con get_lane_pos2 (distancia al centro y alineacion):

      en carretera:  0.5 + 4*velocidad*alineacion + 1.5*centrado*alineacion
      salida/choque: -20  (reward nativo -1000)
      max_steps:       0  (terminar por tiempo no se castiga)

    Cuanto mas tiempo avance centrado y alineado, mas recompensa acumula."""

    LANE_HALF = 0.10  # ~medio ancho de carril (m); a esta distancia centrado=0

    def __init__(self, env):
        super().__init__(env)
        e = env
        while not isinstance(e, DuckieWrapper):
            e = e.env
        self._sim = e.env.unwrapped  # DuckietownEnv (hereda de Simulator)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if terminated:
            shaped = -20.0 if reward < -100.0 else 0.0
            return obs, shaped, terminated, truncated, info
        try:
            lp = self._sim.get_lane_pos2(self._sim.cur_pos, self._sim.cur_angle)
            align = max(0.0, float(lp.dot_dir))
            center = max(0.0, 1.0 - abs(float(lp.dist)) / self.LANE_HALF)
            speed = max(0.0, float(self._sim.speed))
            shaped = 0.5 + 4.0 * speed * align + 1.5 * center * align
        except Exception:
            shaped = 0.0  # fuera del carril (p.ej. cruce) pero sin terminar
        return obs, shaped, terminated, truncated, info


class CustomCNN(BaseFeaturesExtractor):
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
        # normalize_images=False en policy_kwargs: SB3 no divide entre 255,
        # la normalizacion se hace una sola vez aqui.
        return self.linear(self.cnn(observations.float() / 255.0))


class OverwriteCheckpointCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq: int = 20_000):
        super().__init__(verbose=0)
        self.save_path = save_path
        self.save_freq = save_freq
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._last_save = self.num_timesteps
            self.model.save(self.save_path)
            print(f"\n[checkpoint] {self.num_timesteps:,} steps -> {self.save_path}.zip",
                  flush=True)
        return True


class BestModelCallback(BaseCallback):
    def __init__(self, save_path: str, min_episodes: int = 4):
        super().__init__(verbose=0)
        self.save_path = save_path
        self.min_episodes = min_episodes
        self._best_mean = -np.inf

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) >= self.min_episodes:
            mean_rew = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
            if mean_rew > self._best_mean:
                self._best_mean = mean_rew
                self.model.save(self.save_path)
                print(f"\n[best] {self.num_timesteps:,} steps  ep_rew_mean={mean_rew:.1f}"
                      f" -> {self.save_path}.zip", flush=True)
        return True


class EpisodeRewardRecorder(BaseCallback):
    def __init__(self, env_maps=None):
        super().__init__(verbose=0)
        self.episode_rewards = []
        self.env_maps = env_maps or []
        self.per_map = {}

    def _on_step(self) -> bool:
        for i, info in enumerate(self.locals.get("infos", [])):
            ep = info.get("episode") if isinstance(info, dict) else None
            if ep is not None:
                r = float(ep["r"])
                self.episode_rewards.append(r)
                if i < len(self.env_maps):
                    self.per_map.setdefault(self.env_maps[i], []).append(r)
        return True


class LossRecorder(BaseCallback):
    def __init__(self):
        super().__init__(verbose=0)
        self.losses = []
        self._last_n_updates = -1

    def _on_step(self) -> bool:
        n_upd = getattr(self.model, "_n_updates", None)
        if n_upd is not None and n_upd != self._last_n_updates:
            self._last_n_updates = n_upd
            loss = self.model.logger.name_to_value.get("train/loss")
            if loss is not None:
                self.losses.append([int(self.num_timesteps), float(loss)])
        return True


def make_env(map_name, discrete=False, shaping=True, seed=SEED):
    def _init():
        env = DuckieWrapper(map_name, seed=seed)
        if shaping:
            env = LaneFollowingReward(env)
        if discrete:
            env = DiscreteWrapper(env)
        else:
            env = SpeedSteerWrapper(env)
        return env
    return _init


def resolve_maps(maps, n_envs=None):
    if n_envs is None:
        n_envs = len(maps)
    return [maps[i % len(maps)] for i in range(n_envs)]


def make_vec_env(maps, discrete=False, shaping=True, n_stack=N_STACK, n_envs=None):
    env_maps = resolve_maps(maps, n_envs)
    env_fns = [make_env(m, discrete=discrete, shaping=shaping, seed=SEED + i)
               for i, m in enumerate(env_maps)]
    if len(env_fns) == 1:
        vec = DummyVecEnv(env_fns)
    else:
        vec = SubprocVecEnv(env_fns, start_method="spawn")
    vec = VecFrameStack(vec, n_stack=n_stack)
    vec = VecMonitor(vec)
    return vec


def policy_kwargs(features_dim=256):
    # normalize_images=False: la normalizacion /255 la hace CustomCNN.forward,
    # asi evitamos que SB3 vuelva a dividir entre 255 (doble normalizacion).
    return dict(features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=features_dim),
                normalize_images=False)


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
            ep_ret += float(reward[0])
            steps += 1
        returns.append(ep_ret)
    env.close()
    return {"mean_reward": float(np.mean(returns)), "std_reward": float(np.std(returns)),
            "returns": returns}


def train_dqn(timesteps, device="auto", results_dir=None, n_envs=None):
    print("\n=== Entrenando DQN (baseline discreto) ===", flush=True)
    env = make_vec_env(TRAIN_MAPS, discrete=True, shaping=True, n_envs=n_envs)
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs(),
                learning_rate=1e-4, buffer_size=100_000, learning_starts=10_000,
                batch_size=64, gamma=0.99, train_freq=4, target_update_interval=1_000,
                exploration_fraction=0.4, exploration_final_eps=0.05,
                verbose=1, seed=SEED, device=device)
    save_name = str(results_dir / "dqn_duckie_v2") if results_dir else "dqn_duckie_v2_agent"
    best_name = str(results_dir / "dqn_duckie_v2_best") if results_dir else "dqn_duckie_v2_best"
    rew_rec = EpisodeRewardRecorder(env_maps=resolve_maps(TRAIN_MAPS, n_envs))
    loss_rec = LossRecorder()
    model.learn(total_timesteps=timesteps,
                callback=[
                    OverwriteCheckpointCallback(save_path=save_name, save_freq=20_000),
                    BestModelCallback(save_path=best_name),
                    rew_rec, loss_rec,
                ])
    model.save(save_name)
    print(f"\nDQN guardado en {save_name}.zip", flush=True)
    env.close()
    return model, {"ep_rewards": rew_rec.episode_rewards,
                   "losses": loss_rec.losses, "per_map": rew_rec.per_map}


def train_ppo(timesteps, device="auto", results_dir=None, n_envs=None):
    print("\n=== Entrenando PPO (baseline continuo) ===", flush=True)
    env = make_vec_env(TRAIN_MAPS, discrete=False, shaping=True, n_envs=n_envs)
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs(),
                learning_rate=3e-4, n_steps=512, batch_size=128, n_epochs=10,
                gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01,
                verbose=1, seed=SEED, device=device)
    save_name = str(results_dir / "ppo_duckie_v2") if results_dir else "ppo_duckie_v2_agent"
    best_name = str(results_dir / "ppo_duckie_v2_best") if results_dir else "ppo_duckie_v2_best"
    rew_rec = EpisodeRewardRecorder(env_maps=resolve_maps(TRAIN_MAPS, n_envs))
    loss_rec = LossRecorder()
    model.learn(total_timesteps=timesteps,
                callback=[
                    OverwriteCheckpointCallback(save_path=save_name, save_freq=20_000),
                    BestModelCallback(save_path=best_name),
                    rew_rec, loss_rec,
                ])
    model.save(save_name)
    print(f"\nPPO guardado en {save_name}.zip", flush=True)
    env.close()
    return model, {"ep_rewards": rew_rec.episode_rewards,
                   "losses": loss_rec.losses, "per_map": rew_rec.per_map}


def train_sac(timesteps, curriculum=False, device="auto", results_dir=None, n_envs=None):
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
        n_par = n_envs if n_envs else len(TRAIN_MAPS)
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
        env = make_vec_env(TRAIN_MAPS, discrete=False, shaping=True, n_envs=n_envs)
        model = build(env)
        model.learn(total_timesteps=timesteps)
        env.close()

    save_name = str(results_dir / "sac_duckie") if results_dir else "sac_duckie_agent"
    model.save(save_name)
    return model


def _new_ax(figsize=(10, 4)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt, plt.subplots(figsize=figsize)


def _save_curve(ep_rewards, label, color, results_dir, algo):
    import json
    rd = pathlib.Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    json.dump(ep_rewards, open(rd / f"{algo}_ep_rewards.json", "w"))
    if not ep_rewards:
        return
    w = min(20, len(ep_rewards))
    ma = np.convolve(ep_rewards, np.ones(w) / w, mode="valid")
    plt, (fig, ax) = _new_ax()
    ax.plot(ma, color=color)
    ax.axhline(ma[-1], color="tomato", linestyle="--", label=f"Final: {ma[-1]:.1f}")
    ax.set(xlabel="Episodio", ylabel=f"Recompensa (media movil {w})",
           title=f"{label} — curva de entrenamiento en Duckietown")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(rd / f"{algo}_training.png", dpi=120, bbox_inches="tight")
    plt.close()


def _save_loss_curve(losses, label, color, results_dir, algo):
    import json
    rd = pathlib.Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    json.dump(losses, open(rd / f"{algo}_loss.json", "w"))
    if not losses:
        return
    xs = [p[0] for p in losses]
    ys = [p[1] for p in losses]
    w = min(20, len(ys))
    ma = np.convolve(ys, np.ones(w) / w, mode="valid")
    plt, (fig, ax) = _new_ax()
    ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.8)
    ax.plot(xs[w - 1:], ma, color=color, linewidth=1.8, label=f"{label} (media movil {w})")
    ax.set(xlabel="Timesteps", ylabel="train/loss",
           title=f"{label} — loss de entrenamiento (global)")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(rd / f"{algo}_loss.png", dpi=120, bbox_inches="tight")
    plt.close()


def _save_permap(per_map, label, results_dir, algo):
    import json
    rd = pathlib.Path(results_dir)
    rd.mkdir(parents=True, exist_ok=True)
    short = {short_map(k): v for k, v in per_map.items()}
    json.dump(short, open(rd / f"{algo}_per_map_rewards.json", "w"))
    if not short:
        return
    plt, (fig, ax) = _new_ax(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    for idx, (m, rews) in enumerate(sorted(short.items())):
        if not rews:
            continue
        w = min(10, len(rews))
        ma = np.convolve(rews, np.ones(w) / w, mode="valid")
        ax.plot(ma, color=cmap(idx % 10), linewidth=1.5, label=f"{m} (n={len(rews)})")
    ax.set(xlabel="Episodio (por mapa)", ylabel="Recompensa (media movil)",
           title=f"{label} — recompensa por escenario (mapa)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(rd / f"{algo}_per_map.png", dpi=120, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento RL Duckietown.")
    parser.add_argument("--algo", choices=["dqn", "ppo", "sac", "all"], default="all")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--results", type=str, default=None)
    parser.add_argument("--n-envs", type=int, default=len(TRAIN_MAPS),
                        help="Entornos en paralelo (SubprocVecEnv) para saturar CPU y alimentar la GPU.")
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results) if args.results else None
    if results_dir:
        results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train.py] Dispositivo: {device}")
    print(f"[train.py] Entornos en paralelo: {args.n_envs}")

    try:
        from pyvirtualdisplay import Display
        Display(visible=False, size=(1024, 768)).start()
        print("[train.py] Pantalla virtual (Xvfb) iniciada.")
    except Exception as e:
        print(f"[train.py] Sin pantalla virtual: {e}")

    trained = {}
    hist_map = {}

    if args.algo in ("dqn", "all"):
        model, hist = train_dqn(args.timesteps, device, results_dir, n_envs=args.n_envs)
        trained["DQN"] = (model, True)
        hist_map["dqn"] = hist

    if args.algo in ("ppo", "all"):
        model, hist = train_ppo(args.timesteps, device, results_dir, n_envs=args.n_envs)
        trained["PPO"] = (model, False)
        hist_map["ppo"] = hist

    if args.algo in ("sac", "all"):
        trained["SAC"] = (train_sac(args.timesteps, args.curriculum, device, results_dir,
                                    n_envs=args.n_envs), False)

    if results_dir:
        for algo, hist in hist_map.items():
            label = "DQN" if algo == "dqn" else "PPO"
            color = "steelblue" if algo == "dqn" else "seagreen"
            _save_curve(hist["ep_rewards"], label, color, results_dir, algo)
            _save_loss_curve(hist["losses"], label, color, results_dir, algo)
            _save_permap(hist["per_map"], label, results_dir, algo)

    print("\n=== Evaluacion comparativa ===")
    eval_results = {}
    for name, (model, disc) in trained.items():
        res = evaluate_agent(model, TRAIN_MAPS, n_episodes=args.eval_episodes, discrete=disc)
        eval_results[name] = res
        print(f"{name:>4}: recompensa media = {res['mean_reward']:.2f} +/- {res['std_reward']:.2f}")

    continuous = {n: r for n, r in eval_results.items() if n in ("PPO", "SAC")}
    best = max(continuous, key=lambda n: continuous[n]["mean_reward"]) if continuous \
        else max(eval_results, key=lambda n: eval_results[n]["mean_reward"])
    best_path = str(results_dir / "best_agent_v2") if results_dir else "best_duckie_agent_v2"
    trained[best][0].save(best_path)
    print(f"\n>>> Mejor agente: {best} -> {best_path}.zip")


if __name__ == "__main__":
    main()
