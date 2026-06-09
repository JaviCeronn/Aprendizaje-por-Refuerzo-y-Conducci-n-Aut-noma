"""
FASE 3 - SAC (Soft Actor-Critic) en Duckietown.

Se ejecuta como SCRIPT con Python 3.11 + xvfb-run, NO dentro del kernel de Jupyter:
gym-duckietown mantiene estado OpenGL GLOBAL por proceso y SEGFAULTEA si se renderiza
dentro del kernel del notebook. SubprocVecEnv aisla cada simulador en su propio proceso
(un contexto GL independiente) y, de paso, PARALELIZA los escenarios: con N_ENVS mapas
corriendo a la vez se recogen N_ENVS transiciones por paso de vector, por lo que los
mismos timesteps se completan ~N_ENVS veces mas rapido en tiempo de pared.

Por que SAC es el algoritmo avanzado de la Fase 3 (no es DQN ni PPO):
  - off-policy: reutiliza experiencia pasada via replay buffer (eficiencia de muestras,
    como DQN), decisivo cuando cada paso renderiza una escena 3D cara.
  - control continuo nativo: actua sobre las velocidades de rueda sin discretizar (como
    PPO), permitiendo giros suaves y precisos.
  - maxima entropia: premia politicas tan aleatorias como sea posible mientras sigan
    teniendo exito; con temperatura automatica (ent_coef="auto") explora mejor y produce
    politicas mas robustas ante estados no vistos (el mapa oculto de evaluacion).

Uso:
    python src/train_fase3.py --timesteps 300000 --n-envs 5
    python src/train_fase3.py --timesteps 300000 --n-envs 5 --curriculum
"""
from __future__ import annotations
import argparse
import io as _io
import json
import logging
import os
import pathlib
import sys
import warnings

# Silencio del ruido del stack daffy / gym antiguo (NO afecta a la tabla de SB3):
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
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


def short_map(name):
    return name.replace("Duckietown-", "").replace("-v0", "")


# ----------------------------------------------------------------------- WRAPPERS
class DuckieWrapper(gym.Env):
    """Adapta gym-duckietown a Gymnasium: recorta cielo, gris, 64x64 -> (1,64,64)."""
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name="Duckietown-loop_empty-v0", seed=None):
        super().__init__()
        # gym_duckietown imprime pyglet.options al importarse; suprimir stdout.
        _out, sys.stdout = sys.stdout, _io.StringIO()
        try:
            import gym_duckietown  # registra los entornos al importar
        finally:
            sys.stdout = _out
        logging.disable(logging.INFO)   # el stack daffy re-activa sus loggers al importar
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


class LaneFollowingReward(gym.Wrapper):
    """Reward shaping (SOLO entrenamiento): penaliza salirse (-10), premia avanzar (+0.1)."""
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
        # normalize_images=False en policy_kwargs: SB3 no divide entre 255, asi que
        # normalizamos aqui una sola vez (evita la doble normalizacion). El contrato
        # de eval.ipynb usa exactamente esta misma CNN.
        return self.linear(self.cnn(observations.float() / 255.0))


# --------------------------------------------------------------------- CALLBACKS
class OverwriteCheckpointCallback(BaseCallback):
    """Guarda el modelo cada save_freq steps sobreescribiendo siempre el mismo .zip."""
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
    """Guarda el modelo cuando ep_rew_mean supera el mejor visto hasta ahora."""
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
    """Acumula la recompensa total de cada episodio terminado (via VecMonitor) y la
    desglosa POR ESCENARIO usando el indice del entorno paralelo -> mapa. env_maps es
    reasignable entre etapas del curriculum para mantener el mapeo correcto."""
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


class SacMetricsRecorder(BaseCallback):
    """Captura las metricas internas de SAC cada vez que hace una actualizacion de
    gradiente (se leen del logger de SB3): critic_loss, actor_loss y la temperatura de
    entropia ent_coef (que SAC ajusta automaticamente). No son separables por mapa
    porque cada batch mezcla transiciones de los 5 mapas del replay buffer."""
    def __init__(self):
        super().__init__(verbose=0)
        self.critic_loss = []   # [[timestep, valor], ...]
        self.actor_loss = []
        self.ent_coef = []
        self._last_n_updates = -1

    def _on_step(self) -> bool:
        n_upd = getattr(self.model, "_n_updates", None)
        if n_upd is not None and n_upd != self._last_n_updates:
            self._last_n_updates = n_upd
            nv = self.model.logger.name_to_value
            t = int(self.num_timesteps)
            for key, store in (("train/critic_loss", self.critic_loss),
                               ("train/actor_loss", self.actor_loss),
                               ("train/ent_coef", self.ent_coef)):
                val = nv.get(key)
                if val is not None:
                    store.append([t, float(val)])
        return True


# --------------------------------------------------------------------- ENTORNOS
def make_env(map_name, shaping=True, seed=SEED):
    def _init():
        env = DuckieWrapper(map_name, seed=seed)
        if shaping:
            env = LaneFollowingReward(env)
        return env
    return _init


def resolve_maps(maps, n_envs=None):
    """Lista de mapas efectivamente asignada a cada entorno paralelo. Si n_envs > len(maps)
    se reparten ciclicamente para saturar la CPU y alimentar la GPU."""
    if n_envs is None:
        n_envs = len(maps)
    return [maps[i % len(maps)] for i in range(n_envs)]


def make_vec_env(maps, shaping=True, n_stack=N_STACK, n_envs=None):
    # gym-duckietown guarda estado OpenGL GLOBAL por proceso: mas de un simulador en el
    # MISMO proceso (DummyVecEnv) provoca Segmentation fault al renderizar el 2o contexto.
    # SubprocVecEnv aisla cada entorno en su propio proceso; 'spawn' evita conflictos con
    # el contexto CUDA del proceso padre. Con un solo mapa basta DummyVecEnv.
    env_maps = resolve_maps(maps, n_envs)
    env_fns = [make_env(m, shaping=shaping, seed=SEED + i) for i, m in enumerate(env_maps)]
    if len(env_fns) == 1:
        vec = DummyVecEnv(env_fns)
    else:
        vec = SubprocVecEnv(env_fns, start_method="spawn")
    vec = VecFrameStack(vec, n_stack=n_stack)
    vec = VecMonitor(vec)
    return vec


def policy_kwargs(features_dim=256):
    # normalize_images=False: la normalizacion /255 la hace CustomCNN.forward, asi
    # evitamos que SB3 vuelva a dividir entre 255 (doble normalizacion).
    return dict(features_extractor_class=CustomCNN,
                features_extractor_kwargs=dict(features_dim=features_dim),
                normalize_images=False)


# ------------------------------------------------------------------ EVALUACION
def evaluate_agent(model, eval_maps, n_episodes=3, max_total_steps=300_000):
    """Evalua en paralelo (un entorno por mapa, sin reward shaping). Lee el retorno real
    de cada episodio desde info['episode']['r'] de VecMonitor."""
    env = make_vec_env(eval_maps, shaping=False, n_envs=len(eval_maps))
    n = env.num_envs
    counts = np.zeros(n, dtype=int)
    returns = [[] for _ in range(n)]
    obs = env.reset()
    total = 0
    while counts.min() < n_episodes and total < max_total_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, _done, infos = env.step(action)
        total += 1
        for i, info in enumerate(infos):
            ep = info.get("episode") if isinstance(info, dict) else None
            if ep is not None:
                returns[i].append(float(ep["r"]))
                counts[i] += 1
    env.close()
    flat = [r for sub in returns for r in sub]
    per_map = {short_map(eval_maps[i]): returns[i] for i in range(n)}
    return {"mean_reward": float(np.mean(flat)) if flat else 0.0,
            "std_reward": float(np.std(flat)) if flat else 0.0,
            "returns": flat, "per_map_eval": per_map}


# ------------------------------------------------------------------ ENTRENAMIENTO
def train_sac(timesteps, n_envs, curriculum=False, device="auto",
              models_dir=None, buffer_size=100_000, learning_starts=5_000):
    """FASE 3: SAC off-policy de control continuo con maxima entropia. Paraleliza los
    escenarios con SubprocVecEnv (n_envs) para entrenar mas rapido."""
    print("\n=== Entrenando SAC (Fase 3) ===", flush=True)

    save_name = str(models_dir / "sac_duckie_v2") if models_dir else "sac_duckie_v2_agent"
    best_name = str(models_dir / "sac_duckie_v2_best") if models_dir else "sac_duckie_v2_best"

    # gradient_steps = n_envs para mantener la ratio updates:transiciones en 1:1 al
    # paralelizar (train_freq=1 paso de vector recoge n_envs transiciones por ciclo).
    grad_steps = max(1, n_envs)

    def build(env):
        return SAC("CnnPolicy", env, policy_kwargs=policy_kwargs(),
                   learning_rate=3e-4, buffer_size=buffer_size,
                   learning_starts=learning_starts, batch_size=256, tau=0.005,
                   gamma=0.99, train_freq=1, gradient_steps=grad_steps,
                   ent_coef="auto", verbose=1, seed=SEED, device=device)

    rew_rec = EpisodeRewardRecorder()
    met_rec = SacMetricsRecorder()
    ckpt_cb = OverwriteCheckpointCallback(save_path=save_name, save_freq=20_000)
    best_cb = BestModelCallback(save_path=best_name)
    callbacks = [ckpt_cb, best_cb, rew_rec, met_rec]

    if curriculum:
        # Curriculum: misma red y replay buffer entre etapas (reset_num_timesteps=False).
        # Cada etapa rellena las n_envs ranuras ciclando sus mapas, manteniendo el mismo
        # n_envs que exige set_env de SB3.
        stages = [
            ["Duckietown-straight_road-v0"],
            ["Duckietown-small_loop-v0", "Duckietown-loop_empty-v0"],
            TRAIN_MAPS,
        ]
        model = None
        per_stage = max(1, timesteps // len(stages))
        for i, maps in enumerate(stages, 1):
            stage_maps = resolve_maps(maps, n_envs)
            rew_rec.env_maps = stage_maps          # mantener el mapeo entorno->mapa correcto
            print(f"\n--- Curriculum etapa {i}/{len(stages)}: {stage_maps} ---", flush=True)
            env = make_vec_env(stage_maps, shaping=True, n_envs=n_envs)
            if model is None:
                model = build(env)
            else:
                model.set_env(env)
            model.learn(total_timesteps=per_stage, reset_num_timesteps=(i == 1),
                        callback=callbacks)
            env.close()
    else:
        # Por defecto: los 5 mapas EN PARALELO durante todo el presupuesto de pasos.
        # "Paralelizar los escenarios" en su forma mas directa.
        stage_maps = resolve_maps(TRAIN_MAPS, n_envs)
        rew_rec.env_maps = stage_maps
        env = make_vec_env(TRAIN_MAPS, shaping=True, n_envs=n_envs)
        model = build(env)
        model.learn(total_timesteps=timesteps, callback=callbacks)
        env.close()

    model.save(save_name)
    print(f"\nSAC guardado en {save_name}.zip", flush=True)
    return model, {"ep_rewards": rew_rec.episode_rewards, "per_map": rew_rec.per_map,
                   "critic_loss": met_rec.critic_loss, "actor_loss": met_rec.actor_loss,
                   "ent_coef": met_rec.ent_coef}


# ------------------------------------------------------------------ GRAFICAS
def _new_ax(figsize=(10, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def save_reward_curve(ep_rewards, results_dir):
    rd = pathlib.Path(results_dir); rd.mkdir(parents=True, exist_ok=True)
    json.dump(ep_rewards, open(rd / "sac_ep_rewards.json", "w"))
    if not ep_rewards:
        print("[aviso] sin episodios completos para la curva de recompensa")
        return
    w = min(20, len(ep_rewards))
    ma = np.convolve(ep_rewards, np.ones(w) / w, mode="valid")
    fig, ax = _new_ax()
    ax.plot(ma, color="darkorange")
    ax.axhline(ma[-1], color="tomato", linestyle="--", label=f"Final: {ma[-1]:.1f}")
    ax.set(xlabel="Episodio", ylabel=f"Recompensa (media movil {w})",
           title="SAC (Fase 3) - curva de entrenamiento en Duckietown")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(rd / "sac_training.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"Guardado: {rd / 'sac_training.png'}")


def save_permap_curve(per_map, results_dir):
    rd = pathlib.Path(results_dir); rd.mkdir(parents=True, exist_ok=True)
    short = {short_map(k): v for k, v in per_map.items()}
    json.dump(short, open(rd / "sac_per_map_rewards.json", "w"))
    if not short:
        return
    fig, ax = _new_ax(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")
    for idx, (m, rews) in enumerate(sorted(short.items())):
        if not rews:
            continue
        w = min(10, len(rews))
        ma = np.convolve(rews, np.ones(w) / w, mode="valid")
        ax.plot(ma, color=cmap(idx % 10), linewidth=1.5, label=f"{m} (n={len(rews)})")
    ax.set(xlabel="Episodio (por mapa)", ylabel="Recompensa (media movil)",
           title="SAC - recompensa por escenario (mapa)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(rd / "sac_per_map.png", dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"Guardado: {rd / 'sac_per_map.png'}")


def save_metric_curve(pairs, ylabel, title, filename, color, results_dir, json_name):
    rd = pathlib.Path(results_dir); rd.mkdir(parents=True, exist_ok=True)
    json.dump(pairs, open(rd / json_name, "w"))
    if not pairs:
        return
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    w = min(20, len(ys))
    fig, ax = _new_ax()
    ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.8)
    if len(ys) >= w:
        ma = np.convolve(ys, np.ones(w) / w, mode="valid")
        ax.plot(xs[w - 1:], ma, color=color, linewidth=1.8, label=f"media movil {w}")
        ax.legend()
    ax.set(xlabel="Timesteps", ylabel=ylabel, title=title)
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(rd / filename, dpi=120, bbox_inches="tight"); plt.close(fig)
    print(f"Guardado: {rd / filename}")


def save_all_curves(hist, results_dir):
    save_reward_curve(hist["ep_rewards"], results_dir)
    save_permap_curve(hist["per_map"], results_dir)
    save_metric_curve(hist["critic_loss"], "train/critic_loss",
                      "SAC - critic loss (global)", "sac_critic_loss.png",
                      "crimson", results_dir, "sac_critic_loss.json")
    save_metric_curve(hist["ent_coef"], "ent_coef (temperatura)",
                      "SAC - temperatura de entropia (auto)", "sac_ent_coef.png",
                      "purple", results_dir, "sac_ent_coef.json")
    save_metric_curve(hist["actor_loss"], "train/actor_loss",
                      "SAC - actor loss (global)", "sac_actor_loss.png",
                      "teal", results_dir, "sac_actor_loss.json")


# ------------------------------------------------------------------ MAIN
def main():
    parser = argparse.ArgumentParser(description="Fase 3 - SAC en Duckietown.")
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--n-envs", type=int, default=len(TRAIN_MAPS),
                        help="Entornos en paralelo (SubprocVecEnv) para saturar CPU y GPU.")
    parser.add_argument("--curriculum", action="store_true",
                        help="Entrena por etapas de dificultad creciente en vez de los 5 mapas a la vez.")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--eval-episodes", type=int, default=3)
    parser.add_argument("--results", type=str, default=os.path.join("results", "Fase_3"))
    parser.add_argument("--models-dir", type=str, default=os.path.join("models", "Models_v2"))
    args = parser.parse_args()

    results_dir = pathlib.Path(args.results); results_dir.mkdir(parents=True, exist_ok=True)
    models_dir = pathlib.Path(args.models_dir); models_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_fase3] Dispositivo: {device}")
    print(f"[train_fase3] Entornos en paralelo: {args.n_envs}  |  timesteps: {args.timesteps:,}")
    print(f"[train_fase3] Modelos -> {models_dir}  |  Resultados -> {results_dir}")

    # Pantalla virtual para el render OpenGL headless.
    try:
        from pyvirtualdisplay import Display
        Display(visible=False, size=(1024, 768)).start()
        print("[train_fase3] Pantalla virtual (Xvfb) iniciada.")
    except Exception as e:
        print(f"[train_fase3] Sin pantalla virtual: {e}")

    model, hist = train_sac(args.timesteps, args.n_envs, curriculum=args.curriculum,
                            device=device, models_dir=models_dir,
                            buffer_size=args.buffer_size, learning_starts=args.learning_starts)

    save_all_curves(hist, results_dir)

    print("\n=== Evaluacion (modo determinista, sin reward shaping) ===", flush=True)
    res = evaluate_agent(model, TRAIN_MAPS, n_episodes=args.eval_episodes)
    json.dump(res, open(results_dir / "sac_eval.json", "w"))
    print(f"SAC: recompensa media = {res['mean_reward']:.2f} +/- {res['std_reward']:.2f}")
    for m, rews in res["per_map_eval"].items():
        if rews:
            print(f"   {m:>14}: {np.mean(rews):7.2f}  (n={len(rews)})")

    print("\n=== RESUMEN FASE 3 ===")
    print(f"  Mejor modelo : {models_dir / 'sac_duckie_v2_best'}.zip")
    print(f"  Modelo final : {models_dir / 'sac_duckie_v2'}.zip")
    print(f"  Curvas/JSON  : {results_dir}")


if __name__ == "__main__":
    main()
