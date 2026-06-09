"""
Visualiza un agente entrenado en Duckietown y guarda un GIF.

Uso:
    xvfb-run -a python3.11 src/render_agent.py <modelo.zip> [opciones]

Ejemplos:
    xvfb-run -a python3.11 src/render_agent.py results/Fase_2/best_agent_v2.zip
    xvfb-run -a python3.11 src/render_agent.py results/Fase_2/dqn_duckie_v2.zip --algo dqn
    xvfb-run -a python3.11 src/render_agent.py results/Fase_2/best_agent_v2.zip --map Duckietown-zigzag_dists-v0 --episodes 3

El GIF se guarda junto al .zip con el sufijo _render.gif.
El algoritmo se infiere del nombre del archivo (dqn → DQN, resto → PPO)
pero puede forzarse con --algo.
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
import cv2
import gym as old_gym
import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation

IMG_SIZE = 64
N_STACK  = 4
OBS_SHAPE = (1, IMG_SIZE, IMG_SIZE)
SEED = 42

DISCRETE_ACTIONS = np.array([
    [0.6, 0.6],
    [0.35, 0.6],
    [0.6, 0.35],
    [0.2, 0.6],
    [0.6, 0.2],
], dtype=np.float32)


# ---------------------------------------------------------------------------
# Wrappers — idénticos a train.py para que el modelo cargue sin errores
# ---------------------------------------------------------------------------

def _suppress_duckietown_stdout():
    _out, sys.stdout = sys.stdout, _io.StringIO()
    try:
        import gym_duckietown  # noqa: F401
    finally:
        sys.stdout = _out
    logging.disable(logging.INFO)


class DuckieWrapper(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name="Duckietown-loop_empty-v0", seed=None):
        super().__init__()
        _suppress_duckietown_stdout()
        self.env = old_gym.make(env_name)
        if seed is not None:
            try:
                self.env.seed(seed)
            except Exception:
                pass
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(0, 255, shape=OBS_SHAPE, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        return self._process(obs[0] if isinstance(obs, tuple) else obs), {}

    def step(self, action):
        obs, r, done, info = self.env.step(np.asarray(action, np.float32).reshape(-1))
        return self._process(obs), float(r), bool(done), False, info

    def _process(self, obs):
        obs = obs[obs.shape[0] // 2:, :, :]
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, 0).astype(np.uint8)

    def render(self):
        return self.env.render(mode="rgb_array")

    def close(self):
        self.env.close()


class DiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

    def action(self, a):
        return DISCRETE_ACTIONS[int(a)]


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------

def _infer_algo(zip_path: str) -> str:
    name = pathlib.Path(zip_path).stem.lower()
    if "dqn" in name:
        return "dqn"
    if "sac" in name:
        return "sac"
    return "ppo"


def load_model(zip_path: str, algo: str, map_name: str):
    from stable_baselines3 import DQN, PPO, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor

    cls = {"dqn": DQN, "ppo": PPO, "sac": SAC}[algo]
    discrete = algo == "dqn"

    def env_fn():
        base = DuckieWrapper(map_name, seed=SEED)
        return DiscreteWrapper(base) if discrete else base

    vec = VecMonitor(VecFrameStack(DummyVecEnv([env_fn]), N_STACK))
    model = cls.load(zip_path, env=vec)
    return model, vec


# ---------------------------------------------------------------------------
# Ejecución de episodios con captura de frames
# ---------------------------------------------------------------------------

def run_episodes(model, vec, n_episodes: int, max_steps: int):
    all_frames, returns = [], []

    for ep in range(n_episodes):
        obs = vec.reset()
        done = np.array([False])
        ep_frames, ep_reward, steps = [], 0.0, 0

        while not done[0] and steps < max_steps:
            # render() devuelve el frame RGB completo del simulador (vista externa)
            frame = vec.envs[0].env.env.render()
            if frame is not None:
                ep_frames.append(frame)

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = vec.step(action)
            ep_reward += float(reward[0])
            steps += 1

        all_frames.extend(ep_frames)
        returns.append(ep_reward)
        status = "salio" if steps < max_steps else "max_steps"
        print(f"  Ep {ep + 1}/{n_episodes}: {steps} pasos  reward={ep_reward:.1f}  [{status}]")

    return all_frames, returns


# ---------------------------------------------------------------------------
# Guardado del GIF (usa Pillow, ya instalado como dep de gymnasium)
# ---------------------------------------------------------------------------

def save_gif(frames: list, output_path: pathlib.Path, fps: int = 15, skip: int = 2):
    if not frames:
        print("Sin frames — el simulador no devolvio imágenes RGB.")
        return

    frames = frames[::skip]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.axis("off")
    im = ax.imshow(frames[0])
    plt.tight_layout(pad=0)

    def update(i):
        im.set_data(frames[i])
        return [im]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames), interval=1000 // fps, blit=True
    )
    ani.save(str(output_path), writer=animation.PillowWriter(fps=fps))
    plt.close()
    print(f"GIF guardado: {output_path}  ({len(frames)} frames @ {fps} fps)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Renderiza un agente Duckietown y guarda un GIF.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("zip", help="Ruta al archivo .zip del modelo entrenado.")
    parser.add_argument(
        "--algo", choices=["dqn", "ppo", "sac"], default=None,
        help="Algoritmo. Si no se indica, se infiere del nombre del archivo.",
    )
    parser.add_argument(
        "--map", default="Duckietown-loop_empty-v0",
        dest="map_name",
        help="Mapa de Duckietown (default: loop_empty).",
    )
    parser.add_argument("--episodes", type=int, default=2, help="Número de episodios (default: 2).")
    parser.add_argument("--max-steps", type=int, default=400, help="Pasos máximos por episodio (default: 400).")
    parser.add_argument("--fps", type=int, default=15, help="FPS del GIF (default: 15).")
    args = parser.parse_args()

    zip_path = args.zip if args.zip.endswith(".zip") else args.zip + ".zip"
    if not pathlib.Path(zip_path).exists():
        print(f"ERROR: no se encuentra {zip_path}")
        sys.exit(1)

    algo = args.algo or _infer_algo(zip_path)
    output = pathlib.Path(zip_path).with_suffix("").with_name(
        pathlib.Path(zip_path).stem + "_render.gif"
    )

    try:
        from pyvirtualdisplay import Display
        Display(visible=False, size=(640, 480)).start()
    except Exception as e:
        print(f"Sin pantalla virtual: {e}")

    print(f"Modelo : {zip_path}")
    print(f"Algo   : {algo}")
    print(f"Mapa   : {args.map_name}")
    print(f"Salida : {output}\n")

    model, vec = load_model(zip_path, algo, args.map_name)
    frames, returns = run_episodes(model, vec, args.episodes, args.max_steps)
    vec.close()

    print(f"\nRecompensa media: {np.mean(returns):.1f}  |  Total frames: {len(frames)}")
    save_gif(frames, output, fps=args.fps)


if __name__ == "__main__":
    main()
