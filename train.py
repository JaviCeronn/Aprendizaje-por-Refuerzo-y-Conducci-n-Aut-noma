"""
=====================================================================
 FASES 2 y 3 – Pipeline de entrenamiento (DQN, PPO y SAC) en Duckietown
=====================================================================
Este script define TODO lo necesario para entrenar agentes de conducción
autónoma sobre píxeles en bruto y guardar el mejor modelo.

Componentes (todos reutilizables desde eval.ipynb):
    * DuckieWrapper      -> adapta gym-duckietown a Gymnasium, recorta el
                            cielo, pasa a gris y redimensiona a (1, 64, 64).
    * DiscreteWrapper    -> discretiza el espacio de acción para DQN.
    * LaneFollowingReward-> reward shaping opcional (solo en entrenamiento).
    * CustomCNN          -> extractor de características convolucional.

Contrato de evaluación (CRÍTICO):
    - Observación final tras frame stacking: (4, 64, 64), uint8.
    - El modelo "best_duckie_agent" es CONTINUO (SAC/PPO) y por tanto
      compatible directamente con la celda de evaluación del profesor,
      que usa DuckieWrapper (acciones continuas) + VecFrameStack(n_stack=4).

Uso (en Google Colab con GPU, Python 3.11):
    python train.py --algo all --timesteps 200000
    python train.py --algo sac --timesteps 300000 --curriculum
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import gym as old_gym                 # API antigua (la usa gym-duckietown)
import gymnasium as gym               # API moderna (Stable-Baselines3)
from gymnasium import spaces

import cv2
import torch
import torch.nn as nn

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


# =====================================================================
#  CONSTANTES GLOBALES
# =====================================================================
IMG_SIZE = 64
N_STACK = 4
OBS_SHAPE = (1, IMG_SIZE, IMG_SIZE)        # antes del frame-stacking
SEED = 42

# Mapas de entrenamiento PERMITIDOS (el de evaluación es secreto y no se usa).
TRAIN_MAPS = [
    "Duckietown-loop_empty-v0",
    "Duckietown-udem1-v0",
    "Duckietown-zigzag_dists-v0",
    "Duckietown-small_loop-v0",
    "Duckietown-straight_road-v0",
]

# Acciones discretas para DQN: pares [vel_izq, vel_der] (ruedas).
# Permiten: recto, giro suave izq/der, giro fuerte izq/der.
DISCRETE_ACTIONS = np.array(
    [
        [0.6, 0.6],    # 0 - avanzar recto
        [0.35, 0.6],   # 1 - girar suave a la izquierda
        [0.6, 0.35],   # 2 - girar suave a la derecha
        [0.2, 0.6],    # 3 - girar fuerte a la izquierda
        [0.6, 0.2],    # 4 - girar fuerte a la derecha
    ],
    dtype=np.float32,
)


# =====================================================================
#  COMPATIBILIDAD gym 0.25.2
# =====================================================================
# gym 0.25.2 expone np_random como np.random.Generator (con .integers() y
# .random() nativos, además de los alias .randint()/.rand()), que es justo lo
# que usa gym-duckietown. Por tanto NO hace falta ningún parche de seeding.


# =====================================================================
#  WRAPPERS DE ENTORNO
# =====================================================================
class DuckieWrapper(gym.Env):
    """Adapta gym-duckietown (API gym 0.21) a Gymnasium.

    Procesa la observación: recorta el 50 % superior (cielo), convierte a
    escala de grises y redimensiona a 64x64 -> (1, 64, 64) uint8.

    Acciones continuas: Box([-1,-1], [1,1]) = velocidades de las dos ruedas.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_name: str = "Duckietown-loop_empty-v0", seed: int | None = None):
        super().__init__()
        # Importación perezosa: registra los entornos Duckietown al importar.
        import gym_duckietown  # noqa: F401

        self.env_name = env_name
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
        self.observation_space = spaces.Box(
            low=0, high=255, shape=OBS_SHAPE, dtype=np.uint8
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.env.reset()
        # gym 0.21 devuelve solo obs; algunas versiones devuelven (obs, info).
        if isinstance(obs, tuple):
            obs = obs[0]
        return self._process_obs(obs), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, done, info = self.env.step(action)
        terminated = bool(done)
        truncated = False
        return self._process_obs(obs), float(reward), terminated, truncated, info

    def _process_obs(self, obs):
        # Recortar cielo (mitad superior), gris, redimensionar a 64x64.
        obs = obs[obs.shape[0] // 2:, :, :]
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=0).astype(np.uint8)

    def render(self):
        return self.env.render(mode="rgb_array")

    def close(self):
        self.env.close()


class DiscreteWrapper(gym.ActionWrapper):
    """Convierte el espacio de acción continuo en Discrete(n) para DQN.

    Cada índice discreto se mapea a un par [vel_izq, vel_der] de DISCRETE_ACTIONS.
    """

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

    def action(self, action):
        return DISCRETE_ACTIONS[int(action)]


class LaneFollowingReward(gym.Wrapper):
    """Reward shaping para acelerar el aprendizaje (SOLO en entrenamiento).

    Penaliza fuertemente salirse de la carretera y recompensa avanzar. Usa la
    información del simulador cuando está disponible; si no, recae en la
    recompensa original del entorno. NO se usa en la evaluación final.
    """

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        shaped = reward
        # Penalización extra si el episodio termina por salirse (reward muy bajo).
        if terminated and reward < 0:
            shaped = -10.0
        # Pequeño incentivo a mantenerse en marcha.
        shaped += 0.1
        return obs, shaped, terminated, truncated, info


# =====================================================================
#  RED CONVOLUCIONAL PERSONALIZADA
# =====================================================================
class CustomCNN(BaseFeaturesExtractor):
    """CNN para extraer características de imágenes apiladas (4, 64, 64).

    Arquitectura tipo "Nature CNN" (Mnih et al., 2015) adaptada a 64x64.
    Los píxeles se normalizan a [0,1] dentro del forward.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]  # 4 tras frame stacking

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calcular dinámicamente la dimensión de salida del bloque convolucional.
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Normalizar los píxeles uint8 [0,255] -> [0,1].
        observations = observations.float() / 255.0
        return self.linear(self.cnn(observations))


# =====================================================================
#  CONSTRUCCIÓN DE ENTORNOS VECTORIZADOS
# =====================================================================
def make_env(map_name: str, discrete: bool = False, shaping: bool = True, seed: int = SEED):
    """Devuelve una función creadora de un entorno (para DummyVecEnv)."""

    def _init():
        env = DuckieWrapper(map_name, seed=seed)
        if shaping:
            env = LaneFollowingReward(env)
        if discrete:
            env = DiscreteWrapper(env)
        return env

    return _init


def make_vec_env(maps, discrete: bool = False, shaping: bool = True, n_stack: int = N_STACK):
    """Crea un VecEnv con varios mapas + frame stacking + monitor.

    El resultado tiene observación (n_stack, 64, 64), cumpliendo el contrato.
    """
    env_fns = [make_env(m, discrete=discrete, shaping=shaping, seed=SEED + i)
               for i, m in enumerate(maps)]
    vec = DummyVecEnv(env_fns)
    vec = VecFrameStack(vec, n_stack=n_stack)
    vec = VecMonitor(vec)
    return vec


def policy_kwargs(features_dim: int = 256):
    """policy_kwargs común para usar nuestra CustomCNN."""
    return dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=features_dim),
    )


# =====================================================================
#  EVALUACIÓN INTERNA (para comparar algoritmos en el informe)
# =====================================================================
def evaluate_agent(model, eval_maps, n_episodes: int = 5, discrete: bool = False) -> dict:
    """Evalúa un modelo y devuelve recompensa media/desv. típica."""
    env = make_vec_env(eval_maps, discrete=discrete, shaping=False)
    returns = []
    for _ in range(n_episodes):
        obs = env.reset()
        done = np.array([False])
        ep_ret = 0.0
        steps = 0
        while not done[0] and steps < 1000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            ep_ret += float(reward[0])
            steps += 1
        returns.append(ep_ret)
    env.close()
    return {"mean_reward": float(np.mean(returns)), "std_reward": float(np.std(returns)),
            "returns": returns}


# =====================================================================
#  ENTRENAMIENTO DE CADA ALGORITMO
# =====================================================================
def train_dqn(timesteps: int, device: str = "auto") -> DQN:
    print("\n=== Entrenando DQN (baseline, acciones discretas) ===")
    env = make_vec_env(TRAIN_MAPS, discrete=True, shaping=True)
    model = DQN(
        "CnnPolicy", env,
        policy_kwargs=policy_kwargs(),
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        target_update_interval=1_000,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1, seed=SEED, device=device,
    )
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save("dqn_duckie_agent")
    env.close()
    return model


def train_ppo(timesteps: int, device: str = "auto") -> PPO:
    print("\n=== Entrenando PPO (baseline, acciones continuas) ===")
    env = make_vec_env(TRAIN_MAPS, discrete=False, shaping=True)
    model = PPO(
        "CnnPolicy", env,
        policy_kwargs=policy_kwargs(),
        learning_rate=3e-4,
        n_steps=2_048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1, seed=SEED, device=device,
    )
    model.learn(total_timesteps=timesteps, progress_bar=True)
    model.save("ppo_duckie_agent")
    env.close()
    return model


def train_sac(timesteps: int, curriculum: bool = False, device: str = "auto") -> SAC:
    """FASE 3: SAC (Soft Actor-Critic), off-policy de control continuo.

    Si curriculum=True, entrena por etapas de mapas crecientes en dificultad.
    """
    print("\n=== Entrenando SAC (Fase 3, acciones continuas) ===")

    def build_model(env):
        return SAC(
            "CnnPolicy", env,
            policy_kwargs=policy_kwargs(),
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=1, seed=SEED, device=device,
        )

    if curriculum:
        # Curriculum: de lo más fácil (recta) a lo más difícil (zigzag).
        stages = [
            ["Duckietown-straight_road-v0"],
            ["Duckietown-small_loop-v0", "Duckietown-loop_empty-v0"],
            TRAIN_MAPS,
        ]
        model = None
        per_stage = max(1, timesteps // len(stages))
        for i, maps in enumerate(stages, 1):
            print(f"\n--- Curriculum etapa {i}/{len(stages)}: {maps} ---")
            env = make_vec_env(maps, discrete=False, shaping=True)
            if model is None:
                model = build_model(env)
            else:
                model.set_env(env)
            model.learn(total_timesteps=per_stage, progress_bar=True,
                        reset_num_timesteps=False)
            env.close()
    else:
        env = make_vec_env(TRAIN_MAPS, discrete=False, shaping=True)
        model = build_model(env)
        model.learn(total_timesteps=timesteps, progress_bar=True)
        env.close()

    model.save("sac_duckie_agent")
    return model


# =====================================================================
#  ORQUESTACIÓN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Entrenamiento RL Duckietown.")
    parser.add_argument("--algo", choices=["dqn", "ppo", "sac", "all"], default="all")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--curriculum", action="store_true",
                        help="Usar Curriculum Learning en SAC.")
    parser.add_argument("--eval-episodes", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {device}")

    # Pantalla virtual para el render OpenGL headless (Colab / servidor sin GUI).
    # Duckietown necesita un contexto OpenGL incluso para generar observaciones.
    try:
        from pyvirtualdisplay import Display
        Display(visible=False, size=(1024, 768)).start()
        print("[info] Pantalla virtual iniciada.")
    except Exception as e:
        print(f"[aviso] No se pudo iniciar pantalla virtual: {e}")

    results = {}
    trained = {}

    if args.algo in ("dqn", "all"):
        m = train_dqn(args.timesteps, device)
        trained["DQN"] = (m, True)
    if args.algo in ("ppo", "all"):
        m = train_ppo(args.timesteps, device)
        trained["PPO"] = (m, False)
    if args.algo in ("sac", "all"):
        m = train_sac(args.timesteps, curriculum=args.curriculum, device=device)
        trained["SAC"] = (m, False)

    # Evaluación comparativa sobre los mapas de entrenamiento permitidos.
    print("\n=== Evaluación comparativa (mapas de entrenamiento) ===")
    for name, (model, discrete) in trained.items():
        res = evaluate_agent(model, TRAIN_MAPS, n_episodes=args.eval_episodes,
                             discrete=discrete)
        results[name] = res
        print(f"{name:>4}: recompensa media = {res['mean_reward']:.2f} "
              f"± {res['std_reward']:.2f}")

    # Selección y guardado del MEJOR agente.
    # Solo modelos continuos (PPO/SAC) son compatibles con la celda de
    # evaluación del profesor (DuckieWrapper continuo). Si gana DQN, se
    # guarda igualmente el mejor continuo para la entrega.
    continuous = {n: r for n, r in results.items() if n in ("PPO", "SAC")}
    if continuous:
        best_name = max(continuous, key=lambda n: continuous[n]["mean_reward"])
    else:
        best_name = max(results, key=lambda n: results[n]["mean_reward"])

    best_model = trained[best_name][0]
    best_model.save("best_duckie_agent")
    print(f"\n>>> Mejor agente (continuo) para la entrega: {best_name}")
    print(">>> Guardado como best_duckie_agent.zip")

    # Resumen de resultados (útil para el Report).
    print("\n=== RESUMEN DE RESULTADOS ===")
    for name, r in results.items():
        print(f"{name}: {r['mean_reward']:.2f} ± {r['std_reward']:.2f} "
              f"| returns={[round(x, 1) for x in r['returns']]}")


if __name__ == "__main__":
    main()
