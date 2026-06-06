# Aprendizaje por Refuerzo y Conducción Autónoma

Proyecto final del Máster en Inteligencia Artificial — Aprendizaje por Refuerzo.
Agente de conducción autónoma en el simulador 3D **Duckietown** entrenado desde píxeles con Deep RL (Q-Learning → DQN/PPO → SAC + Curriculum Learning).

---

## Inicio rápido

### Local (Windows / Linux / Mac)

Requiere [uv](https://docs.astral.sh/uv/getting-started/installation/). Un solo comando replica el entorno exacto:

```bash
git clone https://github.com/JaviCeronn/Aprendizaje-por-Refuerzo-y-Conducci-n-Aut-noma.git
cd Aprendizaje-por-Refuerzo-y-Conducci-n-Aut-noma
uv sync
```

Ejecutar Fase 1 (funciona en cualquier OS):

```bash
uv run python src/q_learning_sandbox.py
```

Ejecutar Fases 2-3 (solo Linux / WSL2 — requiere display OpenGL):

```bash
xvfb-run -a uv run python src/train.py --algo sac --timesteps 200000 --curriculum
```

---

### Google Colab

1. Abre `notebooks/Challenge_RL.ipynb` en Colab
2. Activa GPU: `Entorno de ejecución → Cambiar tipo de entorno → T4 GPU`
3. Añade tu token en el panel **Secrets** (icono llave): `GITHUB_TOKEN`
4. Ejecuta las celdas en orden — la primera celda clona el repo y configura todo automáticamente

---

## Estructura del repositorio

```
.
├── pyproject.toml                # Dependencias — uv sync para replicar el entorno
├── requirements.txt              # Versiones exactas para pip (entregable al profesor)
├── src/
│   ├── train.py                  # Pipeline de entrenamiento (DQN, PPO, SAC)
│   └── q_learning_sandbox.py     # Fase 1: Q-Learning tabular en FrozenLake
├── notebooks/
│   ├── Challenge_RL.ipynb        # Notebook principal — 3 fases completas (Colab)
│   └── eval.ipynb                # Evaluación del agente + generación de vídeo
├── models/
│   ├── best_duckie_agent.zip     # Mejor agente continuo (SAC o PPO)
│   ├── sac_duckie_agent.zip      # SAC — Fase 3
│   ├── ppo_duckie_agent.zip      # PPO — baseline continuo
│   └── dqn_duckie_agent.zip      # DQN — baseline discreto
├── results/
│   └── q_learning_progress.png   # Curva de entrenamiento Q-Learning
├── docs/
│   ├── Report.md                 # Memoria técnica completa
│   └── Presentacion_Conduccion_Autonoma.pptx
└── challenge/
    └── Challenge_RL_Enunciado    # Enunciado original del profesor
```

---

## Fases del proyecto

| Fase | Algoritmo | Entorno | Descripción |
|------|-----------|---------|-------------|
| 1 | Q-Learning tabular | FrozenLake-v1 8×8 | Ecuación de Bellman desde cero |
| 2 | DQN + PPO | Duckietown (5 mapas) | Baselines discreto y continuo |
| 3 | **SAC** + Curriculum | Duckietown (5 mapas) | Off-policy continuo con entropía máxima |

---

## Compatibilidad por entorno

| Tarea | Windows | Linux / WSL2 | Google Colab |
|-------|:-------:|:------------:|:------------:|
| `uv sync` (setup) | ✅ | ✅ | — |
| Fase 1 Q-Learning | ✅ | ✅ | ✅ |
| Fases 2-3 Duckietown | ❌ | ✅ | ✅ |
| Evaluación + vídeo | ❌ | ✅ | ✅ |

---

## Contrato de evaluación (profesor)

El profesor carga `models/best_duckie_agent.zip` en un entorno limpio con `requirements.txt` y cambia el mapa a `Duckietown-loop_obstacles-v0`. Las clases `DuckieWrapper` y `CustomCNN` están definidas inline en `notebooks/eval.ipynb`.

---

## Reproducibilidad

Semilla global `SEED = 42`. Ver `docs/Report.md` para la memoria técnica completa.
