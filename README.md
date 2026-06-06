# Aprendizaje por Refuerzo y Conducción Autónoma

Proyecto final del Máster en Inteligencia Artificial — Asignatura: Machine Learning Avanzado (Aprendizaje por Refuerzo).

Entrenamiento de un agente de **conducción autónoma** en el simulador 3D **Duckietown**, utilizando únicamente píxeles de la cámara frontal como entrada. El agente aprende a seguir el carril y generalizar a mapas no vistos con obstáculos.

---

## Estructura del repositorio

```
.
├── requirements.txt              # Dependencias (versiones fijadas con ==)
├── src/
│   ├── train.py                  # Pipeline de entrenamiento (DQN, PPO, SAC)
│   └── q_learning_sandbox.py     # Fase 1: Q-Learning tabular en FrozenLake
├── notebooks/
│   ├── Challenge_RL.ipynb        # Notebook principal — las 3 fases completas (Colab)
│   └── eval.ipynb                # Evaluación del agente + generación de vídeo
├── models/
│   ├── best_duckie_agent.zip     # Mejor agente continuo (SAC o PPO)
│   ├── sac_duckie_agent.zip      # SAC — Fase 3 (algoritmo avanzado)
│   ├── ppo_duckie_agent.zip      # PPO — baseline continuo
│   └── dqn_duckie_agent.zip      # DQN — baseline discreto
├── results/
│   └── q_learning_progress.png   # Curva de entrenamiento Q-Learning (Fase 1)
├── docs/
│   ├── Report.md                 # Memoria técnica completa
│   └── Presentacion_Conduccion_Autonoma.pptx
└── challenge/
    └── Challenge_RL_Enunciado    # Enunciado original del desafío (material del profesor)
```

---

## Fases del proyecto

| Fase | Algoritmo | Entorno | Descripción |
|------|-----------|---------|-------------|
| 1 | Q-Learning tabular | FrozenLake-v1 8×8 | Implementación desde cero de la ecuación de Bellman |
| 2 | DQN + PPO | Duckietown (5 mapas) | Baselines: control discreto (DQN) y continuo (PPO) |
| 3 | **SAC** + Curriculum Learning | Duckietown (5 mapas) | Algoritmo avanzado off-policy con exploración entrópica |

---

## Pipeline de visión

Cada frame RGB pasa por:

1. **Recorte** — se elimina el 50 % superior (cielo)
2. **Escala de grises** — de 3 canales a 1
3. **Redimensión** — a 64 × 64 píxeles
4. **Frame stacking** — 4 frames apilados → observación `(4, 64, 64)`

La CNN (`CustomCNN`) sigue el diseño *Nature CNN* con 3 capas convolucionales y un vector de características de 256 dimensiones.

---

## Configuración del entorno

> Requiere Python 3.11. En Google Colab se instala vía deadsnakes (ver `notebooks/Challenge_RL.ipynb`).

```bash
pip install -r requirements.txt
pip install --no-deps git+https://github.com/duckietown/gym-duckietown.git@daffy
```

En Linux/Colab, instalar también las dependencias del sistema:

```bash
sudo apt-get install -y xvfb freeglut3-dev libosmesa6-dev
```

---

## Entrenamiento

```bash
# Entrenar los 3 algoritmos con curriculum (recomendado)
python src/train.py --algo all --timesteps 200000 --curriculum

# Solo SAC (Fase 3) con más timesteps
python src/train.py --algo sac --timesteps 300000 --curriculum

# Solo Fase 1 (Q-Learning)
python src/q_learning_sandbox.py
```

Los modelos se guardan en el directorio de trabajo como `*.zip`. El mejor agente continuo se guarda como `best_duckie_agent.zip`.

---

## Evaluación

Abrir `notebooks/eval.ipynb` y ejecutar todas las celdas. Carga `models/best_duckie_agent.zip`, conduce un episodio en el mapa indicado y genera `duckie_eval_video.mp4`.

**Contrato de evaluación** (profesor): carga `best_duckie_agent.zip` en el mapa oculto `Duckietown-loop_obstacles-v0`. Las clases `DuckieWrapper` y `CustomCNN` están definidas inline en `eval.ipynb` para garantizar la carga.

---

## Mapas de entrenamiento

| Mapa | ID |
|------|----|
| loop_empty | `Duckietown-loop_empty-v0` |
| udem1 | `Duckietown-udem1-v0` |
| zigzag_dists | `Duckietown-zigzag_dists-v0` |
| small_loop | `Duckietown-small_loop-v0` |
| straight_road | `Duckietown-straight_road-v0` |

Mapa de evaluación oculto: `Duckietown-loop_obstacles-v0` (obstáculos estáticos, no visto en entrenamiento).

---

## Reproducibilidad

Semilla global `SEED = 42` fijada en todos los componentes. Ver `docs/Report.md` para la memoria técnica completa con justificación teórica y empírica.
