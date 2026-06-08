# Fase 2 (DQN + PPO en Duckietown) — Contexto completo

> Documento de contexto técnico. Generado el **2026-06-08**; actualizado el mismo día tras
> resolver un SEGUNDO error en Colab. Recoge el enunciado, el diagnóstico verificado de los
> problemas de ejecución en Colab y la solución FINAL aplicada a `notebooks/Fase2_DQN_PPO.ipynb`.

---

## 1. El reto (resumen del enunciado)

Fuente: `docs/Presentacion_Conduccion_Autonoma.pptx`.
**Documento oficial de referencia: el notebook "Challenge RL"** (la presentación es solo informativa).

Desafío final del Máster IA: entrenar agentes de conducción autónoma en **Duckietown**
(simulador 3D OpenGL) mediante Deep RL, partiendo de visión por cámara.

| Fase | Contenido |
|------|-----------|
| Fase 1 | Q-Learning tabular desde cero en FrozenLake-v1 (tabla 64×4, ε-greedy con decaimiento, gráfico media móvil 10k episodios). Sin librerías externas. |
| **Fase 2** | **Baselines DQN (acciones discretas vía Wrapper) y PPO (acciones continuas).** |
| Fase 3 | Algoritmo avanzado que supere los baselines (sugerido SAC / TD3 / A2C). Justificar en el informe. |

### Pipeline de visión (diapositiva 8)
1. **Recorte**: eliminar 50% superior (cielo).
2. **Grayscale**: a escala de grises.
3. **Resize**: 64×64.
4. **Stacking**: apilar 4 frames (percepción de movimiento).

### Generalización en 6 mapas (diapositiva 7)
- **Entrenamiento (5 mapas):** `loop_empty`, `udem1`, `zigzag_dists`, `small_loop`, `straight_road`.
- **Evaluación final (1 mapa OCULTO):** `Duckietown-loop_obstacles-v0` (con obstáculos estáticos).
  **Entrenar en este mapa = descalificación inmediata.**

### Entregables (diapositiva 9)
| Archivo | Descripción |
|---------|-------------|
| `requirements.txt` | Dependencias con versiones exactas (`==`). |
| `q_learning_sandbox.py` | Fase 1 completa (código + gráfico). |
| `train.py` | Pipeline de entrenamiento DQN, PPO y Algoritmo 3. |
| `best_agent.zip` | Mejor modelo entrenado (Fase 2 o 3). |
| `Report.pdf` | Comparativa de los 3 algoritmos + justificación técnica. |

### Contrato de evaluación (diapositiva 10) — CRÍTICO
- **REGLA DE ORO:** si el código no carga a la primera en un entorno limpio → **calificación 0**.
- El modelo **DEBE esperar observaciones de dimensión `(1, 64, 64)`**.
- El `eval.ipynb` debe incluir las definiciones de **CNN y Wrappers** (evitar errores de carga).
- **Dry-run obligatorio:** probar el ZIP en un Colab nuevo antes de subir.

### Ranking (diapositiva 11)
Recompensa acumulada media en el mapa oculto + calidad de código/wrappers + informe + comprensión.

**Fecha límite: 15 de junio, 23:59h.**

---

## 2. Entorno de ejecución

- **Colab usa hoy Python 3.12.13 + numpy 2.x** (confirmado, 2026). torch / tensorflow / scipy
  de Colab están **compilados contra numpy 2.x**.
- El notebook debe ejecutarse en **Colab con GPU T4** (Duckietown usa OpenGL/EGL, que no
  existe en Windows). En Windows el notebook detecta el caso e imprime instrucciones.

---

## 3. Los DOS errores y sus causas (verificado en venvs uv Python 3.12)

### Error 1 — `No module named 'duckietown_world'` (celda 4)
**Dos causas combinadas:**
1. **Dependencias faltantes:** la celda original no instalaba `duckietown-world-daffy` ni el
   ecosistema `zuper` completo.
2. **Incompatibilidad Python 3.12 (causa profunda):** `zuper_typing/monkey_patching_typing.py`
   hace `setattr(TypeVar, "__repr__", ...)`. En **Python 3.12 `typing.TypeVar` es un tipo C
   inmutable** → `TypeError: cannot set '__repr__' attribute of immutable type` → tumba toda la
   cadena `zuper_typing → zuper_ipce → duckietown_world → gym_duckietown`, y aguas arriba
   aparece el engañoso "No module named 'duckietown_world'".
   **Fix:** envolver ese único `setattr` en `try/except TypeError` (parche idempotente).

### Error 2 — `_blas_supports_fpe` (al importar SB3) — EL IMPORTANTE
Tras el primer arreglo (que bajaba numpy a 1.26.4), Colab falló con:
```
AttributeError: module 'numpy._core._multiarray_umath' has no attribute '_blas_supports_fpe'
```
Cadena: `from stable_baselines3 import DQN,PPO` → `torch.utils.tensorboard` → `tensorflow`
→ `scipy` → `from numpy import *` → crash.

**Causa raíz:** Colab trae **numpy 2.x** (y torch/tf/scipy compilados para 2.x). Al instalar
`stable-baselines3==2.2.1`, que declara **`numpy>=1.20,<2.0`**, pip **baja numpy a 1.x**. El
kernel ya tenía la extensión C de numpy 2.x cargada en memoria → ficheros 1.x en disco +
C-ext 2.x en RAM = **numpy en estado MIXTO** → `AttributeError`.
**La versión de numpy la manda torch/SB3:** pinear `numpy==1.26.4` o instalar un torch viejo
provoca el mismo downgrade y el mismo crash.

### Bug pre-existente (corregido)
La celda de clases (`9b680c5f`) tenía un f-string roto partido en 3 líneas físicas
(`print(f"\nDuckietown no disponible: {e}\n")`) → `SyntaxError` que impedía definir las clases.

---

## 4. Solución FINAL: Opción A — no tocar numpy ni torch

| Pin | Decisión final | Motivo |
|---|---|---|
| `numpy` | **NO instalar** (usar el 2.x de Colab) | torch/tf/scipy de Colab compilados para numpy 2.x; bajarlo rompe el kernel |
| `torch` | **NO instalar** (usar el de Colab, GPU/CUDA, numpy 2.x) | reinstalar rompe CUDA y arrastra numpy |
| `scipy`, `scikit-image` | **NO pinear** (deps transitivas; Colab ya las trae numpy-2) | evitar downgrade de numpy |
| `opencv` | usar el `cv2` de Colab (no reinstalar en Colab) | evitar cv2 duplicado; el de Colab es numpy-2 |
| `stable-baselines3` | **`2.6.0`** (antes 2.2.1 ❌) | 2.2.1 capaba `numpy<2.0`; **SB3 ≥2.5.0 acepta `numpy<3.0`** |
| `gymnasium` | **`0.29.1`** | compatible con SB3 2.6.0 (`>=0.29.1,<1.2`) y con el wrapper |
| `gym` (antiguo) | **`0.25.2`** | API gym ≤0.25 (`step()` 4-tupla); importa OK bajo numpy 2.x (solo warning) |
| `pyglet` | **`1.5.27`** | motor gráfico de duckietown |
| `zuper-commons-z6==6.2.4`, `zuper-typing-z6==6.2.3`, `zuper-ipce-z6==6.1.2`, `zuper-nodes-z6==6.2.17`, `PyGeometry-z6==2.1.5`, `pycontracts3==7.2`, `aido-protocols-daffy==6.1.1`, `duckietown-serialization-ds1==1.1.1`, `duckietown-world-daffy==6.4.3`, `carnivalmirror==0.6.2` | **mantener pins** | verificado que importan bajo numpy 2.x con el parche zuper, sin bajar numpy |
| `pyvirtualdisplay==3.0` | mantener | display headless |
| Constraint `numpy>=2.0` en los `pip install` | añadido | si algún paquete intenta bajar numpy, pip **falla visible** en vez de romper el kernel en silencio |

**Resultado:** numpy permanece 2.x → no hay mezcla → **NO hace falta reiniciar el runtime**.

### Verificación realizada (venv uv Python 3.12 + numpy 2.2.6 + torch 2.12.0)
- ✅ `import torch`, `import gym`(0.25.2), todo el stack SB3 (`DQN`, `PPO`, `vec_env`,
  `torch_layers`, `callbacks`) e `import duckietown_world` → **todos OK simultáneamente**.
- ✅ numpy se mantiene en 2.2.6 tras instalar toda la lista (dry-run: numpy/torch no cambian).
- ✅ SB3 2.5.0+ permite `numpy<3.0` (2.4.x aún capaba `<2.0`). Caps gymnasium: 2.5→`<1.1`,
  2.6→`<1.2`, 2.7→`<1.3`.
- ✅ Parche zuper hace importar la cadena en Py3.12 también bajo numpy 2.x.

### NO verificable desde Windows
- ⚠️ Render OpenGL real y entrenamiento → EGL no existe en Windows (es lo que el notebook
  documenta). En Colab los `apt-get` de mesa/EGL lo resuelven. Camino estándar.

### Por qué NO Python 3.11 / NO reinicio / NO numpy 1.26
- Bajar numpy (a 1.23.5 o 1.26.4) recrea el estado mixto y/o exige reiniciar el runtime → frágil.
  `numpy 1.23.5` además no tiene wheel cp312 (falla a compilar; `distutils` fuera en 3.12).
- Forzar Python 3.11 (condacolab) implica reinicio + reinstalar torch+CUDA → frágil.
- La Opción A (Py3.12 + numpy 2.x de Colab + SB3 2.6.0 + parche zuper) es la más limpia.

---

## 5. Solución aplicada al notebook

Archivo: `notebooks/Fase2_DQN_PPO.ipynb` (16 celdas, todas validan sintaxis).

### Celda de instalación (`a8447948`) — reescrita (Opción A)
Solo si `IN_COLAB`:
1. `apt-get`: `xvfb`, mesa GL, **`libegl1` + `libegl1-mesa-dev`** (EGL headless), `libosmesa6-dev`, `libturbojpeg`.
2. Toolchain: `pip`, `setuptools>=68`, `wheel` (distutils para Py3.12).
3. RL: `numpy>=2.0` (constraint), **`stable-baselines3==2.6.0`**, `gymnasium==0.29.1`, `pyvirtualdisplay==3.0`. **NO numpy/torch/scipy.**
4. Stack Duckietown (con constraint `numpy>=2.0`): `gym==0.25.2`, `pyglet==1.5.27`, `carnivalmirror==0.6.2`, `pycontracts3==7.2`, `PyGeometry-z6==2.1.5`, `zuper-commons-z6==6.2.4`, `zuper-typing-z6==6.2.3`, `zuper-ipce-z6==6.1.2`, `zuper-nodes-z6==6.2.17`, `aido-protocols-daffy==6.1.1`, `duckietown-serialization-ds1==1.1.1`, `duckietown-world-daffy==6.4.3`.
5. `gym-duckietown` daffy con `--no-build-isolation --no-deps` (git `@daffy`).
6. **Parche Python 3.12:** localiza `zuper_typing/monkey_patching_typing.py` por ruta y envuelve el `setattr(TypeVar, ...)` en `try/except TypeError`. Idempotente.
7. **Guard de numpy:** comprueba `numpy.__version__`; si quedó <2.x avisa de reiniciar el runtime.
8. Arranca `Display` virtual (pyvirtualdisplay).

### Celda de clases (`9b680c5f`) — f-string roto corregido
Resto idéntico: `DuckieWrapper`, `DiscreteWrapper`, `LaneFollowingReward`, `CustomCNN`
(Nature CNN), `POLICY_KWARGS`, `make_env`, `make_vec_env`, `_duckietown_available`.

---

## 6. Arquitectura del notebook (sin cambios de diseño)

- **`DuckieWrapper(gym.Env)`** (gym = gymnasium): adapta API antigua→gymnasium y preprocesa.
  `observation_space = Box(0,255, (1,64,64), uint8)`. `step()` interno desempaqueta 4-tupla
  (API gym 0.25). `reset()` maneja obs tupla/ndarray.
- **`DiscreteWrapper`**: 5 acciones discretas `DISCRETE_ACTIONS` (recto, 2 giros suaves, 2 fuertes) → para DQN.
- **`LaneFollowingReward`**: shaping −10 al salirse, +0.1 por paso.
- **`CustomCNN`** (Nature CNN, Mnih 2015): 3 conv + linear→256 dims.
- **`make_vec_env`**: en Windows `DummyVecEnv`; en Linux/Colab `SubprocVecEnv(start_method="spawn")` (un proceso/contexto OpenGL por mapa) salvo 1 mapa → DummyVecEnv. Envuelto en `VecFrameStack(n_stack=4)` + `VecMonitor`.
- **DQN**: `lr=1e-4`, `buffer=50k`, `batch=64`, `γ=0.99`, `train_freq=4`, `target_update=1000`, `exploration_fraction=0.2`, `final_eps=0.05`.
- **PPO**: `lr=3e-4`, `n_steps=2048`, `batch=256`, `n_epochs=10`, `γ=0.99`, `gae_λ=0.95`, `clip=0.2`, `ent_coef=0.01`.
- `TIMESTEPS=30_000` (test rápido; subir a 200k+ para resultados reales). `SEED=42`, `IMG_SIZE=64`, `N_STACK=4`.
- Resultados en `results/Fase_2/`. La celda de evaluación guarda el mejor agente continuo como `best_agent.zip` (PPO en Fase 2; será SAC en Fase 3).

---

## 7. Riesgos / temas abiertos pendientes

1. **Contrato (1,64,64) vs (4,64,64):** la diapositiva 10 dice que el modelo debe esperar
   `(1,64,64)`, pero el pipeline apila 4 frames → la política guardada espera `(4,64,64)`.
   Reconciliable (el `(1,64,64)` es la salida del wrapper por frame; el `eval.ipynb` del
   profesor debe aplicar el mismo `VecFrameStack`). **CONFIRMAR contra el notebook oficial
   "Challenge RL" antes de entregar** — afecta a la regla de oro (carga a la primera).
2. **Vec env en notebook → SIEMPRE `DummyVecEnv`:** `SubprocVecEnv(start_method="spawn")`
   FALLA en Colab/Jupyter con `ConnectionResetError: [Errno 104] Connection reset by peer`.
   Los procesos hijo (spawn) arrancan un Python vacío que no puede re-importar las clases
   definidas en celdas (`DuckieWrapper`, `make_env`...) y cada uno necesitaría su propio
   contexto OpenGL → el worker muere al inicializar. **Fix aplicado:** `make_vec_env` usa
   `DummyVecEnv` siempre.

3. **NUNCA crear los 5 mapas a la vez → entrenamiento/evaluación SECUENCIAL (1 env vivo).**
   `DummyVecEnv(TRAIN_MAPS)` con los 5 mapas abre **5 contextos OpenGL simultáneos en el
   proceso → CRASH del kernel** (síntoma en el log: `AsyncIOLoopKernelRestarter: restarting
   kernel`, y luego `NameError: name '_duckietown_available' is not defined` en celdas
   posteriores porque el reinicio borra todas las definiciones). El `NameError` es SÍNTOMA,
   no causa.
   **Fix aplicado (celdas `1c714d0f` DQN, `a3ae0242` PPO, `eval-fase2`):**
   - `train_dqn`/`train_ppo` crean **un solo env** (`make_vec_env([mapa])`), entrenan
     `TIMESTEPS//5` pasos, hacen `env.close(); gc.collect()`, recrean para el siguiente
     mapa con `model.set_env(...)` y `model.learn(..., reset_num_timesteps=(i==0))`.
   - `evaluate_agent` evalúa **un mapa a la vez** (1 episodio por mapa, `n_episodes_per_map`).
   - Solo hay 1 simulador OpenGL vivo en cada instante.
   - **La celda de clases `9b680c5f` NO cambió en este paso** (solo DQN/PPO/eval).

4. **RENDER: usar EGL headless (GPU), NO Xvfb — causa REAL del crash del kernel.**
   El crash del kernel (`restarting kernel`) **seguía con un solo entorno**, así que NO era
   el número de envs: era el **backend de renderizado OpenGL**. El notebook arrancaba un
   display virtual con `pyvirtualdisplay` (Xvfb) → pyglet renderizaba por **GLX software
   (llvmpipe)**, y ese camino **crashea el proceso** al dibujar el primer frame de Duckietown
   en Colab (pista previa: el dump de pyglet mostraba `'headless': False`).
   **Fix aplicado (celda de instalación `a8447948`, paso 8):** eliminado el
   `Display(...).start()` de pyvirtualdisplay; en su lugar, ANTES de importar gym_duckietown:
   ```python
   os.environ["PYGLET_HEADLESS"] = "1"
   os.environ["PYOPENGL_PLATFORM"] = "egl"
   import pyglet
   pyglet.options["headless"] = True
   ```
   Así pyglet crea el contexto OpenGL vía **EGL sobre la GPU T4** (estable). Debe fijarse
   antes de `import gym_duckietown` (que importa `pyglet.gl` al cargar) → por eso va en la
   celda de instalación, que se ejecuta primero.
   **CRÍTICO:** tras este cambio hay que **"Reiniciar y ejecutar todo"**, porque si pyglet ya
   se importó en modo no-headless en una ejecución previa del kernel, `pyglet.options` no
   tiene efecto retroactivo.
   **Fallback si AÚN crashea:** reducir `TRAIN_MAPS` a un solo mapa
   (`["Duckietown-loop_empty-v0"]`) para descartar fugas de contexto al recrear envs.
3. **gym 0.25.2 bajo numpy 2.x:** importa y emite un warning ("Gym does not support NumPy 2.0").
   El warning es inofensivo; pero si en runtime el simulador tropezara con algún alias numpy
   eliminado, habría que parchear puntualmente (no detectado en el barrido: 0 aliases rotos en
   `gym_duckietown`/`duckietown_world`).
4. **Entregables pendientes (diapositiva 9):** falta generar `requirements.txt` con los `==`
   definitivos y `train.py`.

---

## 8. requirements.txt sugerido (Colab Python 3.12 + numpy 2.x)

> NO pinear numpy/torch/scipy/opencv: usar los de Colab (numpy 2.x). En entorno LOCAL
> (sin Colab) sí harían falta, pero ahí Duckietown no corre por falta de EGL en Windows.

```
# RL — SB3 >=2.5 para aceptar numpy 2.x (la 2.2.1 capa numpy<2 y rompe Colab)
stable-baselines3==2.6.0
gymnasium==0.29.1
pyvirtualdisplay==3.0
# Duckietown (API gym antigua + ecosistema zuper); conviven con numpy 2.x
gym==0.25.2
pyglet==1.5.27
carnivalmirror==0.6.2
pycontracts3==7.2
PyGeometry-z6==2.1.5
zuper-commons-z6==6.2.4
zuper-typing-z6==6.2.3
zuper-ipce-z6==6.1.2
zuper-nodes-z6==6.2.17
aido-protocols-daffy==6.1.1
duckietown-serialization-ds1==1.1.1
duckietown-world-daffy==6.4.3
# gym-duckietown: instalar aparte con --no-deps desde git @daffy
#   pip install --no-build-isolation --no-deps git+https://github.com/duckietown/gym-duckietown.git@daffy
# + PARCHE Python 3.12: try/except en setattr(TypeVar,...) de zuper_typing/monkey_patching_typing.py
# numpy / torch / scipy / opencv: NO pinear -> usar los de Colab (numpy 2.x)
```
