README de GitHub — FashionIA
FashionIA: Sistema de recomendación de outfits con RL + Feedback Humano (RLHF)

Autora: Paola Andrea Ospina Baracaldo
Programa: Maestría en Analítica Aplicada y Gerencia de Ingeniería
Fecha: 2025

Contenido README (Markdown)

# FashionIA: Sistema de recomendación de outfits con RL + Feedback Humano (RLHF)

[![Status](https://img.shields.io/badge/status-active-success)](./) 
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](./)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/usuario/fashionia-outfit-recommender)](./)
[![Issues](https://img.shields.io/github/issues/usuario/fashionia-outfit-recommender)](./)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./)

> **Autor:** Paola Andrea Ospina Baracaldo · **Programa:** Maestría en Analítica Aplicada y Gerencia de Ingeniería · **Última actualización:** 2025-08-13

---

## 🧭 Resumen ejecutivo

**FashionIA** es un sistema de recomendación de outfits enfocado en el **usuario individual**. Integra **aprendizaje por refuerzo (RL)** con **retroalimentación explícita** del usuario (tipo RLHF) para explorar y explotar combinaciones de prendas y, con ello, **acelerar la toma de decisiones al vestir** y **mejorar la experiencia de marca**. El proyecto prioriza **reproducibilidad**, **ética en IA** y **documentación técnica** revisable por pares académicos y equipos de ingeniería.

---

## 🎯 Objetivos del repositorio

- Publicar un **pipeline reproducible** (datos → features → entrenamiento → evaluación → despliegue local).
- Demostrar un **modelo RL con política epsilon-greedy** que equilibra exploración/explotación.
- Medir calidad con métricas de **recomendación** (Precision@K, Recall@K, NDCG, CTR simulado, tasa de aceptación).
- Exponer una **API** mínima (FastAPI) para integrar el recomendador con un front-end o cuaderno de pruebas.
- Asegurar **trazabilidad y versionamiento** (datasets, modelos, experimentos, código).
- Incluir una **guía de mantenimiento** y **buenas prácticas de ingeniería**.

---

## 📦 Estructura del repositorio

```
.
├── README.md
├── LICENSE
├── pyproject.toml / requirements.txt
├── .gitignore
├── .env.example
├── data/
│   ├── raw/                # Datos crudos (no versionados por defecto)
│   ├── interim/            # Limpiezas intermedias / features parciales
│   └── processed/          # Datos listos para modelar
├── models/                 # Modelos entrenados (pesos / artefactos)
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_training_rl.ipynb
│   └── 04_evaluation.ipynb
├── reports/
│   ├── figures/            # Gráficas y visualizaciones
│   └── tables/             # Tablas de resultados
├── src/
│   ├── fashionia/
│   │   ├── data.py         # Carga/validación de datos
│   │   ├── features.py     # Ingeniería de características
│   │   ├── policy.py       # Lógica RL (epsilon-greedy, Q-learning, etc.)
│   │   ├── recommend.py    # Generación de outfits, ranking
│   │   ├── eval.py         # Métricas (Precision@K, NDCG, CTR, A/B simulado)
│   │   ├── api.py          # FastAPI para inferencia
│   │   ├── config.py       # Manejo de parámetros desde .env / CLI
│   │   └── utils.py
│   └── scripts/
│       ├── prepare_data.py
│       ├── train.py
│       ├── evaluate.py
│       └── serve.py
├── tests/                  # Pruebas unitarias y de integración
└── Makefile                # Atajos de tareas
```

---

## 🔧 Requisitos e instalación

### Requisitos mínimos
- **Python 3.10+** (recomendado 3.11)
- Soporte **CPU** (opcional GPU)
- **Git** ≥ 2.30
- (Opcional) **conda** o **venv** para entornos virtuales

### Instalación rápida
```bash
git clone https://github.com/usuario/fashionia-outfit-recommender.git
cd fashionia-outfit-recommender
python -m venv .venv && source .venv/bin/activate  # en Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# o con poetry
# poetry install
cp .env.example .env  # Revisar variables
```

---

## 🔑 Configuración (.env)

Copie `.env.example` a `.env` y defina:

```ini
# Ruta de datos
DATA_DIR=./data
RAW_DIR=./data/raw
PROCESSED_DIR=./data/processed

# Parámetros de entrenamiento
SEED=42
EPSILON_START=0.20
EPSILON_MIN=0.05
EPSILON_DECAY=0.995
EPISODES=100
TOP_K=3

# API
API_HOST=0.0.0.0
API_PORT=8000

# Tracking de experimentos (opcional)
MLFLOW_TRACKING_URI=./mlruns
EXPERIMENT_NAME=FashionIA
```

---

## 🗂️ Datos

> **Nota:** Este repositorio **no** incluye datos sensibles ni imágenes propietarias. Use `data/raw/` para almacenar datos locales.

- **Catálogo de prendas:** CSV/Parquet con columnas sugeridas: `id`, `tipo`, `color`, `textura`, `estilo`, `ocasión`, `marca`, `precio`, `temporada`, `tags`.
- **Inventario del usuario:** mapping prenda → disponibilidad/estado/uso reciente.
- **Historial de interacción:** pares (outfit, feedback) con etiquetas {{like/dislike/neutral}} y contexto (fecha, clima opcional).
- **Esquemas validados:** `src/fashionia/data.py` valida tipos y valores.

Ejemplo de esquema mínimo (CSV):
```text
id,tipo,color,estilo,ocasion,marca,tags
101,blusa,negro,elegante,trabajo,By Paola Ospina,"basico;nocturno"
```

---

## 🧪 Cómo ejecutar

### 1) Preparar datos
```bash
python -m src.scripts.prepare_data --input data/raw --output data/processed
```

### 2) Entrenar (RL)
```bash
python -m src.scripts.train   --episodes 100   --epsilon_start 0.20 --epsilon_min 0.05 --epsilon_decay 0.995   --top_k 3   --seed 42
```

### 3) Evaluar
```bash
python -m src.scripts.evaluate --top_k 3 --metrics precision@k recall@k ndcg ctr
```

### 4) Servir API (FastAPI)
```bash
uvicorn src.fashionia.api:app --host 0.0.0.0 --port 8000 --reload
# Endpoint de prueba:
# GET http://localhost:8000/health
# POST http://localhost:8000/recommend  body={{ "user_id": "demo", "top_k": 3 }}
```

---

## 🧠 Metodología (alto nivel)

1. **Representación de prendas/outfits**: codificación de atributos semánticos (one-hot/embeddings simples).
2. **Generación de candidatos**: combinaciones viables (reglas de compatibilidad, ocasión, clima).
3. **Política RL**: `epsilon-greedy` sobre valor Q (o bandits contextuales) para elegir outfit.
4. **Feedback**: like/dislike explícito por el usuario; actualización online (Q-learning).
5. **Ranking final**: mezcla de **exploración** (nuevas combinaciones) y **explotación** (preferencias aprendidas).
6. **Evaluación**: métricas offline + pruebas A/B simuladas (tasa de aceptación/CTR).

> **Limitación**: Sin visión por computadora pesada (para hardware limitado). Se prioriza una representación tabular/semántica con **buena ingeniería de atributos**.

---

## 📊 Métricas de evaluación

- **Precision@K / Recall@K / NDCG** sobre interacciones históricas.
- **CTR simulado** (click/aceptación por outfit mostrado).
- **Tiempo de decisión** autoinformado y **satisfacción** (cuestionario corto).
- **Cobertura de catálogo** (porcentaje de prendas/outfits recomendados).

---

## 🔁 Reproducibilidad y trazabilidad

- **Semillas fijadas** (SEED=42) y control de aleatoriedad.
- Scripts **deterministas** siempre que sea posible.
- **Versionado de datos** por timestamp en `data/processed/`.
- Tracking de experimentos con **MLflow** (opt-in) y guardado de artefactos en `models/`.
- Hash de commit incluido en `reports/tables/exp_runs.csv` (pipeline escribe metadata).

---

## 🧩 API (contrato mínimo)

**POST `/recommend`**

```json
{{
  "user_id": "demo",
  "context": {{"ocasion": "trabajo", "clima": "templado"}},
  "top_k": 3
}}
```

**Respuesta**

```json
{{
  "user_id": "demo",
  "recommendations": [
    {{"outfit_id": "O-001", "items": ["101","205","330"], "score": 0.82}},
    {{"outfit_id": "O-002", "items": ["117","219","301"], "score": 0.79}},
    {{"outfit_id": "O-003", "items": ["108","240","315"], "score": 0.74}}
  ]
}}
```

---

## ✅ Calidad, pruebas y CI

- **Pruebas unitarias** en `tests/` con `pytest`.
- Validadores de esquema para los CSV/Parquet de entrada.
- Linter/format con **ruff** y **black**.
- (Opcional) **GitHub Actions**: ejecutar `pytest` y `ruff` en cada PR.

Ejemplo de workflow (`.github/workflows/ci.yml`):
```yaml
name: CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: pip install pytest ruff black
      - run: ruff check .
      - run: black --check .
      - run: pytest -q
```

---

## 🔐 Privacidad y ética en IA

- **Datos personales**: no se suben a GitHub; se mantienen locales/encriptados.
- **Minimización**: solo atributos necesarios para la tarea.
- **Sesgos**: revisión de distribución por género/edad/talla/tono de piel cuando aplique; reporte en `reports/`.
- **Transparencia**: explicación de reglas de compatibilidad y de la política de exploración.
- **Opt-out**: mecanismo para borrar historial de un usuario (`/users/{{id}}/purge`, opcional).

---

## 🗺️ Roadmap

- Integrar **bandits contextuales** (LinUCB/TS) y comparación con Q-learning.
- Añadir **restricciones blandas** (estilo, modestia, dress code).
- Soporte a **clima real** y calendario.
- Módulo de **visualización** de outfits con librerías ligeras.
- Integración con **app móvil** (PWA) y login.

---

## 📝 Convenciones y versionado

- **SemVer** para releases (`vMAJOR.MINOR.PATCH`).
- Convenciones de **branches**: `main` (estable), `dev` (integración), `feat/*`, `fix/*`.
- Commits estilo **Conventional Commits**.
- Issues y PRs con **plantillas** en `.github/`.

---

## 🤝 Contribución

Agradezco PRs y sugerencias. Consulte `CONTRIBUTING.md` y nuestro `CODE_OF_CONDUCT.md` (pendientes).

---

## 📚 Cita académica

Si usa este repositorio, cite como:

```bibtex
@misc{{fashionia_2025}},
  author       = {Paola Andrea Ospina Baracaldo},
  title        = {FashionIA: Sistema de recomendación de outfits con RL + Feedback Humano (RLHF)},
  howpublished = {GitHub repository},
  year         = {2025},
  url          = {https://github.com/usuario/fashionia-outfit-recommender}
}
```

---

## 🧾 Licencia

Este proyecto se distribuye bajo licencia **MIT**. Consulte `LICENSE` para más detalles.

---

## 📫 Contacto

**Paola Andrea Ospina Baracaldo** · Maestría en Analítica Aplicada y Gerencia de Ingeniería  
E-mail: **<tu_correo@ejemplo.com>** · LinkedIn: **<tu_linkedin>**

---

## 📎 Apéndices

- **Resultados** y tablas clave en `reports/tables/`.
- **Figuras** y gráficos en `reports/figures/`.
- **Notas de investigación** en `reports/notes/`.
- **Checklist de entrega** en `reports/checklists/README_checklist.md`.

 
Checklist de Entrega (para jurados)
•	Resumen ejecutivo claro y contextualizado al caso de uso (moda).
•	Estructura de repositorio estandarizada (data/src/notebooks/tests).
•	Guía de instalación y ejecución reproducible (venv/requirements).
•	Definición de esquemas de datos y validadores.
•	Metodología de RL y criterios de evaluación explicados en lenguaje técnico accesible.
•	Métricas de recomendación (Precision@K, NDCG) y resultados colocados en reports/.
•	Políticas de privacidad y ética explícitas.
•	Plan de pruebas y (opcional) CI con GitHub Actions.
•	Roadmap y limitaciones identificadas.
•	Licencia, citación y datos de contacto.
