# Building Age Classification from Satellite Imagery

**DSRH2026 hackathon project**

This project classifies the age of buildings from Landsat time-series data. We train on labelled pixels from Madrid, then transfer the learned model to Amsterdam using a limited number of Amsterdam labels per class. The final recommended solution is in **Notebook 8**.

## At a glance

| Item | Description |
|---|---|
| Task | Classify a building-containing 30 m pixel into one of four construction-age classes |
| Input | Six Landsat spectral bands observed over time |
| Source city | Madrid |
| Target city | Amsterdam |
| Few-shot budgets | 5, 25, 50, 100, or 200 labelled Amsterdam pixels per class |
| Evaluation metric | Macro-F1 |

The class boundaries are city-specific to reflect local construction history. The notebooks define the exact labels and evaluation protocol.

## Approach

The final notebook investigates why a model trained in Madrid does not transfer directly to Amsterdam, then develops features and models for the cross-city setting. It compares candidate Madrid-trained models using spatial validation and evaluates Amsterdam adaptation at each label budget. The experiments include temporal and neighbourhood features, ensembles, spatial smoothing, and self-training. Notebook 8 presents the end-to-end workflow and its evaluation.

## Which notebook should I use?

**Start with Notebook 8 for the final recommended model.** It is the self-contained end-to-end submission: it loads the two data files, explores the city-to-city differences, builds transfer-oriented features, compares candidate models, evaluates few-shot adaptation, and provides the final model and reusable transfer workflow. It does not require running Notebooks 1–7 first or using generated feature caches.

| Notebook | Purpose |
|---|---|
| [1 - Introduction](1-Introduction%20%28DSRH2026%29.ipynb) | Organizers' challenge introduction, task, and rules |
| [2 - Reading Data](2-Reading_Data%20%28DSRH2026%29.ipynb) | Organizers' starter data exploration |
| [3 - Preprocessing](3-Preprocessing%20%28DSRH2026%29.ipynb) | Organizers' preprocessing baseline |
| [4 - Modelling](4-Modelling%20%28DSRH2026%29.ipynb) | Organizers' baseline model and evaluation |
| [5 - Baseline Model Training](5-Baseline%20Model%20Training%28Our%20first%20model%29%28DSRH2026%29.ipynb) | Our first transfer-learning model |
| [6 - Model Zoo and Transfer Learning](6-Model_Zoo_Transfer%20Learning%282nd%20model%20training%29%28DSRH2026%29.ipynb) | Our comparison of candidate Stage 1 model families |
| [7 - Ensemble, Spatial Smoothing, and Self-Training](7-Ensemble%2BSpatial%20Smoothing%20%2B%20Self%20Training%20%28Third%20Model%20Training%29%28DSRH2026%29.ipynb) | Our third-model experiments and improvements |
| [8 - Final Budget Transfer Learning Model](8-Final_Model_Budget%20Transfer%20Learning%20Model%20%28DSRH2026%29.ipynb) | **Our final recommended, self-contained pipeline** |

[METRIC_LEARNING_APPROACH.md](METRIC_LEARNING_APPROACH.md) is an organizers' suggested metric-learning direction. It is background material, not the approach used by our final model.

## Data

The repository expects these files in `data/`:

- `madrid_train.parquet` - labelled source-city data
- `amsterdam_data.parquet` - target-city data used for few-shot adaptation and evaluation

The parquet files are stored with Git LFS. If they are not present after cloning, install Git LFS and pull them:

```bash
git lfs install
git lfs pull
```

Generated caches, trained model files, and prediction results are not committed. The `data/cache/`, `models/`, and `results/` folders are kept with placeholders and will be populated as needed when the notebooks run. Notebook checkpoints and VS Code settings are also ignored.

## Setup and run

Use Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
```

Activate the environment, then install the project dependencies and start Jupyter:

```bash
# Windows PowerShell
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

Open Notebook 8 and run its cells from top to bottom to reproduce the final workflow and evaluation. It reads the parquet files directly and can create reusable intermediate caches and result files during a run. CatBoost and PyTorch are optional dependencies for optional model comparisons; the core requirements are listed in `requirements.txt`.

Notebooks 5–7 document our earlier model iterations. Notebook 6 depends on features and models from Notebook 5; Notebook 7 builds on earlier experiments. They are not prerequisites for Notebook 8.

## Repository contents

```text
.
├── 1-Introduction (DSRH2026).ipynb
├── 2-Reading_Data (DSRH2026).ipynb
├── 3-Preprocessing (DSRH2026).ipynb
├── 4-Modelling (DSRH2026).ipynb
├── 5-Baseline Model Training(Our first model)(DSRH2026).ipynb
├── 6-Model_Zoo_Transfer Learning(2nd model training)(DSRH2026).ipynb
├── 7-Ensemble+Spatial Smoothing + Self Training (Third Model Training)(DSRH2026).ipynb
├── 8-Final_Model_Budget Transfer Learning Model (DSRH2026).ipynb
├── METRIC_LEARNING_APPROACH.md
├── data/
│   └── cache/       # Generated when needed
├── models/          # Generated when needed
├── results/         # Generated when needed
└── requirements.txt
```
