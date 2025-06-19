# Therasync Physio Pipeline

> **Purpose** – A minimal yet complete Python/Poetry stack for loading, preprocessing, analyzing and visualizing multi-modal physiological recordings collected in the **Therasync** research programme.

---

## ✨ Key facts

* **Project name:** Therasync Physiological Data Pipeline  
* **Scope:** Resting-state & in-session EDA, BVP, Temperature, HR signals ➜ synchrony metrics & Dyadic Poincaré Plot Analysis (DPPA).  
* **Article that sparked this repo:** [Elucidating Interpersonal Cardiac Synchrony During a Naturalistic Social Interaction Using Dyadic Poincaré Plot Analysis](https://ieeexplore.ieee.org/document/10778537).  
* **Licence** **The Unlicense** – public domain; fork, copy, break, remix, commercialize, do anything. No warranty.

---

## 🙌 Contributors
https://github.com/lenaadel  
https://github.com/patricefortin  
https://github.com/marvelchris21  
https://github.com/Ramdam17

---

## 🔍 Repository layout

```
.
├── data/                     # raw, intermediate & output artefacts
│   ├── raw/                  # original .xlsx sensor dumps
│   ├── output/
│   │   ├── processed/        # pickled Session objects
│   │   └── dppa/             # pickled DPPA results
│   └── coi_structure.json    # mapping file (see below)
├── notebooks/
│   ├── 01_LoadData.ipynb(†)  # load + preprocess + epoch
│   ├── 02_ApplyDPPA.ipynb    # run DPPA + clustering
│   └── 03_PlotResults.ipynb  # exploratory visualisations
└── src/
    ├── session.py            # Session orchestration class
    ├── physio_recording.py   # low-level I/O & preprocessing
    ├── dppa.py               # DPPA algorithm wrapper
    ├── visualization.py      # convenience plotting helpers
    ├── subject.py, helpers.py
    └── __init__.py
```

---

## 📂 Data formats

### 1. Raw Excel (.xlsx) files  
Each *segmented* recording contains at least the following **eight sheets**:

```
EDA_rs, EDA_session,
BVP_rs, BVP_session,
TEMP_rs, TEMP_session,
HR_rs,  HR_session
```

Loading logic (see `PhysioRecording.load_raw_data`) assumes:  

* **Row 0, Col 0 = sampling rate** (integer Hz).  
* **Rows 2 … N = signal values** (single column, cast to float).  
* Metadata such as number of samples, duration, etc., are injected on the fly.

### 2. `coi_structure.json`  
Maps a *Cohorts-of-Interest* (family × session × role) to its file on disk:

```jsonc
{
  "session_code": "fam4_session_2",
  "family"      : 4,
  "session"     : 2,
  "sensor"      : "1723456789_B12CDE",
  "role"        : "MOTHER",
  "role_id"     : 1,
  "index"       : 0
}
```

The helper `extract_raw_pathname_from_coi_structure()` resolves paths automatically (see notebook 01 for examples).

---

## 🛠️ Installation (Poetry)

```bash
# 1. Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# 2. Grab the code
git clone https://github.com/your-org/therasync-pipeline.git
cd therasync-pipeline

# 3. Resolve & lock all dependencies
poetry install

# 4. (Optional) open a sub-shell
poetry shell
```

---

## 🚀 Usage

### Interactive notebooks

```bash
poetry run jupyter lab  # launches Jupyter in the locked env
```

* Open `notebooks/01_LoadData.ipynb` → step through cells.  
* Proceed to `02_ApplyDPPA.ipynb`, then `03_PlotResults.ipynb` for visual QC.

---


## 📄 Licence

> **This is free and unencumbered software released into the public domain** (The Unlicense).  

---

Enjoy – and feel free to open an issue if you break something or make it better!
