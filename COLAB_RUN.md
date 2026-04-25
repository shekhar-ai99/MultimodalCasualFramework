# Google Colab Run Guide

This guide gives you ready-to-paste Google Colab cells for running the project from a fresh notebook.

## 1. Clone the repository

```python
REPO_URL = "https://github.com/shekhar-ai99/MultimodalCasualFramework.git"
REPO_DIR = "/content/MultimodalCasualFramework"

import os

if not os.path.exists(REPO_DIR):
    !git clone {REPO_URL} {REPO_DIR}
else:
    %cd {REPO_DIR}
    !git pull

%cd {REPO_DIR}
```

## 2. Install dependencies

```python
import os
from pathlib import Path

requirements = Path("requirements.txt")

if requirements.exists():
    try:
        !pip install -r requirements.txt
    except Exception as exc:
        print("requirements.txt install failed; trying a minimal fallback install.")
        print(exc)
        !pip install numpy pandas scikit-learn matplotlib torch pyyaml
else:
    !pip install numpy pandas scikit-learn matplotlib torch pyyaml
```

## 3. Check GPU availability

```python
import torch

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Runtime > Change runtime type > Hardware accelerator > GPU")
```

## 4. List repository files

```python
from pathlib import Path

for path in sorted(Path(".").glob("*")):
    print(path)
```

## 5. Mount Google Drive and set up MIMIC-III data

Your shared Google Drive folder is:

```text
https://drive.google.com/drive/folders/1s4F5qFepqFJVYJU3ii62RWSYy2Z8bGAF
```

If the folder is in `Shared with me`, add it as a shortcut to `MyDrive`. The cell below first checks common folder names, then searches your Drive for `ADMISSIONS.csv` and uses that folder automatically.

```python
from google.colab import drive
from pathlib import Path
import os

try:
    drive.mount("/content/drive", force_remount=True, timeout_ms=120000)
except ValueError as exc:
    print("Google Drive mount failed.")
    print("Try Runtime > Restart session, then run this cell again.")
    print("If the browser asks for Google authorization, allow access.")
    raise exc

CORE_MIMIC_CSVS = [
    "ADMISSIONS.csv",
    "ICUSTAYS.csv",
    "PATIENTS.csv",
    "D_LABITEMS.csv",
    "DIAGNOSES_ICD.csv",
]

candidate_dirs = [
    Path("/content/drive/MyDrive/mimic3"),
    Path("/content/drive/MyDrive/MIMIC3"),
    Path("/content/drive/MyDrive/mimic3_data"),
    Path("/content/drive/MyDrive/MIMIC-III"),
]

MIMIC_III_DATA_DIR = next(
    (path for path in candidate_dirs if (path / "ADMISSIONS.csv").exists()),
    None,
)

if MIMIC_III_DATA_DIR is None:
    search_roots = [
        Path("/content/drive/MyDrive"),
        Path("/content/drive/Shareddrives"),
    ]
    for root in search_roots:
        if not root.exists():
            continue
        matches = list(root.rglob("ADMISSIONS.csv"))
        if matches:
            MIMIC_III_DATA_DIR = matches[0].parent
            break

if MIMIC_III_DATA_DIR is None:
    MIMIC_III_DATA_DIR = Path("/content/drive/MyDrive/mimic3")

os.environ["MIMIC_III_DATA_DIR"] = str(MIMIC_III_DATA_DIR)

if MIMIC_III_DATA_DIR.exists():
    print("MIMIC-III data directory found:", MIMIC_III_DATA_DIR)
else:
    print("MIMIC-III data directory was not found.")
    print("Add the shared Drive folder as a shortcut in MyDrive, or update MIMIC_III_DATA_DIR manually.")
    print("https://drive.google.com/drive/folders/1s4F5qFepqFJVYJU3ii62RWSYy2Z8bGAF")
```

## 6. Sanity-check MIMIC-III files

```python
from pathlib import Path
import os

data_dir = Path(os.environ.get("MIMIC_III_DATA_DIR", "/content/drive/MyDrive/mimic3"))
expected_csvs = globals().get("CORE_MIMIC_CSVS", [
    "ADMISSIONS.csv",
    "ICUSTAYS.csv",
    "PATIENTS.csv",
    "D_LABITEMS.csv",
    "DIAGNOSES_ICD.csv",
])

if data_dir.exists():
    print("MIMIC-III directory:", data_dir)
    print("First files:")
    for path in sorted(data_dir.glob("*.csv"))[:40]:
        print("-", path.name)

    missing = [name for name in expected_csvs if not (data_dir / name).exists()]
    if missing:
        print("Missing expected MIMIC-III CSV files:")
        for name in missing:
            print("-", name)
    else:
        print("Found expected core MIMIC-III CSV files.")
else:
    print("Data directory is not available yet:", data_dir)
```

## 7. Create a small MIMIC-III CSV sample

This creates a local sample folder capped at 10 MB total, then points `MIMIC_III_DATA_DIR` to that sample. Change `SAMPLE_TOTAL_MB` to `5` if you want a 5 MB cap.

```python
from pathlib import Path
import os

source_dir = Path(os.environ.get("MIMIC_III_DATA_DIR", "/content/drive/MyDrive/mimic3"))
sample_dir = Path("/content/mimic3_sample")
sample_dir.mkdir(parents=True, exist_ok=True)

core_files = globals().get("CORE_MIMIC_CSVS", [
    "ADMISSIONS.csv",
    "ICUSTAYS.csv",
    "PATIENTS.csv",
    "D_LABITEMS.csv",
    "DIAGNOSES_ICD.csv",
])
all_csv_files = sorted(path.name for path in source_dir.glob("*.csv"))
sample_files = list(dict.fromkeys(core_files + all_csv_files))
SAMPLE_TOTAL_MB = 10
total_limit = SAMPLE_TOTAL_MB * 1024 * 1024
per_file_limit = max(total_limit // max(len(sample_files), 1), 128 * 1024)

def copy_csv_head(src, dst, byte_limit):
    copied = 0
    with src.open("rb") as source, dst.open("wb") as target:
        header = source.readline()
        target.write(header)
        copied += len(header)
        for line in source:
            if copied + len(line) > byte_limit:
                break
            target.write(line)
            copied += len(line)
    return copied

total_copied = 0
for name in sample_files:
    remaining = total_limit - total_copied
    if remaining <= 0:
        break
    src = source_dir / name
    dst = sample_dir / name
    if not src.exists():
        print("Skipping missing file:", name)
        continue
    copied = copy_csv_head(src, dst, min(per_file_limit, remaining))
    total_copied += copied
    print(f"Sampled {name}: {copied / (1024 * 1024):.2f} MB")

os.environ["MIMIC_III_FULL_DATA_DIR"] = str(source_dir)
os.environ["MIMIC_III_DATA_DIR"] = str(sample_dir)

print("Full MIMIC-III directory:", source_dir)
print("Sample MIMIC-III directory:", sample_dir)
print(f"Total sample size: {total_copied / (1024 * 1024):.2f} MB")
```

## 8. Auto-detect and run a Python entrypoint

```python
from pathlib import Path
import runpy

candidate_entrypoints = [
    "main.py",
    "train.py",
    "run.py",
    "src/main.py",
    "src/train.py",
]

entrypoint = next((Path(path) for path in candidate_entrypoints if Path(path).exists()), None)

if entrypoint is None:
    print("No common Python entrypoint found. Repository Python files:")
    for path in sorted(Path(".").rglob("*.py")):
        print(path)
else:
    print("Running:", entrypoint)
    runpy.run_path(str(entrypoint), run_name="__main__")
```
