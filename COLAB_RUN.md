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

In Colab, create a shortcut for that folder in `MyDrive` named `mimic3_data`, then run:

```python
from google.colab import drive
from pathlib import Path
import os

drive.mount("/content/drive")

MIMIC_III_DATA_DIR = Path("/content/drive/MyDrive/mimic3_data")
os.environ["MIMIC_III_DATA_DIR"] = str(MIMIC_III_DATA_DIR)

if MIMIC_III_DATA_DIR.exists():
    print("MIMIC-III data directory found:", MIMIC_III_DATA_DIR)
else:
    print("MIMIC-III data directory was not found.")
    print("Add a Drive shortcut named mimic3_data that points to:")
    print("https://drive.google.com/drive/folders/1s4F5qFepqFJVYJU3ii62RWSYy2Z8bGAF")
```

## 6. Sanity-check MIMIC-III files

```python
from pathlib import Path
import os

data_dir = Path(os.environ.get("MIMIC_III_DATA_DIR", "/content/drive/MyDrive/mimic3_data"))

if data_dir.exists():
    for path in sorted(data_dir.iterdir())[:40]:
        print(path.name)
else:
    print("Data directory is not available yet:", data_dir)
```

## 7. Auto-detect and run a Python entrypoint

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
