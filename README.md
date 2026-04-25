# ICU Causal RL with TGN + CQL + Conformal XAI

## Google Colab Quickstart
Run the project in Colab with the single notebook runner:
[MultimodalCasualFramework_Colab.ipynb](MultimodalCasualFramework_Colab.ipynb)

For copy-paste cells, including Google Drive and MIMIC-III setup, see:
[COLAB_RUN.md](COLAB_RUN.md)

## Overview
Implementation scaffold for:
"A Multimodal Causal Framework for Personalized ICU Interventions"

## Features
- Temporal Graph Networks (TGN)
- Conservative Q-Learning (CQL)
- Conformal Prediction
- Uncertainty-Gated XAI
- Evaluation and visualization artifact generation

## Setup
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py
```

## Outputs
Running `python main.py` produces:
- `results/logs/metrics.json`
- `results/figures/loss_curve.png`
- `results/figures/q_value_distribution.png`
- `results/figures/action_distribution.png`
- `results/figures/conformal_set_sizes.png`
- `results/figures/patient_trajectory.png`

## Dataset
- MIMIC-IV (PhysioNet credentialed access required).
- Keep raw extracts in `data/raw/` (excluded from versioned artifacts).

## Next Steps
1. Swap dummy dataloader for processed MIMIC tensors in `src/data/loader.py`.
2. Expand `src/training/evaluate.py` with offline policy evaluation estimators.
3. Replace dummy AUROC label with a clinical outcome label from the dataset.
