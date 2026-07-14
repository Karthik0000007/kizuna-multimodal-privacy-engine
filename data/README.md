# Kizuna Data Directory

This directory contains datasets for training, testing, and evaluation.

## Structure

```
data/
├── raw/                    # Raw datasets (not tracked in git)
│   ├── up-fall/           # UP-Fall Detection Dataset
│   └── esc-50/            # ESC-50 Audio Dataset
├── processed/             # Processed datasets
│   ├── baselines/         # Anomaly baseline embeddings
│   └── calibration/       # Calibration data for DP sensitivity
├── cache/                 # Temporary cache
└── faiss_index_*.bin      # FAISS vector indices (not tracked)
```

## Datasets

### UP-Fall Detection Dataset

**Source**: http://www.up.ac.za/upfall

Fall detection dataset with accelerometer and camera data.

**Download**:
```bash
python scripts/prepare_datasets.py --dataset up-fall
```

**License**: Research use only

### ESC-50 Environmental Sound Classification

**Source**: https://github.com/karolpiczak/ESC-50

Environmental audio samples for sound anomaly detection.

**Download**:
```bash
python scripts/prepare_datasets.py --dataset esc-50
```

**License**: Creative Commons Attribution Non-Commercial

### UrbanSound8K (Optional)

**Source**: https://urbansounddataset.weebly.com/urbansound8k.html

Urban sound classification dataset.

**License**: Creative Commons Attribution Non-Commercial

## Usage

### Data Pipeline

1. **Download raw datasets**:
   ```bash
   python scripts/prepare_datasets.py --all
   ```

2. **Generate synthetic data** (for development):
   ```bash
   python src/ingestion/video_simulator.py --duration 60 --output data/processed/
   ```

3. **Create baseline embeddings**:
   ```bash
   python scripts/collect_baseline.py --scenario elderly_care --duration 24h
   ```

## Privacy Note

**Raw datasets containing PII must never be committed to the repository.**

Only processed embeddings and anonymized metadata should be tracked in version control.

## Citation

If using these datasets, please cite the original authors:

```bibtex
@inproceedings{upfall,
  title={UP-Fall Detection Dataset},
  author={...},
  year={2019}
}

@dataset{esc50,
  title={ESC-50: Dataset for Environmental Sound Classification},
  author={Piczak, Karol J.},
  year={2015}
}
```
