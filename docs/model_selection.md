# Model Selection for Kizuna Privacy Engine

This document outlines the rationale for model selection across all modalities.

## Vision Model Selection

**Selected Model**: MobileCLIP-S0 (40M parameters)

**Rationale**:
- **Latency**: ~50ms inference on 2 CPU cores (meets < 200ms total budget)
- **Accuracy**: 67.8% ImageNet top-1 (acceptable for edge deployment)
- **ONNX Compatibility**: Excellent export support via Hugging Face
- **Multimodal**: Trained with contrastive learning (CLIP), produces semantically meaningful embeddings
- **Size**: 160MB FP32 → ~40MB INT8 (fits edge memory constraints)
- **Embedding Dimension**: 512D (standard for multimodal fusion)

**Alternatives Considered**:
- CLIP-ViT-B/16: Higher accuracy but 2-3× slower on CPU
- EfficientNet-B0: Fast but lacks multimodal alignment
- MobileNetV3-Small: Faster but classification-focused, not embedding-focused

**Download Source**: Hugging Face Hub (`apple/mobileclip-s0`)

---

## Audio Model Selection

**Selected Model**: AudioCLIP (49M parameters, audio encoder only)

**Rationale**:
- **Multimodal Alignment**: Trained jointly with vision and text (CLIP-like), produces embeddings in the same semantic space as MobileCLIP
- **Latency**: ~80ms inference on 2 CPU cores for 1-second audio chunk
- **Input Format**: Mel-spectrogram (128 mel bins, 16kHz, 1-second windows)
- **ONNX Compatibility**: PyTorch model with standard conv/transformer layers (good export support)
- **Embedding Dimension**: 512D (matches vision embeddings for fusion)
- **Edge-Friendly**: Smaller than full AudioCLIP (we only use audio encoder, not text/vision)

**Alternatives Considered**:
- **AST (Audio Spectrogram Transformer)**: High accuracy but 86M parameters, slower inference
- **VGGish**: Fast but only 128D embeddings, not trained for multimodal alignment
- **PANNs (Pretrained Audio Neural Networks)**: Good for sound event detection but lacks multimodal training
- **Wav2Vec2**: Excellent for speech but focused on linguistic features, not general audio semantics

**Download Source**: 
- Hugging Face Hub (`laion/clap-htsat-unfused`) as inspiration
- Or custom AudioCLIP implementation from official repository

**Preprocessing**:
- Resample audio to 16kHz
- Compute 128-bin mel-spectrogram with n_fft=2048, hop_length=512
- Normalize to zero mean, unit variance

---

## Sensor MLP Encoder

**Architecture**: 2-layer MLP with ReLU activation

**Rationale**:
- **Simplicity**: Environmental sensor data is low-dimensional (5-10 features), doesn't require deep models
- **Latency**: < 5ms inference on CPU
- **Trainability**: Small enough to train from scratch on synthetic data
- **Alignment**: Trained with contrastive loss against video/audio embeddings from matching timestamps

**Architecture Details**:
```
Input: [temperature, humidity, motion, light, air_quality] (5D)
       ↓ [normalize to [0, 1]]
Layer 1: Linear(5 → 64) + ReLU
Layer 2: Linear(64 → 64) + ReLU
Layer 3: Linear(64 → 512)  # Project to embedding space
Output: 512D embedding (L2-normalized)
```

**Training**:
- Loss: Contrastive loss (InfoNCE) with video/audio embeddings as anchors
- Dataset: 100,000+ synthetic temporal-aligned (video, audio, sensor) triplets
- Optimizer: AdamW with learning rate 1e-3
- Epochs: 50 with early stopping

---

## Fusion Module

**Architecture**: Late fusion with learned projection head

**Rationale**:
- **Flexibility**: Handles missing modalities gracefully (video-only, audio-only, all modalities)
- **Interpretability**: Each modality's embedding can be analyzed independently
- **Simplicity**: Concatenate embeddings → Linear projection → L2 normalize

**Architecture Details**:
```
Input: 
  - vision_emb: 512D (optional)
  - audio_emb: 512D (optional)
  - sensor_emb: 512D (optional)

Concatenation:
  - If all present: 1536D (512×3)
  - If video+audio: 1024D (512×2)
  - If video only: 512D (pass-through)

Projection Head:
  - Linear(input_dim → 512)
  - L2 normalize

Output: 512D unified embedding
```

**Alternative Considered**:
- **Early Fusion**: Fuse raw data before encoding. Rejected due to: (1) incompatible data types (pixels, waveforms, scalars), (2) loss of modularity.
- **Attention-Based Fusion**: Cross-modal attention. Rejected due to: (1) higher latency (transformer layers), (2) overkill for 3 modalities.

---

## Quantization Strategy

**Target**: INT8 quantization for all models to meet edge constraints

**Method**: ONNX Runtime dynamic quantization + static quantization with calibration

**Process**:
1. Export FP32 PyTorch model to ONNX
2. Run `onnxsim` to optimize graph
3. Apply `quantize_dynamic` for weights-only quantization (quick baseline)
4. Apply `quantize_static` with calibration dataset (500+ representative samples) for full INT8
5. Validate accuracy: FP32 vs. INT8 cosine similarity > 0.95

**Expected Results**:
- Size: 4× reduction (FP32 → INT8)
- Latency: 1.5-2× improvement on CPU
- Accuracy: < 2% drop in anomaly detection recall

---

## Performance Targets

| Component | Target Latency (2 CPU cores) | Target Memory | Embedding Dim |
|-----------|------------------------------|---------------|---------------|
| Vision Encoder | < 60ms | < 200MB | 512D |
| Audio Encoder | < 80ms | < 150MB | 512D |
| Sensor Encoder | < 5ms | < 10MB | 512D |
| Fusion Module | < 10ms | < 20MB | 512D |
| **Total** | **< 155ms** | **< 380MB** | **512D** |

**Note**: Total embedding latency budget is 200ms, leaving 45ms margin for preprocessing and noise addition.

---

## Model Versioning

All models are tracked with DVC (Data Version Control):
- `models/vision/model.onnx` (FP32 baseline)
- `models/vision/model_int8.onnx` (INT8 quantized, production)
- `models/audio/model.onnx` (FP32 baseline)
- `models/audio/model_int8.onnx` (INT8 quantized, production)
- `models/sensor/model.onnx` (FP32 baseline)
- `models/sensor/model_int8.onnx` (INT8 quantized, production)
- `models/fusion/projection.onnx` (FP32, small model, no quantization needed)

Each model file is accompanied by:
- `model_card.md`: Metadata (architecture, training data, performance, license)
- `calibration_data.npz`: Calibration dataset for quantization reproducibility

---

## Licenses

- **MobileCLIP**: Apache 2.0 (Apple Inc.)
- **AudioCLIP**: MIT License (LAION)
- **Sensor MLP**: Apache 2.0 (trained by us)
- **Fusion Module**: Apache 2.0 (trained by us)

All models are compatible with commercial use and redistribution.
