---

````markdown
# Stem Splitter AI (PyTorch UNet)

A deep learning music source separation model that splits a song into:

- Vocals  
- Drums  
- Bass  
- Other instruments  

Built with **PyTorch**, trained on **MUSDB-HQ**, using a **U-Net mask model on spectrograms**.

---

## Features

- Stereo source separation
- UNet-based mask prediction
- Trainable on custom datasets
- Resume training from checkpoints
- GPU accelerated (CUDA)

---

## Requirements

Python 3.10+ recommended

```bash
pip install torch torchaudio tqdm soundfile numpy
````

If you have an NVIDIA GPU:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset (Training)

This project uses **MUSDB-HQ**.

Expected folder structure:

```
musdbhq/
 └── train/
      ├── TrackName1/
      │    ├── mixture.wav
      │    ├── vocals.wav
      │    ├── drums.wav
      │    ├── bass.wav
      │    └── other.wav
      └── ...
```

Set the dataset path in `src/train.py`:

```python
musdb_root = r"D:\Datasets\musdbhq"
```

---

## Training

Start training:

```bash
cd src
python train.py
```

### Default training settings (clarity run)

| Setting            | Value     |
| ------------------ | --------- |
| Sample rate        | 44100 Hz  |
| Segment length     | 6 seconds |
| FFT size (`n_fft`) | 1024      |
| Hop size           | 256       |
| Batch size         | 2         |
| Epochs             | 50        |
| Learning rate      | 1e-4      |

### Checkpoints

Saved in `src/checkpoints/`

| File              | Purpose                                              |
| ----------------- | ---------------------------------------------------- |
| `unetmask_epX.pt` | Model snapshot at epoch X (use for inference)        |
| `last.pt`         | Latest training state (used only to resume training) |

Training can be stopped anytime with **Ctrl+C** and resumed by running `python train.py` again.

---

## 🎧 Inference (Separate a Song)

Use a trained checkpoint:

```bash
cd src
python infer.py "path/to/song.mp3" --ckpt checkpoints/unetmask_ep20.pt --n_fft 1024 --hop 256
```

Outputs will be saved in:

```
src/separated_custom/song_name/
    vocals.wav
    drums.wav
    bass.wav
    other.wav
    mixture.wav
```

**Important:** `--n_fft` and `--hop` must match the settings used during training.

---

## Sharing a Trained Model

When you are happy with model quality:

1. Go to **GitHub → Releases**
2. Upload a file like:

   ```
   unetmask_musdbhq_1024_256_ep40.pt
   ```
3. In release notes include:

   * Sample rate (44100)
   * n_fft / hop values
   * Number of epochs

Users can then run inference **without training**.

---

## 🛠 Project Structure

```
src/
 ├── train.py      # Training loop
 ├── infer.py      # Song separation script
 ├── model.py      # UNet architecture
 ├── datasets.py   # MUSDB dataset loader
 └── checkpoints/  # Saved model weights
```

---

## Expected Training Progress

Typical loss trend:

| Epoch | Loss       |
| ----- | ---------- |
| 1     | ~0.18      |
| 10    | ~0.10      |
| 30    | ~0.06      |
| 50    | ~0.04–0.07 |

Separation quality improves noticeably after 20+ epochs.

---

## GPU Tips

Check GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Training on CPU will be **very slow**.

---

## License

MIT License — free to use and modify.

````

---

### Then push it

```bash
git add README.md
git commit -m "Add project README"
git push
````

* audio demo section
* model download badge
* training progress graph
  to make the repo look 🔥
