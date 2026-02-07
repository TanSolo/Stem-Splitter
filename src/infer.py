import os
import argparse
import torch
import torchaudio
import soundfile as sf

from model import UNetMask

STEMS = ["vocals", "drums", "bass", "other"]


def load_audio(path, target_sr=44100):
    wav, sr = sf.read(path, always_2d=True)  # [T, C]
    wav = torch.from_numpy(wav).float().T    # [C, T]

    # ensure stereo
    if wav.size(0) == 1:
        wav = wav.repeat(2, 1)
    elif wav.size(0) > 2:
        wav = wav[:2]

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    return wav, target_sr


def stft_complex(x, n_fft=1024, hop=256):
    """
    x: [B, 2, T]
    returns complex STFT [B, 2, F, TT]
    """
    B, C, T = x.shape
    window = torch.hann_window(n_fft, device=x.device)
    Xs = []
    for ch in range(C):
        X = torch.stft(
            x[:, ch],
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            return_complex=True,
            center=True,
        )
        Xs.append(X)
    return torch.stack(Xs, dim=1)


def istft_complex(X, length, n_fft=512, hop=128):
    """
    X: [B, 2, F, TT] complex
    returns time audio [B, 2, T]
    """
    B, C, F, TT = X.shape
    window = torch.hann_window(n_fft, device=X.device)
    xs = []
    for ch in range(C):
        x = torch.istft(
            X[:, ch],
            n_fft=n_fft,
            hop_length=hop,
            window=window,
            length=length,
            center=True,
        )
        xs.append(x)
    return torch.stack(xs, dim=1)


@torch.no_grad()
def separate_song(model, mix_2ch, device, sr=44100, segment_seconds=6.0, overlap=0.5, n_fft=1024, hop=256):
    model.eval()

    T = mix_2ch.size(1)
    seg_len = int(segment_seconds * sr)
    step = max(1, int(seg_len * (1.0 - overlap)))

    outs = {s: torch.zeros(2, T, dtype=torch.float32) for s in STEMS}
    weight = torch.zeros(T, dtype=torch.float32)

    win = torch.hann_window(seg_len).clamp_min(1e-6)   # [seg_len]
    win2 = win.unsqueeze(0)                            # [1, seg_len]

    pos = 0
    while pos < T:
        end = min(pos + seg_len, T)
        chunk = mix_2ch[:, pos:end]
        chunk_len = chunk.size(1)

        if chunk_len < seg_len:
            chunk = torch.nn.functional.pad(chunk, (0, seg_len - chunk_len))

        x = chunk.unsqueeze(0).to(device)  # [1, 2, seg_len]

        # complex STFT of mixture
        X = stft_complex(x, n_fft=n_fft, hop=hop)       # [1, 2, F, TT]
        mag = X.abs()                                   # [1, 2, F, TT]

        # predict masks
        masks = model(mag)                               # [1, 4, 2, Fp, TTp]

        # Keep frequency bins compatible for istft; align only TT and pad/crop F if needed
        TTt = min(mag.size(-1), masks.size(-1))
        Xc = X[:, :, :, :TTt]                            # [1,2,F,TTt]
        maskc = masks[:, :, :, :, :TTt]                  # [1,4,2,Fp,TTt]

        Fm = Xc.size(-2)
        Fp = maskc.size(-2)
        if Fp < Fm:
            pad_f = Fm - Fp
            maskc = torch.nn.functional.pad(maskc, (0, 0, 0, pad_f))  # pad freq
        elif Fp > Fm:
            maskc = maskc[:, :, :, :Fm, :]

        stem_X = maskc * Xc.unsqueeze(1)                 # [1,4,2,F,TTt]

        for i, s in enumerate(STEMS):
            y = istft_complex(stem_X[:, i], length=seg_len, n_fft=n_fft, hop=hop).squeeze(0).cpu()  # [2, seg_len]

            out_slice_end = min(pos + seg_len, T)
            valid_len = out_slice_end - pos
            outs[s][:, pos:out_slice_end] += y[:, :valid_len] * win2[:, :valid_len]

        weight[pos:min(pos + seg_len, T)] += win[:min(seg_len, T - pos)]
        pos += step

    weight = weight.clamp_min(1e-6)
    for s in STEMS:
        outs[s] = outs[s] / weight.unsqueeze(0)

    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to input audio (wav/mp3).")
    ap.add_argument("--ckpt", default="checkpoints/unetmask_ep1.pt", help="Checkpoint path")
    ap.add_argument("--outdir", default="separated_custom", help="Output directory")
    ap.add_argument("--sr", type=int, default=44100)
    ap.add_argument("--segment", type=float, default=6.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--n_fft", type=int, default=1024)
    ap.add_argument("--hop", type=int, default=256)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = ap.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available; using CPU.")
            device = "cpu"

    print("Using device:", device)

    mix, sr = load_audio(args.input, target_sr=args.sr)
    print(f"Loaded: {args.input}  sr={sr}  shape={tuple(mix.shape)}")

    model = UNetMask(base=32, stems=len(STEMS)).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    print("Loaded checkpoint:", args.ckpt)

    outs = separate_song(
        model=model,
        mix_2ch=mix,
        device=device,
        sr=sr,
        segment_seconds=args.segment,
        overlap=args.overlap,
        n_fft=args.n_fft,
        hop=args.hop,
    )

    os.makedirs(args.outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.input))[0]
    out_folder = os.path.join(args.outdir, base)
    os.makedirs(out_folder, exist_ok=True)

    for s in STEMS:
        path = os.path.join(out_folder, f"{s}.wav")
        sf.write(path, outs[s].T.numpy(), sr)
        print("Wrote:", path)

    sf.write(os.path.join(out_folder, "mixture.wav"), mix.T.numpy(), sr)
    print("Done.")


if __name__ == "__main__":
    main()
