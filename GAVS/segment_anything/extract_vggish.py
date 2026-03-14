"""Extract VGGish embeddings from raw audio.wav files.

Usage:
    python extract_vggish.py --ver v1m
    python extract_vggish.py --ver v1s
    python extract_vggish.py --ver v2
"""
import os
import argparse
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ver", type=str, required=True, choices=["v1s", "v1m", "v2"])
    parser.add_argument("--data_dir", type=str, default="../../data/AVS")
    parser.add_argument("--out_dir", type=str, default="./feature_extract")
    args = parser.parse_args()

    num_frames = 10 if args.ver == "v2" else 5
    data_path = os.path.join(args.data_dir, args.ver)
    out_path = os.path.join(args.out_dir, f"{args.ver}_vggish_embs")
    os.makedirs(out_path, exist_ok=True)

    # Load VGGish
    model = torch.hub.load("harritaylor/torchvggish", "vggish", trust_repo=True)
    model.eval()

    vids = sorted(os.listdir(data_path))
    total = len(vids)
    skipped, done, failed = 0, 0, 0

    for i, vid in enumerate(vids):
        out_file = os.path.join(out_path, f"{vid}.npy")
        if os.path.exists(out_file) and os.path.getsize(out_file) > 0:
            skipped += 1
            continue

        audio_path = os.path.join(data_path, vid, "audio.wav")
        if not os.path.exists(audio_path):
            print(f"[{i+1}/{total}] SKIP {vid}: no audio.wav")
            failed += 1
            continue

        try:
            with torch.no_grad():
                emb = model.forward(audio_path)  # [T, 128]

            emb_np = emb.cpu().numpy().astype(np.float32)

            # Pad or truncate to num_frames
            if emb_np.shape[0] < num_frames:
                pad = np.zeros((num_frames - emb_np.shape[0], 128), dtype=np.float32)
                emb_np = np.concatenate([emb_np, pad], axis=0)
            elif emb_np.shape[0] > num_frames:
                emb_np = emb_np[:num_frames]

            np.save(out_file, emb_np)
            done += 1
            if (i + 1) % 50 == 0:
                print(f"[{i+1}/{total}] done={done} skip={skipped} fail={failed}")
        except Exception as e:
            print(f"[{i+1}/{total}] FAIL {vid}: {e}")
            failed += 1

    print(f"\nDone! total={total} extracted={done} skipped={skipped} failed={failed}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
