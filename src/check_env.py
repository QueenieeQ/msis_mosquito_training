"""Import every package the project needs and print versions, Python and CUDA state. Exit code 1 if anything is missing."""
import importlib
import importlib.metadata as md
import sys

REQUIRED = ["torch", "torchvision", "ultralytics", "numpy", "pillow", "pyyaml", "opencv-python", "onnxruntime-gpu",
            "loguru", "mmpretrain", "mmengine", "mmcv-lite", "pytest"]
MODULE = {"pillow": "PIL", "pyyaml": "yaml", "opencv-python": "cv2", "onnxruntime-gpu": "onnxruntime", "mmcv-lite": "mmcv"}


def main():
    bad = 0
    for r in REQUIRED:
        try:
            importlib.import_module(MODULE.get(r, r))
            print(f"  ok       {r:<18}{md.version(r)}")
        except Exception as e:
            bad += 1
            print(f"  MISSING  {r:<18}{type(e).__name__}: {e}")
    try:
        import torch
        print(f"\n  python {sys.version.split()[0]} | torch CUDA {torch.version.cuda} | available={torch.cuda.is_available()}"
              f" | gpus={torch.cuda.device_count()}")
    except Exception:
        pass
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
