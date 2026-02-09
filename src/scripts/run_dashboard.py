from __future__ import annotations

import argparse
import socket
from pathlib import Path

import uvicorn

from crinv.dashboard_app import create_app
from crinv.config import InverseDesignConfig
from crinv.surrogate_interface import CRReconSurrogate, MockSurrogate


def _local_ip() -> str:
    try:
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)
    except Exception:
        return "127.0.0.1"


def main() -> int:
    p = argparse.ArgumentParser(prog="run_dashboard")
    p.add_argument("--progress-dir", default="data/progress", help="Directory containing metrics.jsonl/topk_step-*.npz")
    p.add_argument("--config", default="configs/inverse.yaml", help="Config used to locate paths.yaml for surrogate")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--surrogate",
        choices=["auto", "crrecon", "mock", "none"],
        default="auto",
        help="Forward surrogate used for spectrum plot",
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8501)
    args = p.parse_args()

    progress_dir = Path(args.progress_dir)
    surrogate = None
    if args.surrogate != "none":
        try:
            if args.surrogate == "mock":
                surrogate = MockSurrogate(n_channels=30)
            else:
                cfg = InverseDesignConfig.from_yaml(args.config)
                import yaml

                paths = yaml.safe_load(Path(cfg.paths.paths_yaml).read_text(encoding="utf-8")) or {}
                root = Path(paths.get("forward_model_root", ""))
                ckpt = Path(paths.get("forward_checkpoint", ""))
                cfg_yaml = Path(paths.get("forward_config_yaml", ""))
                if args.surrogate in ("auto", "crrecon"):
                    if root.exists() and ckpt.exists() and cfg_yaml.exists():
                        import torch

                        surrogate = CRReconSurrogate(
                            forward_model_root=root,
                            checkpoint_path=ckpt,
                            config_yaml=cfg_yaml,
                            device=torch.device(args.device),
                        )
                    elif args.surrogate == "crrecon":
                        raise FileNotFoundError("CR_recon paths not configured in configs/paths.yaml")
        except Exception as e:
            print(f"[DASHBOARD] surrogate disabled: {e}")
            surrogate = None

    app = create_app(progress_dir=progress_dir, surrogate=surrogate)

    host = str(args.host)
    port = int(args.port)
    print(f"[DASHBOARD] http://{host}:{port}/")
    if host == "127.0.0.1":
        print(f"[DASHBOARD] (LAN) http://{_local_ip()}:{port}/")
    if surrogate is None:
        print("[DASHBOARD] spectrum plot: disabled (no surrogate)")
    else:
        print("[DASHBOARD] spectrum plot: enabled")

    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
