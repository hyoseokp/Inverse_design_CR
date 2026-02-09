from __future__ import annotations

import io
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from .spectral import merge_rggb_to_rgb

_TOPK_RE = re.compile(r"^topk_step-(\d+)\.npz$")


def _json_sanitize(obj: Any) -> Any:
    """Make payload safe for JSON.parse (replace NaN/Inf)."""
    if obj is None:
        return None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


def _tail_jsonl(path: Path, n: int) -> list[dict[str, Any]]:
    if n <= 0 or not path.exists():
        return []
    dq: deque[str] = deque(maxlen=n)
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                dq.append(s)
    out: list[dict[str, Any]] = []
    for s in dq:
        try:
            o = json.loads(s)
            if isinstance(o, dict):
                out.append(o)
        except Exception:
            continue
    return out


def _read_meta(progress_dir: Path) -> dict[str, Any]:
    p = progress_dir / "run_meta.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _ts_start_epoch(meta: dict[str, Any]) -> float | None:
    ts = meta.get("ts_start")
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


@dataclass
class TopKCache:
    step: int | None = None
    npz: dict[str, np.ndarray] | None = None


@dataclass
class SpectrumCache:
    key: tuple[int, int] | None = None
    rgb: np.ndarray | None = None


def create_app(*, progress_dir: Path, surrogate=None) -> FastAPI:
    progress_dir = Path(progress_dir)
    cache = TopKCache()
    scache = SpectrumCache()

    app = FastAPI(title="CR Inverse Dashboard")

    static_dir = Path(__file__).parent / "dashboard_static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = static_dir / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>Dashboard UI missing</h1>", status_code=500)

    @app.get("/api/ping")
    def ping() -> JSONResponse:
        routes = sorted({getattr(r, "path", "") for r in app.router.routes if getattr(r, "path", "")})
        return JSONResponse({"ok": True, "routes": routes})

    @app.get("/api/meta")
    def meta() -> JSONResponse:
        return JSONResponse(_json_sanitize(_read_meta(progress_dir)))

    @app.get("/api/metrics")
    def metrics(tail: int = Query(default=2000, ge=1, le=20000)) -> JSONResponse:
        items = _tail_jsonl(progress_dir / "metrics.jsonl", int(tail))
        return JSONResponse({"items": _json_sanitize(items)})

    @app.get("/api/ls")
    def ls() -> JSONResponse:
        items = []
        if progress_dir.exists():
            for p in sorted(progress_dir.iterdir(), key=lambda x: x.name):
                try:
                    st = p.stat()
                    items.append({"name": p.name, "bytes": int(st.st_size), "is_dir": p.is_dir()})
                except Exception:
                    items.append({"name": p.name, "bytes": None, "is_dir": p.is_dir()})
        return JSONResponse({"progress_dir": str(progress_dir), "exists": progress_dir.exists(), "items": items})

    def _latest_topk_step() -> int | None:
        if not progress_dir.exists():
            return None
        meta = _read_meta(progress_dir)
        nsteps = meta.get("n_steps")
        max_step = None
        try:
            nsteps = int(nsteps) if nsteps is not None else None
            if nsteps is not None and nsteps > 0:
                max_step = nsteps - 1
        except Exception:
            max_step = None

        ts0 = _ts_start_epoch(meta)
        best = None
        for p in progress_dir.iterdir():
            m = _TOPK_RE.match(p.name)
            if not m:
                continue
            step = int(m.group(1))
            if max_step is not None and step > max_step:
                continue
            if ts0 is not None:
                try:
                    if p.stat().st_mtime < ts0:
                        continue
                except Exception:
                    pass
            if best is None or step > best:
                best = step
        return best

    def _load_topk(step: int) -> dict[str, np.ndarray]:
        if cache.step == step and cache.npz is not None:
            return cache.npz
        p = progress_dir / f"topk_step-{int(step)}.npz"
        z = np.load(p, allow_pickle=False)
        data = {k: z[k] for k in z.files}
        cache.step = step
        cache.npz = data
        return data

    @app.get("/api/topk/latest")
    def topk_latest() -> JSONResponse:
        step = _latest_topk_step()
        if step is None:
            return JSONResponse({"step": None, "k": 0, "metrics": {}})
        try:
            data = _load_topk(step)
        except Exception as e:
            return JSONResponse({"step": int(step), "k": 0, "error": f"failed to load npz: {e}"}, status_code=500)
        struct = data.get("struct128_topk")
        k = int(struct.shape[0]) if isinstance(struct, np.ndarray) and struct.ndim == 3 else 0
        metrics: dict[str, Any] = {}
        for key, arr in data.items():
            if key.startswith("metric_"):
                metrics[key] = arr.tolist()
            if key == "metric_best_loss":
                metrics[key] = arr.tolist()
        fill = []
        if isinstance(struct, np.ndarray) and struct.ndim == 3:
            fill = [float(struct[i].mean()) for i in range(struct.shape[0])]
        return JSONResponse(
            _json_sanitize(
                {
                    "step": int(step),
                    "k": k,
                    "images": [f"/api/topk/{int(step)}/{i}.png" for i in range(k)],
                    "metrics": metrics,
                    "fill_frac": fill,
                }
            )
        )

    @app.get("/api/topk/{step}/{idx}.png")
    def topk_png(step: int, idx: int, invert: int = Query(default=1, ge=0, le=1)) -> Response:
        try:
            data = _load_topk(int(step))
        except Exception:
            return Response(status_code=500)
        struct = data["struct128_topk"]
        if idx < 0 or idx >= struct.shape[0]:
            return Response(status_code=404)
        s = struct[idx].astype(np.uint8)
        if int(invert) == 1:
            img = (1 - np.clip(s, 0, 1)) * 255
        else:
            img = np.clip(s, 0, 1) * 255
        im = Image.fromarray(img, mode="L")
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "no-store, max-age=0, must-revalidate"},
        )

    @app.get("/api/topk/{step}/{idx}/spectrum")
    def topk_spectrum(step: int, idx: int) -> JSONResponse:
        if surrogate is None:
            return JSONResponse({"error": "surrogate not configured"}, status_code=400)
        key = (int(step), int(idx))
        if scache.key == key and scache.rgb is not None:
            rgb = scache.rgb
        else:
            data = _load_topk(int(step))
            struct = data["struct128_topk"]
            if idx < 0 or idx >= struct.shape[0]:
                return JSONResponse({"error": "idx out of range"}, status_code=404)
            x = torch.from_numpy(struct[idx].astype(np.float32))[None, ...]  # (1,128,128)
            with torch.no_grad():
                y = surrogate.predict(x)
                rgb_t = merge_rggb_to_rgb(y)[0]
            rgb = rgb_t.detach().cpu().numpy().astype(np.float32)
            scache.key = key
            scache.rgb = rgb
        C = int(rgb.shape[-1])
        return JSONResponse(_json_sanitize({"step": int(step), "idx": int(idx), "n_channels": C, "rgb": rgb.tolist()}))

    @app.get("/api/status")
    def status(window: str = Query(default="all")) -> JSONResponse:
        """Convenience endpoint for the Chart.js UI.

        window: all|50|200|1000
        """
        meta = _read_meta(progress_dir)
        nsteps = int(meta.get("n_steps", 0) or 0)
        ts0 = _ts_start_epoch(meta)
        if window == "all":
            tail = 20000
        else:
            try:
                tail = int(window)
            except Exception:
                tail = 200
        tail = max(1, min(20000, tail))
        items = _tail_jsonl(progress_dir / "metrics.jsonl", tail)
        # Filter to this run only.
        if ts0 is not None:
            filt = []
            for it in items:
                try:
                    t = datetime.fromisoformat(str(it.get("ts", "")).replace("Z", "+00:00")).timestamp()
                    if t < ts0:
                        continue
                except Exception:
                    continue
                filt.append(it)
            items = filt
        # Clamp steps.
        if nsteps > 0:
            tmp = []
            for it in items:
                try:
                    s = int(it.get("step"))
                except Exception:
                    continue
                if s <= nsteps - 1:
                    tmp.append(it)
            items = tmp

        # Build series (use last value per step).
        by_step: dict[int, dict[str, Any]] = {}
        for it in items:
            try:
                s = int(it.get("step"))
            except Exception:
                continue
            by_step[s] = it
        steps_sorted = sorted(by_step.keys())
        def _f(v) -> float:
            try:
                x = float(v)
                return x if math.isfinite(x) else float("nan")
            except Exception:
                return float("nan")

        loss_total = [_f(by_step[s].get("loss_total")) for s in steps_sorted]
        loss_spec = [_f(by_step[s].get("loss_spec")) for s in steps_sorted]
        loss_reg = [_f(by_step[s].get("loss_reg")) for s in steps_sorted]
        loss_purity = [_f(by_step[s].get("loss_purity")) for s in steps_sorted]
        latest = by_step[steps_sorted[-1]] if steps_sorted else {}

        return JSONResponse(
            _json_sanitize(
                {
                    "meta": meta,
                    "series": {
                        "steps": steps_sorted,
                        "loss_total": loss_total,
                        "loss_spec": loss_spec,
                        "loss_reg": loss_reg,
                        "loss_purity": loss_purity,
                    },
                    "latest": latest,
                }
            )
        )

    return app
