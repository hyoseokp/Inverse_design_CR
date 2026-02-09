from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fdtd_scripts import extract_spectra_script, gds_import_script


def load_lumapi(*, lumerical_root: Path):
    """Load Lumerical's lumapi from the installation (Windows-friendly).

    Avoid importing a stray pip-installed lumapi that can't find interopapi.dll.
    """
    lumerical_root = Path(lumerical_root)
    api_py = lumerical_root / "api" / "python"
    lumapi_py = api_py / "lumapi.py"
    if not lumapi_py.exists():
        raise FileNotFoundError(f"lumapi.py not found at: {lumapi_py}")

    # Ensure DLL search path includes the directories containing interopapi.dll.
    dll_dirs = [
        api_py,
        lumerical_root / "bin",
        lumerical_root / "api" / "c",
    ]
    for d in dll_dirs:
        if d.exists():
            try:
                os.add_dll_directory(str(d))
            except Exception:
                # Best-effort; older Python/Windows may not support it.
                pass

    # Ensure python can import modules relative to lumapi.py.
    if str(api_py) not in sys.path:
        sys.path.insert(0, str(api_py))

    spec = importlib.util.spec_from_file_location("lumapi", str(lumapi_py))
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load spec for lumapi.py at {lumapi_py}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class LumapiBridge:
    """Concrete FDTDBridge using lumapi.FDTD session."""

    lumerical_root: Path
    template_fsp: Path
    hide: bool = True

    # Lumerical scripts are template-dependent; these are overridable hooks.
    layer_map: str = "1:0"

    def __post_init__(self) -> None:
        self.lumerical_root = Path(self.lumerical_root)
        self.template_fsp = Path(self.template_fsp)
        if not self.lumerical_root.exists():
            raise FileNotFoundError(f"lumerical_root not found: {self.lumerical_root}")
        if not self.template_fsp.exists():
            raise FileNotFoundError(f"template_fsp not found: {self.template_fsp}")
        self._lumapi = None
        self._fdtd = None

    def open(self) -> None:
        if self._fdtd is not None:
            return
        self._lumapi = load_lumapi(lumerical_root=self.lumerical_root)

        # Try common constructor patterns.
        fdtd = None
        try:
            fdtd = self._lumapi.FDTD(hide=bool(self.hide))
            try:
                fdtd.load(str(self.template_fsp))
            except Exception:
                # Some versions accept filename in constructor only.
                fdtd.close()
                fdtd = None
        except Exception:
            fdtd = None

        if fdtd is None:
            fdtd = self._lumapi.FDTD(filename=str(self.template_fsp), hide=bool(self.hide))

        self._fdtd = fdtd

    def close(self) -> None:
        if self._fdtd is None:
            return
        try:
            self._fdtd.close()
        finally:
            self._fdtd = None
            self._lumapi = None

    def import_gds(self, *, gds_path: Path, cell_name: str) -> None:
        if self._fdtd is None:
            raise RuntimeError("FDTD session not open")
        script = gds_import_script(gds_path=str(gds_path), cell_name=cell_name, layer_map=self.layer_map)
        self._fdtd.switchtolayout()
        self._fdtd.eval(script)

    def run(self) -> None:
        if self._fdtd is None:
            raise RuntimeError("FDTD session not open")
        self._fdtd.run()

    def extract_spectra(self) -> np.ndarray:
        """Extract template-dependent spectra as a numpy array.

        This default expects the template extraction script to define:
          - f_vec, T1, T2, T3
        and returns a stacked array: (4, N) where rows are [f_vec, T1, T2, T3].
        """
        if self._fdtd is None:
            raise RuntimeError("FDTD session not open")
        self._fdtd.eval(extract_spectra_script())
        f_vec = np.asarray(self._fdtd.getv("f_vec")).astype(np.float32).reshape(-1)
        t1 = np.asarray(self._fdtd.getv("T1")).astype(np.float32).reshape(-1)
        t2 = np.asarray(self._fdtd.getv("T2")).astype(np.float32).reshape(-1)
        t3 = np.asarray(self._fdtd.getv("T3")).astype(np.float32).reshape(-1)
        return np.stack([f_vec, t1, t2, t3], axis=0)

