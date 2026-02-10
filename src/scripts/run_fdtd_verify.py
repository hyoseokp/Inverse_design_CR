from __future__ import annotations

from crinv.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["fdtd-verify", *(__import__("sys").argv[1:])]))

