"""Command-line entry point for the local browser viewer."""

import argparse
import importlib.util
import os
from pathlib import Path
import sys


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Open the local pysegy viewer")
    parser.add_argument("path", nargs="?", help="SEG-Y file or directory to open")
    parser.add_argument("--port", type=int, default=8501, help="Local server port")
    return parser


def main() -> None:
    """Start Streamlit on localhost, optionally pre-filling a dataset path."""

    args = _parser().parse_args()
    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit(
            'The viewer dependencies are not installed. Run: pip install "pysegy[viewer]"'
        )
    from streamlit.web import cli as streamlit_cli

    if args.path:
        os.environ["PYSEGY_VIEWER_PATH"] = str(Path(args.path).expanduser().resolve())
    app = Path(__file__).with_name("app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app),
        "--server.address=127.0.0.1",
        f"--server.port={args.port}",
        "--browser.gatherUsageStats=false",
    ]
    streamlit_cli.main()


if __name__ == "__main__":
    main()
