"""
Archive a pipeline run and render the analysis notebook.

Copies pipeline output to exp/{name}/, executes the analysis
notebook against it, exports to HTML, and stores everything
together.

Run with:
    uv run python scripts/archive.py ./pipeline_output my_run
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_SRC = REPO_ROOT / "notebooks" / "diagnostic_plots.ipynb"


def patch_notebook(nb, exp_name):
    """Patch notebook for headless execution against exp/{name}."""
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])

        # Swap widget backend for inline (headless-safe)
        if "%matplotlib widget" in source:
            source = source.replace(
                "%matplotlib widget",
                "%matplotlib inline",
            )
            cell["source"] = source.splitlines(keepends=True)

        # Patch output_dir in the config cell
        if "USER-EDITABLE PARAMETERS" in source:
            source = re.sub(
                r'output_dir\s*=\s*Path\(["\'].*?["\']\)',
                f'output_dir = Path("../exp/{exp_name}")',
                source,
            )
            cell["source"] = source.splitlines(keepends=True)

    # Clear all outputs for a clean run
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None


def main():
    parser = argparse.ArgumentParser(
        description="Archive pipeline output and render analysis notebook.",
    )
    parser.add_argument(
        "pipeline_output",
        help="Path to pipeline output directory",
    )
    parser.add_argument(
        "name",
        help="Experiment name (e.g. 2525)",
    )
    args = parser.parse_args()

    src = Path(args.pipeline_output)
    if not src.is_dir():
        print(
            f"Error: {src} is not a directory",
            file=sys.stderr,
        )
        sys.exit(1)

    exp_dir = REPO_ROOT / "exp" / args.name
    if exp_dir.exists():
        print(
            f"Error: {exp_dir} already exists",
            file=sys.stderr,
        )
        sys.exit(1)

    # 1. Copy pipeline output to exp/{name}/
    print(f"Copying {src} -> {exp_dir}")
    shutil.copytree(src, exp_dir)

    # 2. Prepare notebook: patch config, write temp copy
    #    in notebooks/ so data_path stays valid
    with open(NOTEBOOK_SRC) as f:
        nb = json.load(f)

    patch_notebook(nb, args.name)

    tmp_nb = REPO_ROOT / "notebooks" / f"_analyze_{args.name}.ipynb"
    with open(tmp_nb, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")

    tmp_html = tmp_nb.with_suffix(".html")

    try:
        # 3. Execute notebook
        print("Executing notebook...")
        subprocess.run(
            [
                "uv",
                "run",
                "jupyter",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=600",
                str(tmp_nb),
            ],
            check=True,
        )

        # 4. Export to HTML
        print("Exporting to HTML...")
        subprocess.run(
            [
                "uv",
                "run",
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                str(tmp_nb),
            ],
            check=True,
        )

        # 5. Move rendered files to exp/{name}/
        shutil.move(
            str(tmp_nb),
            str(exp_dir / "diagnostic_plots.ipynb"),
        )
        shutil.move(
            str(tmp_html),
            str(exp_dir / "diagnostic_plots.html"),
        )

    except Exception:
        tmp_nb.unlink(missing_ok=True)
        tmp_html.unlink(missing_ok=True)
        raise

    print(f"\nDone! Results archived in {exp_dir}")


if __name__ == "__main__":
    main()
