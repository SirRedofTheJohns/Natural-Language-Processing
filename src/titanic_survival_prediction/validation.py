"""Structural validation helpers used locally and in lightweight CI."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import nbformat


def validate_notebook(
    path: str | Path, require_executed: bool = False
) -> dict[str, int]:
    """Validate notebook structure, code syntax, outputs, and execution status."""

    source = Path(path)
    notebook = nbformat.read(source, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    if len(code_cells) > 8:
        raise ValueError(f"{source} has more than eight executable cells.")
    for index, cell in enumerate(code_cells, 1):
        ast.parse(cell.source, filename=f"{source}:code-cell-{index}")
        errors = [
            output
            for output in cell.get("outputs", [])
            if output.get("output_type") == "error"
        ]
        if errors:
            raise ValueError(f"{source} contains an error output in code cell {index}.")
        if require_executed and cell.execution_count is None:
            raise ValueError(f"{source} code cell {index} was not executed.")
    return {"cells": len(notebook.cells), "code_cells": len(code_cells)}


def validate_all_notebooks(
    notebook_dir: str | Path = "notebooks",
    require_executed: bool = False,
) -> dict[str, dict[str, int]]:
    """Validate exactly the two approved project notebooks."""

    directory = Path(notebook_dir)
    expected = {
        "01_data_audit.ipynb",
        "02_model_experiments.ipynb",
    }
    actual = {path.name for path in directory.glob("*.ipynb")}
    if actual != expected:
        raise ValueError(
            f"Expected notebook files {sorted(expected)}, found {sorted(actual)}."
        )
    return {
        name: validate_notebook(directory / name, require_executed=require_executed)
        for name in sorted(expected)
    }


def main(argv: list[str] | None = None) -> int:
    """Run lightweight structural validation from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebooks",
        action="store_true",
        help="Validate the two project notebooks.",
    )
    parser.add_argument(
        "--require-executed",
        action="store_true",
        help="Require every code cell to have an execution count.",
    )
    args = parser.parse_args(argv)
    if not args.notebooks:
        parser.error("Select --notebooks.")
    results = validate_all_notebooks(require_executed=args.require_executed)
    for name, result in results.items():
        print(f"{name}: {result['cells']} cells, {result['code_cells']} code cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
