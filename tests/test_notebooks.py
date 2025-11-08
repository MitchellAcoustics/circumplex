"""
Tests to ensure Jupyter notebooks in the documentation have valid structure.

This test only validates the structure of the notebooks, but does not execute them.
"""

import os
import pytest
import nbformat

# Find all notebooks in the tutorials directory
TUTORIALS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "docs", "tutorials"
)
NOTEBOOKS = [
    os.path.join(TUTORIALS_DIR, nb)
    for nb in os.listdir(TUTORIALS_DIR)
    if nb.endswith(".ipynb") and not nb.startswith(".")
]


@pytest.mark.parametrize("notebook_path", NOTEBOOKS)
def test_notebook_structure(notebook_path):
    """
    Test that the notebook has valid structure and can be parsed.

    This test does NOT execute the notebook cells, it only checks
    that the notebook can be loaded as a valid nbformat document.

    Args:
        notebook_path (str): Path to the notebook file
    """
    try:
        # Load the notebook
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = nbformat.read(f, as_version=4)

        # Check that it has cells
        assert len(notebook.cells) > 0, "Notebook has no cells"

        # Verify cells have required attributes
        for i, cell in enumerate(notebook.cells):
            assert "source" in cell, f"Cell {i} is missing source"
            assert "cell_type" in cell, f"Cell {i} is missing cell_type"

        print(f"Successfully validated notebook: {os.path.basename(notebook_path)}")
    except Exception as e:
        pytest.fail(
            f"Failed to parse notebook {os.path.basename(notebook_path)}: {str(e)}"
        )
