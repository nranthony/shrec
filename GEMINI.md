# shrec

## Project Overview
**shrec** (shared recurrence) is a Python library designed to identify causal drivers in collections of time series data using recurrence manifolds. It provides tools for both continuous and discrete driver identification, leveraging techniques from dynamical systems and graph theory.

## Key Technologies
*   **Language:** Python 3.8+
*   **Core Libraries:** `numpy`, `scipy`, `scikit-learn`
*   **Graph/Network Analysis:** `networkx`, `graspologic`, `leidenalg`, `cdlib`
*   **Manifold Learning:** `umap-learn`, `isomap` (via scikit-learn)
*   **Build System:** `setuptools`

## Directory Structure
*   `src/shrec/`: Main source code for the library.
    *   `models/`: Contains the implementation of recurrence models (`RecurrenceManifold`, `RecurrenceClustering`, etc.).
    *   `utils/`: Utility functions for time series processing and graph operations.
*   `benchmarks/`: Scripts and notebooks for benchmarking and evaluating model performance.
*   `tests/`: Unit tests for the library.
*   `docs/`: Documentation files (Sphinx).
*   `resources/`: Images and other static assets.

## Building and Running

### Installation
To install the package in editable mode (recommended for development):
```bash
pip install -e .
```

### Running Tests
The project uses `unittest` for testing. To run all tests:
```bash
python -m unittest
```
Or to run a specific test file:
```bash
python -m unittest tests/test_models.py
```

### Usage Example
```python
from shrec.models import RecurrenceManifold
from benchmarks.dynamical_systems import load_data

# Load sample data (replace with your data loading logic)
X, y_true = load_data() 

# Initialize and fit the model
model = RecurrenceManifold()
y_recon = model.fit_predict(X)
```

## Development Conventions
*   **Code Style:** Follows standard Python conventions (PEP 8).
*   **Testing:** New features or bug fixes should be accompanied by unit tests in the `tests/` directory.
*   **Dependencies:** Core dependencies are listed in `pyproject.toml`.
