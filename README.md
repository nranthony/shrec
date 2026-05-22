## shrec

The **sh**ared **rec**urrence method for identifying causal drivers of collections of time series 

![A diagram of the algorithm](resources/fig_github.png)

## Usage

```python
from shrec.models import RecurrenceManifold
from benchmarks.dynamical_systems import load_data

X, y_true = load_data() # shape (n_timepoints, n_series)
model = RecurrenceManifold()
y_recon = model.fit_predict(X)
```

Additional examples can be found in the [`demos.ipynb`](demos.ipynb) notebook.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage the virtual environment and dependencies.

Install `uv` if you don't already have it (see the [uv install docs](https://docs.astral.sh/uv/getting-started/installation/)), then:

	git clone https://github.com/williamgilpin/shrec
	cd shrec
	uv sync

This creates a `.venv/` in the project root with all dependencies from `pyproject.toml` installed. Activate it with:

	source .venv/bin/activate

Or run commands directly through uv without activating:

	uv run python -m unittest
	uv run jupyter lab

### Removing the old conda/mamba environment

If you previously installed this project with the now-removed `environment.yml`, drop the old env:

	conda env remove -n shrec    # or: mamba env remove -n shrec
	conda clean -a               # optional: free up disk space

Dependencies
+ scikit-learn
+ scipy
+ numpy
+ umap-learn
<!-- + python-igraph
+ leidenalg -->

Additional dependencies only necessary for discrete drivers
+ networkx (necessary for learning discrete drivers only)
+ graspologic (necessary for learning discrete drivers only)

Additional dependencies for certain demonstrations
+ matplotlib 
+ seaborn
+ pandas
+ h5py

## Development and Contributing

We welcome any suggestions or improvments. Please feel free to reach reach out, raise issues, or submit pull requests.


