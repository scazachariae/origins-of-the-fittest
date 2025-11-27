# Origins of the Fittest

Python tooling for simulating and approximating clonal-interference dynamics on heterogeneous networks, with emphasis on identifying where the next high-fitness lineages originate.

## Overview

The package provides simulations and analytical approximations for structured populations. Empirically, the fittest strains arise at nodes that can spread mutations fastest before the next mutation arrives. As the mutation rate increases, the relevant network “scale” shifts from global centrality metrics to local descriptors such as node degree.

The package includes:

- **Gillespie and tau-leap simulations** of strain mutation and spread on networks optimised simulations for well-mixed systems.
- **Approximation framework** using Markov Renewal Processes to predict fitness distributions
- **Network construction utilities** for reproducible simulations

### Paper:

The corresponding research paper is available as a preprint at [biorxiv](https://doi.org/10.1101/2025.10.20.683208).

### Approximation framework

- **Markov Renewal Methods** – `approximation/tau_h_integrals.py` builds the transition matrix `H` (origin probabilities) and inter-event times `τ` for the fittest lineage.
- **Distance-based approximations** – `distance_message_passing.py` (epidemic-style message passing) covers the main approximation used in the paper.
- `distance_shortest_path.py` (geometric shortest-path surrogates) is a fast aternative.

### Networks, data, and utilities

- **Network construction** – `network_construction/construct_simple_networks.py` generates families of synthetic networks; `load_example_networks.py` loads deterministic adjacency matrices shipped under `origins_of_the_fittest/data/`.
- **Utilities** – `utility/io.py` handles bundled-data loading, `utility/phylogeny_helpers.py` adds post-processing helpers, and `utility/plot_altair.py` defines the Altair theme used inside notebooks.

## Installation

### pip

```bash
python -m venv .venv
source .venv/bin/activate           # Windows: .\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install .
```

### uv

```bash
uv sync                               # creates .venv/ and installs all deps
source .venv/bin/activate             # optional once the env exists
```

### conda

If you prefer Conda/Miniconda, create an environment and seed it with `requirements.txt` before installing the package:

```bash
conda create -n origins_of_the_fittest python=3.11
conda activate origins_of_the_fittest
python -m pip install -r requirements.txt
python -m pip install .
```

The `requirements.txt` file mirrors the dependencies listed in `pyproject.toml`.

## Quick Start

```python
from origins_of_the_fittest.network_construction.load_example_networks import load_ba32
from origins_of_the_fittest.simulation.full_system.one_layer_gillespie import simulation_phylogeny

# Load a Barabási–Albert network with 32 nodes
A = load_ba32()

# Run the reference simulation for 10 time units at mutation rate mu = 0.1
phy = simulation_phylogeny(T=10.0, A=A.values, mu=0.1, rng_seed=42)
print(phy.head())
```

## Notebooks and examples

Example usage can be found in [`notebooks/`](notebooks), with an overview in [`notebooks/README.md`](notebooks/README.md):

- `example_basic_simulation.ipynb` – Simple simulation example covering network loading, `simulation_phylogeny`, and Altair visualisation.
- `example_dmp_approximation_ba1024.ipynb` – Dynamic message passing (unweighted/weighted) approximation demo on a Barabási–Albert network.
- `example_shortest_path_approximation_ba1024.ipynb` – Shortest-path and τ/H integral workflow on the same network. (Much faster )

## Testing

After installation:

```bash
pytest -q                     # default (stochastic tests skipped)
pytest -q -m stochastic       # probabilistic checks (slow, could break if )
pyright                       # type checking
```

Reference data for the regression tests live in `origins_of_the_fittest/data/test/`. To (re)generate them deterministically:

```bash
python scripts/generate_reference_simulation.py
```

See `tests/README.md` for the full breakdown of deterministic vs. stochastic suites, category descriptions, and guidance on selecting subsets.



## Change log

Version `1.0.0` is the first public snapshot.

## License

Published under the MIT License (see [`LICENSE`](LICENSE)).

## Repository

GitHub mirror: <https://github.com/scazachariae/origins-of-the-fittest>
