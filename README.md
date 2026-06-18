# FactorGP: A conditional-independence-aware approximate Gaussian process model for mapping

## Overview

This is research code for the approximate GP model **FactorGP**, submitted to the RSS From Perception to Action workshop. Our model learns a GP approximating GMRF prior that can encode spatially varying lengthscales and conditional independences, capturing non-stationarity and discontinuities in data respectively.

The model code is contained within `fgp/`:
- `fgp/graph_learning.py`: SVGD-based graph learning module
- `fgp/gbp.py` : factor graph representation of the GMRF prior

## Requirements

- Python ≥ 3.9
- Other dependencies:

```
pip install -r requirements.txt
```

## Reproducing results

### Terrain mapping (SRTM)

```
python active_mapping_srtm.py --env <environment_name> [--samples N]
```

Valid environments: `N17E073`, `N43W080`, `N45W123`, `N47W124`

Example:
```
python active_mapping_srtm.py --env N17E073
python active_mapping_srtm.py --env N43W080 --samples 700
```

Results are saved as `results_<env>_seed<N>.png` in the working directory and the eval table is printed to terminal.

### Gas diffusion mapping

```
python active_sampling_gas.py
```

Per-step figures and final metrics are saved under `figs/active_sampling/` and timestamped run data under `results/active_sampling/`. Eval table is printed to terminal.

## Gas diffusion simulation

Code to produce gas fields via a forward-Euler Fick's law solver is in `data/gas/diffusion_test.py`. You can define your own building floor plan and source/sink locations to produce new simulated gas fields.
