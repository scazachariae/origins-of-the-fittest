# Test Suite Organization

This document describes the reorganized test structure for the `origins_of_the_fittest` codebase.

## Test Categories

### 1. **Reference Tests** (`reference/`)

- **Purpose**: Compare current outputs against pre-run simulation results
- **Characteristics**: Deterministic, fast, part of standard test suite
- **Examples**:
  - `test_simulation_reference.py` - Gillespie, tau-leap, well-mixed reference comparisons

### 2. **Stochastic Consistency Tests** (`consistency/`)

- **Purpose**: Compare different implementations that compute the same quantity
- **Characteristics**: Use statistical tests (KS tests), marked as stochastic
- **Marker**: `@pytest.mark.stochastic`
- **Examples**:
  - `test_tauleap_gillespie_equivalence.py` - Gillespie vs tau-leap simulation statistics

### 3. **Unit Tests** (`unit/`)

- **Purpose**: Basic correctness and sanity checks
- **Characteristics**: Fast, deterministic, check logical conditions
- **Examples**:
  - `test_network_construction.py` -  reproducible with seeding, networks are fully connected, etc. 

### 4. **Numerical Consistency Tests** (`numerical/`)

- **Purpose**: Validate against analytical solutions and manual calculations
- **Characteristics**: Test against toy examples with known solutions
- **Examples**:
  - `tests/numerical/test_rate_calculations_analytical.py` - Tests rate caculations in very simple network configurations against manually calculated rates

## Running Tests

```bash
# Run all tests (excluding stochastic by default)
pytest -q

# Run only stochastic tests
pytest -q -m stochastic

# Run specific category
pytest -q tests/unit/
pytest -q tests/numerical/
pytest -q tests/reference/

# Quick run (unit tests only)
pytest -q tests/unit/ -m "not stochastic"
```

## Test Infrastructure

### Shared Fixtures (`conftest.py`)

- Registers the `stochastic` marker for pytest
- Individual tests set up their own fixtures locally or via helper modules

### Test Data Management

- Reference data in `origins_of_the_fittest/data/test/`
- Regenerate via `python scripts/generate_reference_simulation.py
