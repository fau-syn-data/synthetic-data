# synthedata

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

Synthedata is a powerful tool for creating synthetic datasets with various data types and relationships. It provides a convenient interface for generating synthetic data, making it useful for testing and experimenting with data-driven applications.

## Installation

You can install synthedata using pip:
```bash
pip install synthedata
```
## Example Usage

Below is an example of how to use synthedata to create a synthetic dataset:

```python
from synthedata import data_creator
from synthedata.data_creator import add_noise

# Initialize the DataCreator object
data_creator = datacreator.DataCreator()


# Add variables with specified functions and dependencies
df = (
    data_creator
    .add_var("x1", lambda: list(range(1, 9)), {})
    .add_var("x1", lambda: list(range(1, 9)), {})
    .add_var("x1", lambda: list(range(1, 9)), {})
    .add_var("x1", lambda: list(range(1, 9)), {})
    .add_var("x2", lambda x: x**2 - x + add_noise(len(x), sd=5), ["x1"])
    .add_var("x3", lambda: 2)
    .add_var("x4", lambda a1, a2: a1 + a2, {"a1": "x1", "a2": "x3"})
    .add_var("x5", lambda x: [
        "odd" if _ in [1, 3, 5]
        else "even" if _ in [2, 4, 6]
        else "unknown"
        for _ in x
    ], {"x": "x1"})
    .add_var("x6", lambda x: np.random.choice(['male', 'female'], len(x), p=[0.5, 0.5]), ["x1"])
    .add_nominal()
    .add_ordinal()
    .add_interval()
    .add_ratio()
    .add_ratio()
    .add_ratio()
    .add_target("target", "random")
    .create_df()
)

```

## Features
- Easily generate synthetic datasets with various data types and relationships.
- Define variables with custom functions and dependencies.
- Supports nominal, ordinal, interval, and ratio data types.
- Generate a target variable with random values.
