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
	#Generate 25 numeric features with 5000 rows each; Relationships include level 1 - level 4 
	.gen_multi(n=5000, num_cols=25, cat_cols=0, n_layers=4)
	
	#Add features with a specific function
		.add_var("col26", lambda x: x**2 - x + add_noise(len(x), sd=5), ["x1"])
		.add_var("col27", lambda a1, a2: a1 + a2, {"a1": "x1", "a2": "x3"})
	
    #Add a feature, with random nominal values
		.add_nominal()
	
	#Add a feature, with random ordinal values
		.add_ordinal()
	
	#Add a feature, with random interval values
		.add_interval()
	
	#Add a feature, with random ratio values
		.add_ratio()
	
    #Add a biased and unbiased target which depends on randomly picked features
	#The target is dependent on 20% of all features
	#Target is binarized
		.add_target(dependency_rate = 0.2, target_type = "binary")
	
	#Return the dataframe with the features specified above
		.create_df()
)

```

## Features
- Easily generate synthetic datasets with various data types and relationships.
- Define variables with custom functions and dependencies.
- Supports nominal, ordinal, interval, and ratio data types.
- Generate a target variable with random values.
