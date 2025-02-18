# AnomalyLab

## Authors

Chen Haiwei, Deng Haotian

## Overview

This Python package implements various empirical methods from the book *Empirical Asset Pricing: The Cross Section of Stock Returns* by Turan G. Bali, Robert F. Engle, and Scott Murray. The package includes functionality for:

- Summary statistics
- Correlation analysis
- Persistence analysis
- Portfolio analysis
- Fama-MacBeth regression (FM regression)

Additionally, we have added several extra features, such as:

- Missing value imputation
- Data normalization
- Leading and lagging variables
- Winsorization/truncation
- Transition matrix calculation
- Formatting output tables

## Installation

The package can be installed via:

```bash
pip install anomalylab
```

## Usage

This package provides a comprehensive suite of tools for empirical asset pricing analysis. Below are key functions with explanations and example usage to help you get started.

### Importing Data

```python
from importlib import resources

import pandas as pd
from pandas import DataFrame

from anomalylab import Panel, TimeSeries, pp
from anomalylab.datasets import DataSet

df: DataFrame = DataSet.get_panel_data()
ts: DataFrame = DataSet.get_time_series_data()

# Specifying Factor Models:
Models: dict[str, list[str]] = {
    "CAPM": ["MKT(3F)"],  # Capital Asset Pricing Model with Market Factor
    "FF3": ["MKT(3F)", "SMB(3F)", "HML(3F)"],  # Fama-French 3 Factor Model
    "FF5": ["MKT(5F)", "SMB(5F)", "HML(5F)", "RMW(5F)", "CMA(5F)"],  # Fama-French 5 Factor Model
}

# Creating Panel and Time Series Objects:
panel = Panel(
    df,
    name="Stocks",
    id="permno",
    time="date",
    frequency="M",
    ret="return",
    classifications="industry",
    drop_all_chars_missing=True,
    is_copy=False,
)
time_series: TimeSeries = TimeSeries(
    df=ts, name="Factor Series", time="date", frequency="M", is_copy=False
)
pp(panel)
```

### Preprocessing Data

Several preprocessing functions are available for handling missing values, normalizing data, shifting variables, and winsorizing data.

```python
# Filling Data:
# Filling Group Columns
panel.fill_group_column(group_column="industry", value="Other")
# Filling Missing Values
panel.fillna(method="mean", group_columns="date")

# Normalizing Data:
# panel.normalize(method="zscore", group_columns="date")

# Shifting Data:
# panel.shift(periods=1, drop_original=False)

# Winsorizing Data:
panel.winsorize(method="winsorize")
pp(panel)
```

### Summary statistics

You can compute summary statistics for your dataset using the summary() function:

```python
summary = panel.summary()
pp(summary)
```

### Correlation analysis

The correlation() function computes the correlations between different variables in the panel data:

```python
correlation = panel.correlation()
pp(correlation)
```

### Persistence analysis

Persistence analysis helps you understand the stability of certain variables over time.
The persistence() function computes persistence for a given set of periods to analyze the stability of a variable.
The transition_matrix() function calculates the transition matrix to evaluate how a variable moves between different states (e.g., deciles) over time.

```python
person = panel.persistence(periods=[1, 3, 6, 12, 36, 60])
pp(persistence)
pp(
    panel.transition_matrix(
        var="MktCap",
        group=10,
        lag=12,
        draw=False,
        # path="...",
        decimal=2,
    )
)
```

### Portfolio analysis

You can group data, and perform univariate and bivariate portfolio analyses based on factors.

```python
# Grouping
group_result = panel.group("return", "MktCap", "Illiq", 10)

# Univariate portfolio analysis
uni_ew, uni_vw = panel.univariate_analysis(
    "return", "MktCap", "Illiq", 10, Models, time_series, factor_return=False
)
pp(uni_ew)
pp(uni_vw)

# Bivariate portfolio analysis
bi_ew, bi_vw = panel.bivariate_analysis(
    "return",
    "MktCap",
    "Illiq",
    "IdioVol",
    5,
    5,
    Models,
    time_series,
    True,
    False,
    "dependent",
    factor_return=False,
)
pp(bi_ew)
pp(bi_vw)
```

### Fama-MacBeth regression

You can run Fama-MacBeth regressions with multiple independent variables:

```python
fm_result = panel.fm_reg(
    regs=[
        ["return", "MktCap"],
        ["return", "Illiq"],
        ["return", "IdioVol"],
        ["return", "MktCap", "Illiq", "IdioVol"],
    ],
    exog_order=["MktCap", "Illiq", "IdioVol"],
    weight="MktCap",
    industry="industry",
    industry_weighed_method="value",
    is_winsorize=False,
    is_normalize=True,
)
pp(fm_result)
```

### Formatting results

Finally, you can save and format the results to an Excel file:

```python
output_file_path = "..."
with pd.ExcelWriter(output_file_path) as writer:
    summary.to_excel(writer, sheet_name="summary")
    correlation.to_excel(writer, sheet_name="correlation")
    persistence.to_excel(writer, sheet_name="persistence")
    uni_ew.to_excel(writer, sheet_name="uni_ew")
    uni_vw.to_excel(writer, sheet_name="uni_vw")
    bi_ew.to_excel(writer, sheet_name="bi_ew")
    bi_vw.to_excel(writer, sheet_name="bi_vw")
    fm_result.to_excel(writer, sheet_name="fm_result")

panel.format_excel(
    output_file_path,
    align=True,
    line=True,
    convert_brackets=False,
    adjust_col_widths=True,
)
```
