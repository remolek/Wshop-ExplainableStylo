# CLARIN-PL's explainable stylometric pipeline
 
This package has been developed as a modular pipeline for stylometric studies.

Currently it uses [spaCy](https://spacy.io) for text preprocessing and feature extraction, [Light Gradient-Boosting Machine (LGBM)](https://github.com/microsoft/LightGBM) as the state-of-the-art boosted tree classifier, [Shapley Additive Explanations (SHAP)](https://github.com/shap/shap) for computing explanations of the classifier's decisions, [scikit-learn](https://scikit-learn.org/stable/index.html) for feature counting and cross-validation.

## Installation

Right now, the package is not distributed in any package manager.
It can be downloaded and imported locally in Python.

At the moment one needs to install:
1. [spaCy](https://spacy.io/usage) 

2. an appropriate spaCy language model, e.g., the large English one:
```python
python -m spacy download en_core_web_lg
```

3. `lightgbm, shap, sklearn` available from pip

4. other standard libraries distributed, e.g., via pip:
`re, json, pickle, tqdm, matplotlib, pandas, numpy`.

## Usage

Presently, manual control over the whole pipeline is recommended. [This notebook](notebooks/test_base.ipynb) shows how to load texts, preprocess them, extract features, classify texts, and visualise the classifier explanations.
Loading 
```python
from cl_explainable_stylo import base

exp = base.explain_style('init_metadata.json',manual = True)
```
should get you started.
