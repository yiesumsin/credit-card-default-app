"""Loads the Gradient Boosted Trees model that KNIME exported as a PMML file and scores clients with it.

PyPMML runs the model inside Java, so Java 8 or newer must be installed (on Streamlit Community Cloud this is
done by the packages.txt file in this folder; on your own computer install a Java runtime once).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from pypmml import Model

from credit_features import MODEL_COLUMNS, add_features


class PmmlModel:
    """Small wrapper so the rest of the app only has to call predict_default_probability()."""

    def __init__(self, pmml_path):
        self.path = Path(pmml_path)
        self.model = Model.load(str(self.path))

    def predict_default_probability(self, raw: pd.DataFrame) -> np.ndarray:
        """raw = table with the raw client columns. Returns P(default) for every row, as a float array."""
        features = add_features(raw)[MODEL_COLUMNS]
        scored = self.model.predict(features)
        # KNIME names the class-1 probability column "P (1)"; other exporters use "probability(1)" or similar.
        for name in ("P (1)", "probability(1)", "probability_1"):
            if name in scored.columns:
                return scored[name].to_numpy(dtype=float)
        raise KeyError(f"No default-probability column found in the PMML output. Columns were: {list(scored.columns)}")
