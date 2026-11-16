"""Interaction regression checks. Run with: python -m unittest test_app -v."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


class AssessmentFlowTest(unittest.TestCase):
    def test_submission_examples_cutoff_and_batch(self):
        app = AppTest.from_file(str(Path(__file__).with_name("app.py")), default_timeout=60).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.number_input), 14)
        self.assertEqual(len(app.selectbox), 10)
        self.assertEqual(app.slider(key="cutoff").value, 0.30)
        self.assertEqual(app.slider(key="batch_cutoff").value, 0.30)
        self.assertEqual(app.selectbox(key="MARRIAGE").value, 2)
        self.assertEqual(len(app.metric), 0)

        def submit():
            next(button for button in app.button if button.label == "Calculate risk").click().run()
            self.assertFalse(app.exception)

        app.selectbox(key="MARRIAGE").set_value(1)
        submit()
        self.assertEqual(app.session_state["assessment"]["inputs"]["MARRIAGE"], 1)
        self.assertEqual(app.metric[2].value, "30%")
        baseline = app.session_state["assessment"]["probability"]
        self.assertTrue(0 <= baseline <= 1)
        app.number_input(key="LIMIT_BAL").set_value(123456)
        app.number_input(key="BILL_AMT1").set_value(-1234)
        app.number_input(key="PAY_AMT1").set_value(123)
        submit()
        snapshot = app.session_state["assessment"]
        self.assertEqual(snapshot["inputs"]["LIMIT_BAL"], 123456)
        self.assertEqual(snapshot["inputs"]["BILL_AMT1"], -1234)
        self.assertEqual(snapshot["inputs"]["PAY_AMT1"], 123)
        app.slider(key="cutoff").set_value(0.4).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state["assessment"], snapshot)
        self.assertEqual(app.metric[2].value, "40%")
        self.assertEqual(app.slider(key="batch_cutoff").value, 0.4)
        app.slider(key="batch_cutoff").set_value(0.3).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.slider(key="cutoff").value, 0.3)
        self.assertEqual(app.metric[2].value, "30%")
        self.assertEqual(app.session_state["assessment"], snapshot)

        app.button(key="example_High risk").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.metric), 0)
        self.assertEqual(app.number_input(key="LIMIT_BAL").value, 60000)
        submit()
        self.assertGreater(app.session_state["assessment"]["probability"], baseline)

        self.assertEqual(len(app.radio), 0)
        self.assertEqual(len(app.get("file_uploader")), 1)


class MarriageEncodingTest(unittest.TestCase):
    def test_encoding_matches_pmml_schema(self):
        import xml.etree.ElementTree as ET
        import pandas as pd
        from credit_features import add_features, MODEL_COLUMNS
        raw = pd.read_csv(Path(__file__).with_name("sample_data") / "synthetic_clients_1000.csv").iloc[:3].copy()
        raw["MARRIAGE"] = [1, 2, 3]
        features = add_features(raw)
        columns = ["MARRIAGE_married", "MARRIAGE_single", "MARRIAGE_others"]
        self.assertEqual(features[columns].values.tolist(), [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        root = ET.parse(Path(__file__).with_name("credit_card_default_classifier.pmml")).getroot()
        ns = {"p": "http://www.dmg.org/PMML-4_2"}
        schema = root.find("p:MiningModel/p:MiningSchema", ns)
        active = {field.attrib["name"] for field in schema if field.attrib.get("usageType", "active") == "active"}
        self.assertEqual(set(MODEL_COLUMNS), active)
        raw.loc[raw.index[0], "MARRIAGE"] = 0
        with self.assertRaises(ValueError):
            add_features(raw)


if __name__ == "__main__":
    unittest.main()
