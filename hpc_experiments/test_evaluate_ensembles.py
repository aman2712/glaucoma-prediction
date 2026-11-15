import unittest
import numpy as np
import pandas as pd
from evaluate_ensembles import ensemble,bootstrap


class Checks(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame(dict(image_id=['a','b','c','d'],label=[0,0,1,1],
                                 analysis_group=['a','b','c','d'],probability=[.1,.3,.6,.9]))

    def test_identical_ensemble_is_identity(self):
        p=self.fixture()
        result=ensemble([p,p.iloc[::-1],p])
        np.testing.assert_allclose(result.probability,p.probability,atol=1e-12)

    def test_mismatched_labels_rejected(self):
        p=self.fixture();other=p.copy();other.loc[0,'label']=1
        with self.assertRaises(ValueError):ensemble([p,other])

    def test_paired_bootstrap_identical_scores_has_zero_difference(self):
        p=self.fixture()
        ci,difference=bootstrap({'E0':p,'E2':p},{'E0':.5,'E2':.5},9,30)
        self.assertEqual(len(ci),12)
        self.assertEqual(float(difference.iloc[0].estimate),0.)
        self.assertEqual(float(difference.iloc[0].lower_95),0.)
        self.assertEqual(float(difference.iloc[0].upper_95),0.)


if __name__=='__main__':unittest.main()
