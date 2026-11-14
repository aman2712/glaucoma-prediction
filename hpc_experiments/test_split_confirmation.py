import tempfile
import unittest
from pathlib import Path
import pandas as pd
import split_confirmation as cv


class Checks(unittest.TestCase):
    def fixture(self):
        return pd.DataFrame([dict(image_id=f'image_{i}',analysis_group=f'patient_{i//2}',
                                  binary_label=(i//2)%2) for i in range(200)])

    def test_bilateral_groups_stay_together_and_outer_coverage_is_exact(self):
        data=self.fixture();table=cv.assignments(data)
        held=table[table.role=='held']
        self.assertEqual(len(held),len(data))
        self.assertFalse(held.image_id.duplicated().any())
        for _,rows in table.groupby('fold'):
            self.assertTrue((rows.groupby('analysis_group').role.nunique()==1).all())
            self.assertEqual(set(rows.image_id),set(data.image_id))

    def test_assignments_are_deterministic_under_input_reordering(self):
        a=cv.assignments(self.fixture())
        b=cv.assignments(self.fixture().sample(frac=1,random_state=42))
        pd.testing.assert_frame_equal(a,b)

    def test_empty_group_and_duplicate_identity_rejected(self):
        data=self.fixture();data.loc[0,'analysis_group']=''
        with self.assertRaises(ValueError):cv.assignments(data)
        data=self.fixture();data.loc[0,'image_id']=data.loc[1,'image_id']
        with self.assertRaises(ValueError):cv.assignments(data)


if __name__=='__main__':unittest.main()
