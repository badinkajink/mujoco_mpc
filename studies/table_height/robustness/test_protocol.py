import unittest,json
from pathlib import Path
import numpy as np
from evaluate import longest

class ProtocolTests(unittest.TestCase):
    def test_single_sample_has_zero_duration(self):
        self.assertEqual(longest([5],[True]),(0.,None))
    def test_failed_sample_breaks_hold(self):
        t=np.arange(0,4.02,.02);mask=np.ones(len(t),bool);mask[100]=False
        self.assertLess(longest(t,mask)[0],2.)
    def test_missing_sample_breaks_hold(self):
        t=np.r_[np.arange(0,1,.02),np.arange(3,4,.02)]
        self.assertLess(longest(t,np.ones(len(t),bool))[0],1.)
    def test_two_seconds_not_sample_count(self):
        t=np.arange(101)*.02
        self.assertAlmostEqual(longest(t,np.ones(101,bool))[0],2.)
        self.assertLess(longest(t[:-1],np.ones(100,bool))[0],2.)
    def test_ablation_is_one_factor(self):
        vs=json.loads(Path(__file__).with_name('variants.json').read_text());r=vs['reference']
        expected={'no_pose':'pose','legacy_model':'sync','far_target':'depth','short_hold':'dwell'}
        for name,key in expected.items():self.assertEqual([k for k in r if r[k]!=vs[name][k]],[key])
    def test_fresh_seeds_and_balanced_cells(self):
        jobs=json.loads(Path(__file__).with_name('ablation.json').read_text());counts={}
        for j in jobs:
            self.assertIn(j['seed'],[10,11,12]);key=(j['arm'],j['h']);counts[key]=counts.get(key,0)+1
        self.assertEqual(set(counts.values()),{3});self.assertEqual(len({j['tag'] for j in jobs}),len(jobs))

if __name__=='__main__':unittest.main()
