import sys
import uproot
import numpy as np
import yaml
from tqdm import tqdm
import ROOT
import boost_histogram as bh

from .fitter import Fitter
from .sample import Sample
from .group  import Group
sys.path.append("../../TauPOG/TauIDSFs/python/")
from TauIDSFTool import TauIDSFTool
from TauIDSFTool import TauESTool
import ScaleFactor as SF
import fakeFactor2

class Data(Group):
    def __init__(self, categories, antiJet_SF, antiEle_SF, antiMu_SF, year, fitter=None):
        Group.__init__(self, categories, antiJet_SF, antiEle_SF, antiMu_SF, fitter=None)
        self.year = year
        self.h_group = []

    def get_fake_weights(self, f1, f2):
        w1 = f1/(1.0-f1)
        w2 = f2/(1.0-f2)
        return w1, w2, w1*w2

    def apply_fake_weights(self, sample, tight1, tight2, WP=16):
        fe, fm, ft_et, ft_mt, f1_tt, f2_tt   = 0.0390, 0.0794, 0.1397, 0.1177, 0.0756, 0.0613
        fW1, fW2, fW0 = {}, {}, {}
        fW1['et'], fW2['et'], fW0['et'] = self.get_fake_weights(fe, ft_et)
        fW1['mt'], fW2['mt'], fW0['mt'] = self.get_fake_weights(fm, ft_mt)
        fW1['tt'], fW2['tt'], fW0['tt'] = self.get_fake_weights(f1_tt, f2_tt)
        fW1['em'], fW2['em'], fW0['em'] = self.get_fake_weights(fe, fm)

        self.h_group = np.array(['Reducible' for _ in range(sample.n_entries)])
        for i, (t1, t2) in enumerate(zip(tight1, tight2)):
            if (not t1) and t2:   sample.weights[i] = fW1[sample.tt[i]]
            elif t1 and (not t2): sample.weights[i] = fW2[sample.tt[i]]
            elif not (t1 or t2):  sample.weights[i] = -fW0[sample.tt[i]]
            else:
                sample.weights[i] = 1.0
                self.h_group[i] = 'data'

    def process_samples(self, tight_cuts, sign, data_driven, tau_ID_SF, redo_fit, LT_cut):
        progress_bar= tqdm(self.samples.items())
        for name, sample in progress_bar:
            if (sample.n_entries < 1): continue
            progress_bar.set_description("{0}".format(name.ljust(20)[:20]))
            sample.weights = np.ones(sample.n_entries)
            sample.parse_categories(self.categories, sample.events.array('cat'))

            self.fill_cutflow(0.5, sample)

            if (tight_cuts):
                self.sign_cut(sample, sign, fill_value=1.5)
                self.btag_cut(sample, fill_value=2.5)
                self.lepton_cut(sample, fill_value=3.5)
                tight1, tight2 = self.get_tight_taus(sample)

                if (data_driven):
                    self.apply_fake_weights(sample, tight1, tight2, WP=16)
                else:
                    h_group = ['data' for _ in range(sample.n_entries)]
                    self.tau_cut(sample, tight1, tight2, fill_value=4.5)

            self.fill_cutflow(5.5, sample)
            self.H_LT_cut(LT_cut, sample, fill_value=6.5)
            #self.mtt_fit_cut(sample, fill_value=7.5)
            self.fill_hists(sample, blind=True)
