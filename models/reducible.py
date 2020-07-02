import uproot
import numpy as np
from tqdm import tqdm
import ROOT
import boost_histogram as bh

from fitter import Fitter
from sample import Sample
from group  import Group
from TauPOG.TauIDSFs.TauIDSFTool import TauIDSFTool
from TauPOG.TauIDSFs.TauIDSFTool import TauESTool
import ScaleFactor as SF

class Reducible(Group):
    def __init__(self, categories, tau_SF, antiEle_SF, antiMu_SF):
        Group.__init__(self, categories, tau_SF, antiEle_SF, antiMu_SF)

    def reweight_nJets(self, lumi):
        for i in range(1, 5):
            DYnJets = "DY{0:d}JetsToLL".format(i)
            norm_1 = self.samples['DYJetsToLL'].total_weight/self.samples['DYJetsToLL'].x_sec
            norm_2 = self.samples[DYnJets].total_weight/self.samples[DYnJets].x_sec
            self.samples[DYnJets].sample_weight = lumi/(norm_1 + norm_2)

        for i in range(1, 4):
            WnJets = "W{0:d}JetsToLNu".format(i)
            norm_1 = self.samples['WJetsToLNu'].total_weight/self.samples['WJetsToLNu'].x_sec
            norm_2 = self.samples[WnJets].total_weight/self.samples[WnJets].x_sec
            self.samples[WnJets].sample_weight = lumi/(norm_1 + norm_2)

    def process_samples(self, tight_cuts, sign, data_driven, tau_ID_SF, redo_fit, LT_cut):
        progress_bar= tqdm(self.samples.items())
        for name, sample in progress_bar:
            if (sample.n_entries < 1): continue
            progress_bar.set_description("{0}".format(name.ljust(20)[:20]))
            if (name == "DYJetsToLL" or name == "WJetsToLNu"):
                self.reweight_nJet_events(sample, sample.events.array('LHE_Njets'))
            sample.weights *= sample.events.array('weightPUtrue')
            sample.weights *= sample.events.array('Generator_weight')
            sample.parse_categories(self.categories, sample.events.array('cat'))
            self.fill_cutflow(0.5, sample)

            if (tight_cuts):
                self.sign_cut(sample, sign, fill_value=1.5)
                self.btag_cut(sample, fill_value=2.5)
                self.lepton_cut(sample, fill_value=3.5)
                tight1, tight2 = self.get_tight_taus(sample)
                self.tau_cut(sample, tight1, tight2, fill_value=4.5)

            if (data_driven):
                self.data_driven_cut(sample, fill_value=5.5)
                match_3 = sample.events.array('gen_match_3')
                match_4 = sample.events.array('gen_match_4')
                sample.mask[match_4 != 5] = False
                sample.mask[(sample.tt == 'tt') & (match_3 != 5) & (match_4 != 5)] = False

                if tau_ID_SF: self.add_SFs(sample)

            self.H_LT_cut(LT_cut, sample, fill_value=6.6)
            if (redo_fit): self.fitter.fit(sample)
            self.fill_hists(sample)

    def reweight_nJet_events(self, sample, LHE_nJets):
        for j in np.where(LHE_nJets > 0)[0]:
            nJets = LHE_nJets[j]
            if (sample.name == "DYJetsToLL"):
                DYnJets = "DY{0}JetsToLL".format(nJets)
                self.samples[sample.name].weights[j] = self.samples[DYnJets].sample_weight

            if (sample.name == "WJetsToLNu"):
                WnJets = "W{0}JetsToLNu".format(nJets)
                self.samples[sample.name].weights[j] = self.samples[WnJets].sample_weight
