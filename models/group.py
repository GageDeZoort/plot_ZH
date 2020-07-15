import uproot
import numpy as np
import yaml
from tqdm import tqdm
import ROOT
import boost_histogram as bh

from .fitter import Fitter
from .sample import Sample
from TauPOG.TauIDSFs.TauIDSFTool import TauIDSFTool
from TauPOG.TauIDSFs.TauIDSFTool import TauESTool
import ScaleFactor as SF
import fakeFactor2

class Group(object):
    def __init__(self, categories, tau_SF, antiEle_SF, antiMu_SF):
        self.categories = categories
        self.tau_SF = tau_SF
        self.antiEle_SF = antiEle_SF
        self.antiMu_SF = antiMu_SF
        self.samples = {}
        self.fitter = None 
        
        self.mtt_hists = {cat:bh.Histogram(bh.axis.Regular(10, 0, 200)) 
                          for cat in categories.values()}
        self.LT_hists  = {cat:bh.Histogram(bh.axis.Regular(10, 0, 200))
                          for cat in categories.values()}
        self.pt1_hists = {cat:bh.Histogram(bh.axis.Regular(10, 0, 200))
                          for cat in categories.values()}
        self.ESratio_hists = {cat:bh.Histogram(bh.axis.Regular(200, 0.9, 1.1))
                              for cat in categories.values()}
        self.cutflow_hists = {cat:bh.Histogram(bh.axis.Regular(20,  0.0, 20.0))
                              for cat in categories.values()}

    def get_hists(self, cat):
        return {'mtt':self.mtt_hists[cat], 'LT':self.LT_hists[cat], 'pt1':self.pt1_hists[cat],
                'ESratio':self.ESratio_hists[cat], 'cutflow':self.cutflow_hists[cat]}
        
    def add_sample(self, sample):
        sample.get_events()
        if (sample.n_entries == 0):
            print("WARNING: {0} has {1} entries"
                  .format(sample.name, sample.n_entries))
        self.samples[sample.name] = sample

    def reweight_samples(self, factor):
        for sample in self.samples.values():
            sample.weights *= factor

    def fill_cutflow(self, fill_value, sample):
        for cat in self.categories.values():
            good_evts = (sample.cats == cat) & sample.mask
            to_fill = np.ones(np.count_nonzero(good_evts)) * fill_value
            self.cutflow_hists[cat].fill(to_fill, weight=sample.weights[good_evts])

    def sign_cut(self, sample, sign, fill_value):
        q_3, q_4 = sample.events.array('q_3'), sample.events.array('q_4')
        signs = np.dot(q_3, q_4)
        if (sign == 'SS'): sample.mask[signs < 0] = False
        elif (sign == 'OS'): sample.mask[signs > 0] = False
        self.fill_cutflow(fill_value, sample)

    def btag_cut(self, sample, fill_value):
        nbtag = sample.events.array('nbtag')
        try: condition = (nbtag[:,0] > 0)
        except: condition = (nbtag > 0)
        sample.mask[condition] = False
        self.fill_cutflow(fill_value, sample)

    def lepton_cut(self, s, fill_value):
        iso_1, iso_2 = s.events.array('iso_1'), s.events.array('iso_2')
        global_1, global_2 = s.events.array('isGlobal_1'), s.events.array('isGlobal_2')
        tracker_1, tracker_2 = s.events.array('isTracker_2'), s.events.array('isTracker_2')
        disc_1 = s.events.array('Electron_mvaFall17V2noIso_WP90_1')
        disc_2 = s.events.array('Electron_mvaFall17V2noIso_WP90_2')

        # tight muon selections
        mm_iso = (iso_1 > 0.2) | (iso_2 > 0.2)
        if (s.name == "W1JetsToLNu"):
            print("iso_1", iso_1)
            print("(iso_1 > 0.2)", (iso_1 > 0.2))
            print("mm_iso=", mm_iso)
        mm_io  = ((global_1 < 1) & (tracker_1 < 1)) | ((global_2 < 1) & (tracker_2 < 1))
        mm_selections = (mm_iso | mm_io) & (s.ll == 'mm')

        # tight electron selections
        ee_iso = (iso_1 > 0.15) | (iso_2 > 0.15)
        ee_selections = (ee_iso | (disc_1 < 1) | (disc_2 < 1)) & (s.ll == 'ee')

        s.mask[(ee_selections | mm_selections)] = False
        self.fill_cutflow(fill_value, s)

    def get_tight_taus(self, sample):
        iso_3     = sample.events.array('iso_3')
        iso_4     = sample.events.array('iso_4')
        vsJet_3   = sample.events.array('idDeepTau2017v2p1VSjet_3') 
        vsJet_4   = sample.events.array('idDeepTau2017v2p1VSjet_4')
        vsMu_3    = sample.events.array('idDeepTau2017v2p1VSmu_3')
        vsMu_4    = sample.events.array('idDeepTau2017v2p1VSmu_4')
        vsEle_3   = sample.events.array('idDeepTau2017v2p1VSe_3')
        vsEle_4   = sample.events.array('idDeepTau2017v2p1VSe_4')
        global_3  = sample.events.array('isGlobal_3')
        global_4  = sample.events.array('isGlobal_4')
        tracker_3 = sample.events.array('isTracker_3')
        tracker_4 = sample.events.array('isTracker_4')
        disc_3    = sample.events.array('Electron_mvaFall17V2noIso_WP90_3')

        # tight em selections
        em_tight1 = (sample.tt == 'em') & (iso_3 < 0.15) & (disc_3 > 0)
        em_tight2 = (sample.tt == 'em') & (iso_4 < 0.15) & ((global_4 > 0) | (tracker_4 > 0))

        # tight mt selections
        mt_tight1 = (sample.tt == 'mt') & (iso_3 < 0.15)  & ((global_3 > 0) | (tracker_3 > 0))
        mt_tight2 = (sample.tt == 'mt') & (vsJet_4 >= 15) & (vsMu_4 >= 0) & (vsEle_4 >= 0)

        # tight et selections
        et_tight1 = (sample.tt == 'et') & (iso_3 < 0.15)  & (disc_3 > 0)
        et_tight2 = (sample.tt == 'et') & (vsJet_4 >= 15) & (vsMu_4 >= 0) & (vsEle_4 >= 0)

        # tight tt selections
        tt_tight1 = (sample.tt == 'tt') & (vsJet_3 >= 15) & (vsMu_3 >= 0) & (vsEle_3 >= 0)
        tt_tight2 = (sample.tt == 'tt') & (vsJet_4 >= 15) & (vsMu_4 >= 0) & (vsEle_4 >= 0)

        tight1 = em_tight1 | mt_tight1 | et_tight1 | tt_tight1
        tight2 = em_tight2 | mt_tight2 | et_tight2 | tt_tight2
        return tight1, tight2

    def tau_cut(self, sample, tight1, tight2, fill_value=0):
        tight = tight1 & tight2       
        not_tight = np.array([not t for t in tight], dtype=bool)
        sample.mask[not_tight] = False
        self.fill_cutflow(fill_value, sample)
        return tight1, tight2

    def data_driven_cut(self, sample, fill_value):
        match_3 = sample.events.array('gen_match_3')
        match_4= sample.events.array('gen_match_4')
        em_cut = (sample.tt == 'em') & ((match_4 == 15) | (match_3 == 15))
        et_mt_cut = ((sample.tt == 'et') | (sample.tt == 'mt')) & ((match_3 == 15) | (match_4 > 5))
        sample.mask[et_mt_cut | em_cut] = False
        sample.mask[(sample.tt == 'tt') & ((match_3 > 5) | (match_4 > 5))] = False
        self.fill_cutflow(fill_value, sample)
        
    def add_SFs(self, sample):
        pt_3, pt_4 = sample.events.array('pt_3'), sample.events.array('pt_4')
        eta_3, eta_4 = sample.events.array('eta_3'), sample.events.array('eta_4')
        match_3 = sample.events.array('gen_match_3')
        match_4 = sample.events.array('gen_match_4')

        for i in range(sample.n_entries):
            if (sample.tt[i] == 'et' or sample.tt[i] == 'mt'):
                if (match_4[i] == 1 or match_4[i] == 3):
                    sample.weights[i] *= self.antiEle_SF.getSFvsEta(eta_4[i], match_4[i])
                if (match_4[i] == 2 or match_4[i] == 4):
                    sample.weights[i] *= self.antiMu_SF.getSFvsEta(eta_4[i], match_4[i])
            elif (sample.tt[i] == 'tt'):
                sample.weights[i] *= self.tau_SF.getSFvsPT(pt_3[i], match_3[i]) 
                sample.weights[i] *= self.tau_SF.getSFvsPT(pt_4[i], match_4[i])
                if (match_3[i] == 1 or match_3[i] == 3):
                    sample.weights[i] *= self.antiEle_SF.getSFvsEta(eta_3[i], match_3[i])
                if (match_3[i] == 2 or match_3[i] == 4):
                    sample.weights[i] *= self.antiMu_SF.getSFvsEta(eta_3[i], match_3[i])
                if (match_4[i] == 1 or match_4[i] == 3):
                    sample.weights[i] *= self.antiEle_SF.getSFvsEta(eta_4[i], match_4[i])
                if (match_4[i] == 2 or match_4[i] == 4):
                    sample.weights[i] *= self.antiMu_SF.getSFvsEta(eta_4[i], match_4[i])
            
    def H_LT_cut(self, LT_cut, sample, fill_value):
        pt_3, pt_4 = sample.events.array('pt_4'), sample.events.array('pt_3')
        sample.mask[(pt_3 + pt_4) < LT_cut] = False
        self.fill_cutflow(fill_value, sample)

    def fill_hists(self, sample, blind=False):
        for cat in self.categories.values():
            m_sv = sample.events.array('m_sv')
            good_evts = (sample.cats == cat) & sample.mask
            if (blind): 
                good_evts = good_evts & ((m_sv < 80.) | (m_sv > 140.))

            weights = sample.weights[good_evts]
            self.mtt_hists[cat].fill(sample.m_fit[good_evts], weight=weights)
            self.ESratio_hists[cat].fill(sample.m_fit[good_evts]/m_sv[good_evts])
            
            LT = sample.events.array('pt_3') + sample.events.array('pt_4')
            self.LT_hists[cat].fill(LT[good_evts], weight=weights)
            
            self.pt1_hists[cat].fill(sample.events.array('pt_1')[good_evts], 
                                     weight=weights)
            
    def process_samples(self, tight_cuts, sign, data_driven, tau_ID_SF, redo_fit, LT_cut):
        progress_bar= tqdm(self.samples.items())
        for name, sample in progress_bar:
            if (sample.n_entries < 1): continue
            progress_bar.set_description("{0}".format(name.ljust(20)[:20]))
            sample.weights *= sample.sample_weight
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
                if tau_ID_SF: self.add_SFs(sample)

            self.H_LT_cut(LT_cut, sample, fill_value=6.6)
            if (redo_fit): self.fitter.fit(sample)
            self.fill_hists(sample)
