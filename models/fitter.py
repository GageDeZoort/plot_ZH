import os
import numpy as np
import ROOT

from sample import Sample

class Fitter:
    def __init__(self, mode, ES_tool=None, shift='None', constrain=False):
        self.mode = mode
        self.ES_tool = ES_tool
        self.shift = shift
        self.constrain = constrain

        if (mode == 'SVfit'):
            SV_dir = "TauAnalysis/ClassicSVfit/src/"
            SV_files = ["SVfitIntegratorMarkovChain","ClassicSVfitIntegrand", "ClassicSVfit",
                        "svFitAuxFunctions", "MeasuredTauLepton", "svFitHistogramAdapter"]
            ROOT.gInterpreter.ProcessLine(".include .")
            for SV_file in SV_files:
                path = "{0}{1}.cc++".format(SV_dir, SV_file)
                ROOT.gInterpreter.ProcessLine(".L {0}".format(path))
            
        elif (mode == 'FastMTT'):
            for baseName in ['../SVFit/MeasuredTauLepton', '../SVFit/svFitAuxFunctions'
                             ,'../SVFit/FastMTT'] :
                if os.path.isfile("{0:s}_cc.so".format(baseName)) :
                    ROOT.gInterpreter.ProcessLine(".L {0:s}_cc.so".format(baseName))
                else :
                    ROOT.gInterpreter.ProcessLine(".L {0:s}.cc++".format(baseName))
        else:
            print("ERROR: initializing fitter with invalid mode '{0}'".format(mode))


    def fit(self, s):
        met, metphi = s.events.array('met'), s.events.array('metphi')
        measuredMETx, measuredMETy = met*np.cos(metphi), met*np.sin(metphi)
        covMET_00, covMET_01 = s.events.array('metcov00'), s.events.array('metcov01')
        covMET_10, covMET_11 = s.events.array('metcov10'), s.events.array('metcov11')
        
        ele_mass, muo_mass = 0.511*10**-3, 0.105
        pt_3,  pt_4  = s.events.array('pt_3'),  s.events.array('pt_4')
        eta_3, eta_4 = s.events.array('eta_3'), s.events.array('eta_4')
        phi_3, phi_4 = s.events.array('phi_3'), s.events.array('phi_4')
        m_3,   m_4   = s.events.array('m_3'),   s.events.array('m_4')
        dm_3,  dm_4  = s.events.array('decayMode_3'),  s.events.array('decayMode_4')
        match_3, match_4 = s.events.array('gen_match_3'), s.events.array('gen_match_4')
    
        m_sv = s.events.array('m_sv')
        for i in range(s.n_entries):
            if (not s.mask[i]): continue 

            if (s.tt[i] == 'em'): 
                s.m_fit[i] = m_sv[i]
                continue

            tau_1, tau_2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
            tau_1.SetPtEtaPhiM(pt_3[i], eta_3[i], phi_3[i], m_3[i])
            tau_2.SetPtEtaPhiM(pt_4[i], eta_4[i], phi_4[i], m_4[i])

            # shift tau_1 too? 
            if (s.tt[i] == 'tt'): tau_1 *= self.ES_tool.getTES(pt_3[i], dm_3[i], match_3[i])
            if (self.shift.lower != 'none'):
                tau_2 *= self.ES_tool.getTES(pt_4[i], dm_4[i], match_4[i], self.shift)

            covMET = ROOT.TMatrixD(2,2)
            VectorOfTaus = ROOT.std.vector('MeasuredTauLepton')
            measuredTaus = VectorOfTaus()

            covMET[0][0] = covMET_00[i]
            covMET[0][1] = covMET_01[i]
            covMET[1][0] = covMET_10[i]
            covMET[1][1] = covMET_11[i]

            ele_decay = ROOT.MeasuredTauLepton.kTauToElecDecay
            mu_decay  = ROOT.MeasuredTauLepton.kTauToMuDecay
            had_decay = ROOT.MeasuredTauLepton.kTauToHadDecay
            if (s.tt[i] == 'et'):
                measuredTaus.push_back(ROOT.MeasuredTauLepton(ele_decay, tau_1.Pt(), 
                                                              tau_1.Eta(), tau_1.Phi(), ele_mass))
            elif (s.tt[i] == 'mt'): 
                measuredTaus.push_back(ROOT.MeasuredTauLepton(mu_decay, tau_1.Pt(), 
                                                              tau_1.Eta(), tau_1.Phi(), muo_mass))
            elif (s.tt[i] == 'tt'): 
                measuredTaus.push_back(ROOT.MeasuredTauLepton(had_decay, tau_1.Pt(), 
                                                              tau_1.Eta(), tau_1.Phi(), tau_1.M()))
            
            measuredTaus.push_back(ROOT.MeasuredTauLepton(had_decay, tau_2.Pt(), 
                                                          tau_2.Eta(), tau_2.Phi(), tau_2.M()))

            if (self.mode == 'SVfit'):
                svFitAlgo = ROOT.ClassicSVfit(0)
                svFitAlgo.addLogM_fixed(True, 6.)
                if (self.constrain):
                    massConstraint = 125
                    svFitAlgo.setDiTauMassConstraint(massConstraint)
                svFitAlgo.integrate(measuredTaus, measuredMETx[i], measuredMETy[i], covMET)
                valid_c = svFitAlgo.isValidSolution()
                s.m_fit[i] = svFitAlgo.getHistogramAdapter().getMass()

            else:
                FMTT = ROOT.FastMTT()
                FMTT.run(measuredTaus, measuredMETx[i], measuredMETy[i], covMET)
                ttP4 = FMTT.getBestP4()
                s.m_fit[i] = ttP4.M()
            print(m_sv[i], " ", s.m_fit[i])

