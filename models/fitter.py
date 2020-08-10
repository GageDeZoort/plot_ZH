import os
import numpy as np
import ROOT
import pickle
from .sample import Sample
from tqdm import tqdm

class Fitter:
    def __init__(self, mode, ES_tool=None, shift='None', 
                 save_table=False, redo_fit=True):
        self.mode = mode
        self.ES_tool = ES_tool
        self.shift = shift
        self.save_table = save_table
        self.redo_fit = redo_fit

        # load in the SVfit dependencies...
        if (mode == 'SVfit'):
            SV_dir = "TauAnalysis/ClassicSVfit/src/"
            SV_files = ["SVfitIntegratorMarkovChain","ClassicSVfitIntegrand", 
                        "ClassicSVfit", "svFitAuxFunctions", "MeasuredTauLepton", 
                        "svFitHistogramAdapter"]
            ROOT.gInterpreter.ProcessLine(".include .")
            for SV_file in SV_files:
                path = "{0}{1}.cc++".format(SV_dir, SV_file)
                ROOT.gInterpreter.ProcessLine(".L {0}".format(path))
            
        # ...or load in the FastMTT dependencies
        elif (mode == 'FastMTT'):
            for baseName in ['../SVFit/MeasuredTauLepton', 
                             '../SVFit/svFitAuxFunctions'
                             ,'../SVFit/FastMTT'] :
                if os.path.isfile("{0:s}_cc.so".format(baseName)) :
                    ROOT.gInterpreter.ProcessLine(".L {0:s}_cc.so"
                                                  .format(baseName))
                else :
                    ROOT.gInterpreter.ProcessLine(".L {0:s}.cc++"
                                                  .format(baseName))
        # otherwise, throw error
        else:
            print("ERROR: initializing fitter with invalid mode '{0}'"
                  .format(mode))


    def fit(self, s):
        
        # grab event info
        run, evt, lumi = s.events.array('run'), s.events.array('evt'), s.events.array('lumi')
    
        # grab MET info
        met, metphi = s.events.array('met'), s.events.array('metphi')
        measuredMETx, measuredMETy = met*np.cos(metphi), met*np.sin(metphi)
        covMET_00, covMET_01 = s.events.array('metcov00'), s.events.array('metcov01')
        covMET_10, covMET_11 = s.events.array('metcov10'), s.events.array('metcov11')
        
        # grab the lepton arrays
        ele_mass, muo_mass = 0.511*10**-3, 0.105
        pt_1, pt_2   = s.events.array('pt_1'),  s.events.array('pt_2')
        eta_1, eta_2 = s.events.array('eta_1'), s.events.array('eta_2')
        phi_1, phi_2 = s.events.array('phi_1'), s.events.array('phi_2')
        
        # grab the tau arrays
        pt_3,  pt_4  = s.events.array('pt_3'),  s.events.array('pt_4')
        eta_3, eta_4 = s.events.array('eta_3'), s.events.array('eta_4')
        phi_3, phi_4 = s.events.array('phi_3'), s.events.array('phi_4')
        m_3,   m_4   = s.events.array('m_3'),   s.events.array('m_4')
        dm_3,  dm_4  = s.events.array('decayMode_3'),  s.events.array('decayMode_4')
        match_3, match_4 = s.events.array('gen_match_3'), s.events.array('gen_match_4')
    
        # grab original mass fit
        m_sv = s.events.array('m_sv')
        progress_bar = tqdm(np.arange(s.n_entries)[s.mask])
        for i in progress_bar:

            # attempt to find value in lookup table
            found_masses = False
            if (self.mode == "SVfit"):
                tag = str(run[i]) + str(evt[i]) + str(lumi[i])
                try: 
                    masses = s.lookup_table[tag]
                    s.mtt_fit[i] = masses['mtt_fit']
                    s.mA[i] = masses['mA']
                    s.mA_c[i] = masses['mA_c']
                    found_masses = True
                except:
                    s.n_recalculated += 1
                    
            if (s.n_recalculated % 2500 == 0 and
                s.n_recalculated >= 2500): s.write_lookup_table()
            
            # if FastMTT, em channel is good-to-go
            if (s.tt[i] == 'em' and self.mode == 'FastMTT'): 
                s.mtt_fit[i] = m_sv[i]
                continue

            # grab the ll 4-vectors
            l1, l2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
            if (s.ll[i] == 'ee'):
                l1.SetPtEtaPhiM(pt_1[i], eta_1[i], phi_1[i], ele_mass)
                l2.SetPtEtaPhiM(pt_2[i], eta_2[i], phi_2[i], ele_mass)
            elif (s.ll[i] == 'mm'):
                l1.SetPtEtaPhiM(pt_1[i], eta_1[i], phi_1[i], muo_mass)
                l2.SetPtEtaPhiM(pt_2[i], eta_2[i], phi_2[i], muo_mass)
            
            # build tau 4-vectors
            t1, t2 = ROOT.TLorentzVector(), ROOT.TLorentzVector()
            t1.SetPtEtaPhiM(pt_3[i], eta_3[i], phi_3[i], m_3[i])
            t2.SetPtEtaPhiM(pt_4[i], eta_4[i], phi_4[i], m_4[i])

            # apply tau ES corrections
            if (s.tt[i] != 'em'):
                if (s.tt[i] == 'tt'): 
                    t1 *= self.ES_tool.getTES(pt_3[i], dm_3[i], match_3[i])
                t2 *= self.ES_tool.getTES(pt_4[i], dm_4[i], match_4[i])

            # store raw 4l mass
            s.m4l[i]  = (l1 + l2 + t1 + t2).M()
            if ((l1+l2+t1+t2).M() < 100): 
                print("eh!!", s.m4l[i], s.ll[i]+s.tt[i], l1, l2, t1, t2)

            # continue past the mass fit if we already found one
            if (found_masses): continue

            # build ROOT objects
            covMET = ROOT.TMatrixD(2,2)
            VectorOfTaus = ROOT.std.vector('MeasuredTauLepton')
            tau_pair = VectorOfTaus()
            covMET[0][0] = covMET_00[i]
            covMET[0][1] = covMET_01[i]
            covMET[1][0] = covMET_10[i]
            covMET[1][1] = covMET_11[i]
            ele_decay = ROOT.MeasuredTauLepton.kTauToElecDecay
            mu_decay  = ROOT.MeasuredTauLepton.kTauToMuDecay
            had_decay = ROOT.MeasuredTauLepton.kTauToHadDecay
            
            if (s.tt[i] == 'et' or s.tt[i] == 'em'):
                tau_pair.push_back(ROOT.MeasuredTauLepton(ele_decay, t1.Pt(), 
                                                          t1.Eta(), t1.Phi(), 
                                                          ele_mass))
            elif (s.tt[i] == 'mt'): 
                tau_pair.push_back(ROOT.MeasuredTauLepton(mu_decay, t1.Pt(), 
                                                          t1.Eta(), t1.Phi(), 
                                                          muo_mass))
            elif (s.tt[i] == 'tt'): 
                tau_pair.push_back(ROOT.MeasuredTauLepton(had_decay, t1.Pt(), 
                                                          t1.Eta(), t1.Phi(), 
                                                          t1.M()))
            if (s.tt[i] != 'em'):
                tau_pair.push_back(ROOT.MeasuredTauLepton(had_decay, t2.Pt(), 
                                                          t2.Eta(), t2.Phi(), 
                                                          t2.M()))
            elif (s.tt[i] == 'em'):
                tau_pair.push_back(ROOT.MeasuredTauLepton(mu_decay, t2.Pt(),
                                                          t2.Eta(), t2.Phi(),
                                                          muo_mass))

            # run SVfit algorithm
            if (self.mode == 'SVfit'):
                svFitAlgo = ROOT.ClassicSVfit(0)
                svFitAlgo.addLogM_fixed(True, 6.)
                svFitAlgo.integrate(tau_pair, measuredMETx[i], measuredMETy[i], covMET)
                mass = svFitAlgo.getHistogramAdapter().getMass()
                s.mtt_fit[i] = mass
                
                # calculate A mass
                tt = ROOT.TLorentzVector()
                tt.SetPtEtaPhiM(svFitAlgo.getHistogramAdapter().getPt(),
                                svFitAlgo.getHistogramAdapter().getEta(),
                                svFitAlgo.getHistogramAdapter().getPhi(),
                                svFitAlgo.getHistogramAdapter().getMass())
                s.mA[i]   = (l1 + l2 + tt).M()
                
                # perform a constrained di-tau mass fit
                massConstraint = 125
                svFitAlgo.setDiTauMassConstraint(massConstraint)
                svFitAlgo.integrate(tau_pair, measuredMETx[i], measuredMETy[i], covMET)
                mass_c = svFitAlgo.getHistogramAdapter().getMass()

                # calculate constrained A mass 
                tt_c = ROOT.TLorentzVector()
                tt_c.SetPtEtaPhiM(svFitAlgo.getHistogramAdapter().getPt(),
                                  svFitAlgo.getHistogramAdapter().getEta(),
                                  svFitAlgo.getHistogramAdapter().getPhi(),
                                  svFitAlgo.getHistogramAdapter().getMass())
                s.mA_c[i] = (l1 + l2 + tt_c).M()
                
                tag = str(run[i]) + str(evt[i]) + str(lumi[i])
                s.lookup_table[tag] = {'mtt_fit':s.mtt_fit[i], 
                                       'mA':s.mA[i], 'mA_c':s.mA_c[i]}
                
            # run FastMTT algorithm
            elif (self.mode == 'FastMTT'):
                FMTT = ROOT.FastMTT()
                FMTT.run(tau_pair, measuredMETx[i], measuredMETy[i], covMET)
                ttP4 = FMTT.getBestP4()
                s.mtt_fit[i] = ttP4.M()
            
        # in case we've added to the lookup table, write it out
        if (self.mode == 'SVfit'): s.write_lookup_table()
            
