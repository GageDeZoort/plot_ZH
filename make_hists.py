import sys
import argparse
import uproot
import numpy as np
import yaml
from tqdm import tqdm
import pickle
import ROOT
import boost_histogram as bh
import mplhep as hep
from matplotlib import pyplot as plt

from models.fitter import Fitter
from models.sample import Sample
from models.group import Group
from models.data import Data
from models.reducible import Reducible
sys.path.append("../../TauPOG/TauIDSFs/python/")
from TauIDSFTool import TauIDSFTool
from TauIDSFTool import TauESTool
import ScaleFactor as SF
        
# >> python MHBG.py configs/config_MHBG.yaml
parser = argparse.ArgumentParser('MHBG.py')
add_arg = parser.add_argument
add_arg('config', nargs='?', default='configs/config_MHBG.yaml')
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f)

# read in parameters from config file
era, era_int = str(config['year']), config['year']
tight_cuts, loose_cuts = not config['loose_cuts'], config['loose_cuts']
loose_cuts = config['loose_cuts']
data_driven = config['data_driven']
shift_ES = config['shift_ES']
analysis = config['analysis']
sign = config['sign']
tau_ID_SF = config['tau_ID_SF']
LT_cut = config['LT_cut']
redo_fit = config['redo_fit']
fitter = config['fitter']
data_dir = config['data_dir']
mass = config['mass']

if shift_ES not in ['None', 'Down', 'Up']:
    raise ValueError("{0} is not a valid tau_ES (please use 'None', 'Down', or 'Up')"
                     .format(tau_ES))

categories = {1:'eeet', 2:'eemt', 3:'eett', 4:'eeem', 
              5:'mmet', 6:'mmmt', 7:'mmtt', 8:'mmem'}
lumi = {'2016' : 35.92*10**3, '2017' : 41.53*10**3, '2018' : 59.74*10**3}

# configure the tau energy scale by year
campaign  = {2016:'2016Legacy', 2017:'2017ReReco', 2018:'2018ReReco'}
t_ES_tool = TauESTool(campaign[era_int]) # properly ID'd taus
f_ES_tool = TauESTool(campaign[era_int]) # incorrectly ID'd taus

# configure the TauID Scale Factor (SF) Tool
antiJet_SF = TauIDSFTool(campaign[era_int], 'DeepTau2017v2p1VSjet', 'Medium')
antiEle_SF = TauIDSFTool(campaign[era_int], 'antiEleMVA6', 'Loose')
antiMu_SF  = TauIDSFTool(campaign[era_int], 'antiMu3', 'Tight')

# trigger scale factors
mu_files   = {2016:'SingleMuon_Run2016_IsoMu24orIsoMu27.root',
              2017:'SingleMuon_Run2017_IsoMu24orIsoMu27.root',
              2018:'SingleMuon_Run2018_IsoMu24orIsoMu27.root'}
ele_files  = {2016:'SingleElectron_Run2016_Ele25orEle27.root',
              2017:'SingleElectron_Run2017_Ele32orEle35.root',
              2018:'SingleElectron_Run2018_Ele32orEle35.root'}
trigger_SF = {'dir':'../tools/ScaleFactors/TriggerEffs/',
              'fileMuon':'Muon/{0:s}'.format(mu_files[era_int]),
              'fileElectron':'Electron/{0:s}'.format(ele_files[era_int])}

mu_trig_SF = SF.SFs()
mu_trig_SF.ScaleFactor("{0:s}{1:s}".format(trigger_SF['dir'],
                                           trigger_SF['fileMuon']))
ele_trig_SF = SF.SFs()
ele_trig_SF.ScaleFactor("{0:s}{1:s}".format(trigger_SF['dir'],
                                            trigger_SF['fileElectron']))

# build diTau mass fitter
FastMTT = Fitter(config['fitter'], ES_tool=t_ES_tool, shift=shift_ES, save_table=False, redo_fit=False)

# build analyzers for each MC group
reducible = Reducible(categories, antiJet_SF, antiEle_SF, antiMu_SF, fitter=FastMTT)
rare = Group(categories, antiJet_SF, antiEle_SF, antiMu_SF, fitter=FastMTT)
signal = Group(categories, antiJet_SF, antiEle_SF, antiMu_SF, fitter=FastMTT)
ZZ = Group(categories, antiJet_SF, antiEle_SF, antiMu_SF, fitter=FastMTT)
MC_groups = {"Reducible" : reducible, "Rare" : rare, "Signal" : signal, "ZZ" : ZZ}

# open sample csv file
for line in open("../MC/MCsamples_{0:s}_{1:s}.csv".format(era, analysis), 'r').readlines():
    vals = line.split(',')
    if (vals[5].lower() == 'ignore'): continue
    nickname, group = vals[0], vals[1]
    if (analysis == 'AZH' and 'AToZh' in nickname):
        if (str(mass) not in nickname): continue 
    xsec, total_weight = float(vals[2]), float(vals[4])
    sample_weight = lumi[era]*xsec/total_weight 
    path = "../MC/condor/{0:s}/{1:s}_{2:s}/{1:s}_{2:s}.root".format(analysis, nickname, era)
    sample = Sample(nickname, path, xsec, total_weight, sample_weight,
                    lookup_path="lookup_tables")
    MC_groups[group].add_sample(sample)
    print(" ... added {0} to {1}".format(nickname, group))

reducible.reweight_nJets(lumi[era])
#signal.reweight_samples(10.0)

for group in MC_groups.keys():
    print("Analyzing {0} events".format(group.lower()))    

    # add "free" hists from ntuple
    for var, hist in config['var_hists'].items():
        MC_groups[group].add_hist(var, hist[0], hist[1], hist[2], from_ntuple=True)

    if (group != "Signal"): continue
    MC_groups[group].process_samples(tight_cuts=tight_cuts, sign=sign, data_driven=data_driven, 
                                     tau_ID_SF=tau_ID_SF, redo_fit=redo_fit, LT_cut=LT_cut)

# build a data analyzer
#data_path = data_dir + "/condor/{0:s}/{1:s}/{1:s}_data.root".format(analysis, era)
#data = Data(categories, antiJet_SF, antiEle_SF, antiMu_SF, era_int)
#data_sample = Sample('data', data_path, 1.0, 1.0, 1.0)
#data.add_sample(data_sample)
#data.process_samples(tight_cuts=tight_cuts, sign=sign, data_driven=data_driven,
#                     tau_ID_SF=tau_ID_SF, redo_fit=redo_fit, LT_cut=LT_cut)


# ---------- output histograms ---------- 
def output_hists(group, hists, cat=None):
    outdir = "/eos/uscms/store/user/jdezoort/AZH_hists"
    with open("{0}/{1}_{2}_M{3}_{4}.pkl"
              .format(outdir, analysis, era, mass, group), 'wb') as f:
        pickle.dump(hists, f, protocol=pickle.HIGHEST_PROTOCOL)

def output_root(group, hists, cat=None):
    outdir = "/eos/uscms/store/user/jdezoort/AZH_hists"
    root_file = uproot.recreate("{0}/{1}_{2}_M{3}_{4}.root"
                                .format(outdir, analysis, era, mass, group))
    for name, hists_per_cat in hists.items():
        print(name, hists_per_cat)
        for cat, hist in hists_per_cat.items():
            #print(cat, hist)
            root_file["{0}_{1}".format(cat, name)] = hist.to_numpy()
        
outdir = "/eos/uscms/store/user/jdezoort/AZH_hists"
for group in MC_groups.keys(): 
    output_hists(group.lower(), MC_groups[group].get_hists())
    output_root(group.lower(), MC_groups[group].get_hists())
    #output_hists("data", data.get_hists())

"""def pickle_hists(cat, hists, tag):
    for name, hist in hists.items():
        print("histograms/{0}_{1}_{2}.pkl".format(tag, cat, name), hist)
        with open("histograms/{0}_{1}_{2}.pkl".format(tag, cat, name), 'wb') as f:
            pickle.dump(hist, f, protocol=pickle.HIGHEST_PROTOCOL)

# write output to pickle file
for cat in categories.values():
    outfile = open("histograms/{0}_hists.pkl".format(cat), "w+")
    pickle_hists(cat, data.get_core_hists(cat), "data")
    pickle_hists(cat, data.get_extra_hists(cat), "data")
    for group in MC_groups.keys():
        pickle_hists(cat, MC_groups[group].get_core_hists(cat), group.lower())
        pickle_hists(cat, MC_groups[group].get_extra_hists(cat), group.lower())
""" 


