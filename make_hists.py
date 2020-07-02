import argparse
import uproot
import numpy as np
import yaml
from tqdm import tqdm
import pickle
import ROOT
import boost_histogram as bh

from models.fitter import Fitter
from models.sample import Sample
from models.group import Group
from models.data import Data
from models.reducible import Reducible
from TauPOG.TauIDSFs.TauIDSFTool import TauIDSFTool
from TauPOG.TauIDSFs.TauIDSFTool import TauESTool
import ScaleFactor as SF
        
# >> python MHBG.py configs/config_MHBG.yaml
parser = argparse.ArgumentParser('MHBG.py')
add_arg = parser.add_argument
add_arg('config', nargs='?', default='configs/config_MHBG.yaml')
args = parser.parse_args()
with open(args.config) as f:
    config = yaml.load(f)

# read in parameters from config file
era_int = config['year']
era = str(config['year'])
tight_cuts = not config['loose_cuts']
loose_cuts = config['loose_cuts']
data_driven = not config['MC_only']
shift_ES = config['shift_ES']
analysis = config['analysis']
sign = config['sign']
tau_ID_SF = config['tau_ID_SF']
LT_cut = config['LT_cut']
redo_fit = config['redo_fit']
mass_reco = config['mass_reco']
unblind = config['unblind']

if shift_ES not in ['None', 'Down', 'Up']:
    raise ValueError("{0} is not a valid tau_ES (please use 'None', 'Down', or 'Up')"
                     .format(tau_ES))

categories = cats = {1:'eeet', 2:'eemt', 3:'eett', 4:'eeem', 
                     5:'mmet', 6:'mmmt', 7:'mmtt', 8:'mmem'}
lumi = {'2016' : 35.92*10**3, '2017' : 41.53*10**3, '2018' : 59.74*10**3}

# configure the tau energy scale by year
campaign  = {2016:'2016Legacy', 2017:'2017ReReco', 2018:'2018ReReco'}
t_ES_tool = TauESTool(campaign[era_int]) # properly ID'd taus
f_ES_tool = TauESTool(campaign[era_int]) # incorrectly ID'd taus

# configure the TauID Scale Factor (SF) Tool
tau_SF     = TauIDSFTool(campaign[era_int], 'DeepTau2017v2p1VSjet', 'Medium')
antiEle_SF = TauIDSFTool(campaign[era_int], 'antiEleMVA6', 'Loose')
antiMu_SF  = TauIDSFTool(campaign[era_int], 'antiMu3', 'Tight')

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
FastMTT = Fitter("FastMTT", ES_tool=t_ES_tool, shift=shift_ES, constrain=False)

# build reducible background analyzer
reducible = Reducible(categories, tau_SF, antiEle_SF, antiMu_SF)
reducible.fitter = FastMTT

# build rare processes analyzer
rare = Group(categories, tau_SF, antiEle_SF, antiMu_SF)
rare.fitter = FastMTT

# build signal processes analyzer
signal = Group(categories, tau_SF, antiEle_SF, antiMu_SF)
signal.fitter = FastMTT

# build ZZ processes analyzer
ZZ = Group(categories, tau_SF, antiEle_SF, antiMu_SF)
ZZ.fitter = FastMTT

# open sample csv file
for line in open("../MC/MCsamples_{0:s}_{1:s}.csv".format(era, analysis), 'r').readlines():
    vals = line.split(',')
    if (vals[5].lower=='ignore'): continue
   
    nickname, group = vals[0], vals[1]
    xsec, total_weight = float(vals[2]), float(vals[4])
    sample_weight = lumi[era]*xsec/total_weight
    path = "../MC/condor/{0:s}/{1:s}_{2:s}/{1:s}_{2:s}.root".format(analysis, nickname, era)
    sample = Sample(nickname, path, xsec, total_weight, sample_weight)
    
    if (group == "Reducible"): reducible.add_sample(sample)
    if (group == "Rare"): rare.add_sample(sample)
    if (group == "Signal"): signal.add_sample(sample)
    if (group == "ZZ"): ZZ.add_sample(sample) 

reducible.reweight_nJets(lumi[era])
signal.reweight_samples(10.0)

print("Analyzing reducible events")    
reducible.process_samples(tight_cuts=tight_cuts, sign=sign, data_driven=data_driven, 
                          tau_ID_SF=tau_ID_SF, redo_fit=redo_fit, LT_cut=LT_cut)

for hist in reducible.cutflow_hists.values():
    print(hist)

print("Analyzing rare events")
rare.process_samples(tight_cuts=tight_cuts, sign=sign, data_driven=data_driven,
                     tau_ID_SF=tau_ID_SF, redo_fit=redo_fit, LT_cut=LT_cut)

print("Analyzing signal events")
signal.process_samples(tight_cuts=tight_cuts, sign=sign, data_driven=data_driven,
                       tau_ID_SF=tau_ID_SF, redo_fit=redo_fit, LT_cut=LT_cut)

print("Analyzing ZZ events")
ZZ.process_samples(tight_cuts=tight_cuts, sign=sign, data_driven=data_driven,
                   tau_ID_SF=tau_ID_SF, redo_fit=redo_fit, LT_cut=LT_cut)

# build a data analyzer
data_path = "../data/condor/{0:s}/{1:s}/{1:s}_data.root".format(analysis, era)
data = Data(categories, tau_SF, antiEle_SF, antiMu_SF, era_int)
data_sample = Sample('data', data_path, 1.0, 1.0, 1.0)
data.add_sample(data_sample)
data.process_samples(tight_cuts=tight_cuts, sign=sign, data_driven=data_driven,
                     tau_ID_SF=tau_ID_SF, redo_fit=redo_fit, LT_cut=LT_cut)

# write output to pickle file
for cat in categories.values():
    outfile = open("histograms/{0}_hists.pkl".format(cat), "w+")
    hists = {}
    hists['data'] = data.get_hists(cat)
    hists['reducible'] = reducible.get_hists(cat)
    hists['signal'] = signal.get_hists(cat)
    hists['rare'] = rare.get_hists(cat)
    hists['ZZ'] = ZZ.get_hists(cat)
    pickle.dump(hists, outfile)
    outfile.close()
