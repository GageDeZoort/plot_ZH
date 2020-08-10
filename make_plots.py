import os 
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import boost_histogram as bh
import ROOT as root
from ROOT import gROOT
import mplhep as hep
import aghast

# plot styling 
plt.style.use(hep.style.CMS) 
plt.rcParams.update({
        'font.size': 14,
        'mathtext.default' : 'regular',
        'axes.titlesize': 18,
        'axes.labelsize': 18,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12
})

def combine_cats(hists):
    all_cats_hist = hists['eemt']
    for cat, cat_hist in hists.items():
        if (cat=='eemt'): continue
        all_cats_hist += cat_hist
    return all_cats_hist

def get_hists(year, mass, group):
    path = "/eos/uscms/store/user/jdezoort/AZH_hists"
    hists_path = path + "/AZH_{0}_M{1}_{2}.pkl".format(year, mass, group)
    hists_file = open(hists_path, 'rb')
    try: hists = pickle.load(hists_file)
    except EOFError: print("Error: {0} not found".format(hists_path))
    else: print("READING {0}".format(hists_path))
    return hists

def make_plot(hist, cat, var, xlabel, ylabel="Events", tag=""):
    #plt.style.use([hep.style.ROOT, hep.style.firamath])
    f, ax = plt.subplots()
    ax.bar(hist.axes[0].centers, hist.view(), width=hist.axes[0].widths,
           color="mediumseagreen", edgecolor="black", linewidth=5)
    hep.cms.label(loc=0, year='2018')
    print(xlabel.encode('unicode_escape').decode())
    plt.xlabel(xlabel.encode('unicode_escape').decode())
    plt.ylabel(ylabel)
    plt.ylim(bottom=0)
    plt.savefig("plots/{0}_{1}_{2}.png".format(tag, cat, var), dpi=1200)
    plt.show()
    plt.clf()

def make_mass_plots(m4l, mA, mA_c, cat, mass=300, year=2018, show=False):
    f, ax = plt.subplots()
    plt.step(m4l.axes[0].edges[:-1], m4l, where='mid',
             color='orange', lw=3, label='$M_{lltt}$')
    plt.step(mA.axes[0].edges[:-1], mA,  where='mid',
             color='mediumseagreen', lw=3, label='$M_{lltt}^{fit}$')
    plt.step(mA_c.axes[0].edges[:-1], mA_c, where='mid',
             color='slateblue', lw=3, label='$M_{lltt}^{constrained}$')
    hep.cms.label(loc=0, year='2018')
    plt.xlabel("Mass [GeV]", ha='right', x=1.0)
    plt.ylabel("Events") #, va='top', ha='left', y=0.8)
    plt.title("{0}, {1} GeV".format(cat, mass))
    ax.legend(loc='upper left', prop={'size' : 20})
    plt.savefig("plots/AZH_{0}_{1}_{2}_masses.png".format(year, mass, cat))
    if (show): plt.show()
    plt.clf()

def make_pyROOT_plot(hist, cat, var, xlabel, ylabel="Events", tag=""):
    canvas = TCanvas("canvas", "canvas")
    

groups = ['signal']
#masses = [220, 240, 260, 280, 300, 320, 340, 350, 400]
masses = [240, 300]
years = [2018]
categories = {1:'eeet', 2:'eemt', 3:'eett', 4:'eeem',
              5:'mmet', 6:'mmmt', 7:'mmtt', 8:'mmem'}

"""for group in groups:
    for year in years:
        for mass in masses:
            hists = get_hists(year, mass, group)
            for cat in categories.values():
                make_mass_plots(hists['m4l'][cat], hists['mA'][cat], 
                                hists['mA_c'][cat], cat, 
                                mass=str(mass), year=str(year))
"""

for group in groups:
    for year in years:
        for mass in masses:
            hists = get_hists(year, mass, group)
            m4l = combine_cats(hists['m4l'])
            mA  = combine_cats(hists['mA'])
            mA_c = combine_cats(hists['mA_c'])
            make_mass_plots(m4l, mA, mA_c, 'all', 
                            mass=str(mass), year=str(year))

"""for group in groups:
    hists = get_hists(group)
    var_list = hists.keys()
    for var in var_list:
        var_hists = hists[var]
        print("...plotting {0}".format(var))
        for cat in categories.values():
            hist = var_hists[cat]
            make_plot(hist, cat, var, labels[var], tag="AZH_2018")
            """         
