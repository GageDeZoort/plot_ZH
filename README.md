# plot_ZH
A set of classes and scripts to perform the ZHtoTauTau and AtoZHtoTauTau analyses. 

## Quickstart
``` 
export SCRAM_ARCH=sl6_amd64_gcc700
cmsrel CMSSW_10_2_14
cd CMSSSW_10_2_14/src
git clone https://github.com/princeton-cms-run2/ZH_Run2.git -b devel
git clone https://github.com/cms-tau-pog/TauIDSFs TauPOG/TauIDSFs
git clone https://github.com/SVfit/ClassicSVfit TauAnalysis/ClassicSVfit -b release_2018Mar20
git clone https://github.com/SVfit/SVfitTF TauAnalysis/SVfitTF
cmsenv
scram b -j 8
cd ZH_Run2
git clone https://github.com/GageDeZoort/plot_ZH.git
source /cvmfs/sft.cern.ch/lcg/views/LCG_92python3/x86_64-slc6-gcc62-opt/setup.sh
```
