import pickle,joblib
import matplotlib.pyplot as plt
import argparse
from genericpath import exists
import uproot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import awkward as ak
from pathlib import Path

from uncertainties import ufloat, unumpy
import atlas_mpl_style as ampl


with open('nominal/MC_merged_hist.pkl', 'rb') as f:
    obj = joblib.load(f)
#print(obj)
a = obj.keys()
#a = obj['event_weight'].keys()
#a = obj['event_weight']['500_LeadingJet_Forward_Gluon_jet_nTracks'].values()
#lent = len(obj['event_weight']['500_LeadingJet_Forward_Gluon_jet_pt'].values())
print(a)
#plt.hist(obj['event_weight']['500_LeadingJet_Forward_Gluon_jet_nTracks'])
#plt.show()
