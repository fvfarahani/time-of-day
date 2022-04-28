#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 12:36:55 2020

Permutation tests for correlations

@author: Farzad
"""
# Necessary to run the following line in your terminal
# before running this code for visulization (pay attention to your freesurfer version)
    # export FREESURFER_HOME=/Applications/freesurfer/7.1.1
    # export SUBJECTS_DIR=$FREESURFER_HOME/subjects
    # source $FREESURFER_HOME/SetUpFreeSurfer.sh
#%%
import os
import numpy as np
import scipy.io as sio

# Loading the measures
Assortativity_list = []
BetweennessCentrality_list = []
CommunityIndex_list = []
DegreeCentrality_list = []
Hierarchy_list = []
NetworkEfficiency_list = []
NodalClustCoeff_list = []
NodalEfficiency_list = []
NodalLocalEfficiency_list = []
NodalShortestPath_list = []
RichClub_list = []
SmallWorld_list = []
Synchronization_list = []

path = '/Users/Farzad/Desktop/TimeOfDay/Measure'
session_list = sorted(os.listdir(path), reverse=True)
session_label = {'Morning':0,'Evening':1}
for session in session_list:
    if session != '.DS_Store':
        measure_list = os.listdir(path + '/' + session)
        for measure in measure_list:
            if measure != '.DS_Store':
                # use iterator as variable name in a loop
                exec(str(measure) + "= sio.loadmat('/Users/Farzad/Desktop/TimeOfDay/Measure/' + session + '/' + measure + '/' + measure + '.mat')")
                for key in list(eval(measure)): # get the value of a variable given its name in a string
                    if key.startswith('_'):
                        del eval(measure)[key]
                    else:
                        eval(measure)[key] = np.squeeze(eval(measure)[key])
                exec(str(measure) + "_list.append(eval(measure))")

##############################################################################
#%% Permutation tests for correlations (non-parametric)
# --------------------------------
#from scipy import stats
from netneurotools import stats as nnstats
from nilearn import datasets # Automatic atlas fetching
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
labels = atlas.labels

# Loading questionnaires's indices
am = []
file_in = open('/Users/Farzad/Desktop/TimeOfDay/Questionnaire/AM.txt', 'r')
for z in file_in.read().split('\n'):
    am.append(float(z))
ess = []
file_in = open('/Users/Farzad/Desktop/TimeOfDay/Questionnaire/ESS.txt', 'r')
for z in file_in.read().split('\n'):
    ess.append(float(z))
pw = []
file_in = open('/Users/Farzad/Desktop/TimeOfDay/Questionnaire/PW.txt', 'r')
for z in file_in.read().split('\n'):
   pw.append(float(z))  

questionnaire = {'AM': np.array(am), 'ESS': np.array(ess), 'PW': np.array(pw)}
questionnaire_list = ['AM', 'ESS', 'PW']

#%% time-consuming part (not necessary for visualization)

for index in questionnaire_list:

    myfile = open('/Users/Farzad/Desktop/TimeOfDay/corrTest_' + index + '.txt', 'w')    

    for session in session_list:
        if session != '.DS_Store':
            label = session_label[session]
            for measure in measure_list:
                if measure != '.DS_Store':
                    myfile.write('---------------------------------------------------------------------------' + "\n")
                    myfile.write('-------------------------' + session + ' :: ' + measure + '-------------------------' + "\n")
                    myfile.write('---------------------------------------------------------------------------' + "\n")
                    for key in list(eval(measure)): 
                        if np.shape(eval(measure)[key]) == (62,): # participants
                            # Return p-value under the null hypothesis
                            corr = nnstats.permtest_pearsonr(eval(measure + '_list')[label][key], questionnaire[index], n_perm=30000) #seed=2222 (set a seed for reproducibility)
                            if corr[1] <= 0.05: # p-value < 5%
                                myfile.write(measure + '_' + key + ' => ' + 'corr:' + "%s" % corr[0] + ', p-value:' + "%s\n" % corr[1])
                        if np.shape(eval(measure)[key]) == (62, 10): # participants * thresholds
                            for thres in range(10):
                                corr = nnstats.permtest_pearsonr(eval(measure + '_list')[label][key][:,thres], questionnaire[index], n_perm=30000)
                                if corr[1] <= 0.05:
                                    myfile.write(measure + '_' + key + '_' + str((thres+1)*5) + '%' + ' => ' + 'corr:' + "%s" % corr[0] + ', p-value:' + "%s\n" % corr[1])
                        if np.shape(eval(measure)[key]) == (62, 200): # participants * ROIs
                           for roi in range(200):
                                  corr = nnstats.permtest_pearsonr(eval(measure + '_list')[label][key][:,roi], questionnaire[index], n_perm=30000)
                                  if corr[1] <= 0.05:
                                        myfile.write(measure + '_' + key + '_' + labels[roi].decode("utf-8") + ' => ' + 'corr:' + "%s" % corr[0] + ', p-value:' + "%s\n" % corr[1])
                        #if np.shape(eval(measure)[key]) == (62, 200, 10): # participants * ROIs * thresholds
                            #for roi in range(200):
                                #for thres in range(10):
                                    #corr = perm_test(eval(measure + '_list')[label][key][:,roi,thres], questionnaire[index], n_perm=30000)
                                    #if corr <= 0.05:
                                        #myfile.write(measure + '_' + key + '_' + labels[roi].decode("utf-8") + '_' + str((thres+1)*5) + '%' + ' => ' + 'corr:' + "%s" % corr[0] + ', p-value:' + "%s\n" % corr[1])
                
myfile.close()  

# standard parametric p-value:
# print(stats.pearsonr(x, y))

#%% Plot
import os
import numpy as np
import nibabel as nib
from surfer import Brain
from mayavi import mlab

subject_id = "fsaverage"
hemi = "rh"
surf = "inflated" #inflated, orig, pial, white, 

# Bring up the visualization.
brain = Brain(subject_id, hemi, surf, background="white")

# Read in the automatic parcellation of sulci and gyri.
aparc_file = os.path.join(os.environ["SUBJECTS_DIR"],
                          subject_id, "label",
                          hemi + ".Schaefer2018_200Parcels_7Networks_order.annot")
labels, ctab, names = nib.freesurfer.read_annot(aparc_file)

# Make a random vector of scalar data corresponding to a value for each region in the parcellation.
#rs = np.random.RandomState(4)
#roi_data = rs.uniform(.3, .8, size=len(names))

# Set up a correlation vector (measure * Q_index) corresponding to a value for each region in the parcellation.
label = 0 # 0: Morning, 1: Evening
measure = 'DegreeCentrality'
key = 'aDc' # Index inside the measure 
index = 'ESS' # Questionnaire index {AM, ESS, PW}

corr = nnstats.permtest_pearsonr(eval(measure + '_list')[label][key], questionnaire[index], n_perm=30000) 
# Exctract significant correlations # [0]:corr, [1]:pvalue
for i, pvalue in enumerate(corr[1]):
    if pvalue >= 10000: # default 0.1
        corr[0][i] = 0       

corr_l = corr[0][0:100] 
corr_r = corr[0][100:200]
if hemi == 'lh':
    roi_data = np.append(0, corr_l)
if hemi == 'rh':
    roi_data = np.append(0, corr_r)    
    
# Make a vector containing the data point at each vertex.
vtx_data = roi_data[labels]

# Handle vertices that are not defined in the annotation.
vtx_data[labels == -1] = -1

# Display these values on the brain. Use a sequential colormap (assuming
# these data move from low to high values), and add an alpha channel so the
# underlying anatomy is visible.
brain.add_data(vtx_data, -0.5, 0.5, thresh=min(roi_data), colormap="seismic_r", alpha=.8) # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

# Show the different brain views¶
#brain.show_view('lateral') # {'lateral', 'm', 'rostral', 'caudal', 've', 'frontal', 'par', 'dor'}

os.chdir('/Users/Farzad/Desktop/TimeOfDay/Figures/CorrelationsBrain/')
brain.save_imageset(subject_id, ['med', 'lat', 've', 'dor'], 'pdf')

# Save plot
#mlab.test_plot3d()
#mlab.savefig('/Users/Farzad/Desktop/TimeOfDay/correlationPlot.pdf')

#%% Plot (Global)

import numpy as np
import pandas as pd
np.random.seed(0)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')


d1 = {'Session': 'Morning', 
      'Small-worldness': SmallWorld_list[0]['Sigma'][:,6], 
      'Modularity': CommunityIndex_list[0]['Q'][:,6], 
      'Path Length1': SmallWorld_list[0]['Lp'][:,5], 
      'Path Length2': SmallWorld_list[0]['Lp'][:,4], 
      'Global Efficiency1': NetworkEfficiency_list[0]['Eg'][:,5], 
      'Global Efficiency2': NetworkEfficiency_list[0]['Eg'][:,4], 
      'Assortativity1': Assortativity_list[0]['r'][:,8], 
      'Assortativity2': Assortativity_list[0]['rzscore'][:,8], 
      'AM': questionnaire['AM'], 'ESS': questionnaire['ESS']}
d2 = {'Session': 'Evening', 
      'Small-worldness': SmallWorld_list[1]['Sigma'][:,6], 
      'Modularity': CommunityIndex_list[1]['Q'][:,6], 
      'Path Length1': SmallWorld_list[1]['Lp'][:,5], 
      'Path Length2': SmallWorld_list[1]['Lp'][:,4], 
      'Global Efficiency1': NetworkEfficiency_list[1]['Eg'][:,5], 
      'Global Efficiency2': NetworkEfficiency_list[1]['Eg'][:,4], 
      'Assortativity1': Assortativity_list[1]['r'][:,8], 
      'Assortativity2': Assortativity_list[1]['rzscore'][:,8], 
      'AM': questionnaire['AM'], 'ESS': questionnaire['ESS']}
df1 = pd.DataFrame(data=d1)
df2 = pd.DataFrame(data=d2)
dt = df1.append(df2, ignore_index = True) 

sns.lmplot(x="Small-worldness", y="AM", data=dt, hue='Session', palette=dict(Morning="y", Evening="b"), legend = False, scatter_kws={"s": 50}) # 'lw':5
#plt.legend(bbox_to_anchor=(1, 0.9), loc='lower right', fontsize=10, frameon=True)
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/TimeOfDay/ScatterPlot1.pdf') 

sns.lmplot(x="Modularity", y="AM", data=dt, hue='Session', palette=dict(Morning="y", Evening="b"), legend = False, scatter_kws={"s": 50}) # 'lw':5
#plt.legend(bbox_to_anchor=(1, 0.9), loc='lower right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/TimeOfDay/ScatterPlot2.pdf') 

sns.lmplot(x="Path Length1", y="AM", data=dt, hue='Session', palette=dict(Morning="y", Evening="b"), legend = False, scatter_kws={"s": 50}) # 'lw':5
#plt.legend(bbox_to_anchor=(1, 0.9), loc='lower right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/TimeOfDay/ScatterPlot3.pdf') 

sns.lmplot(x="Global Efficiency1", y="AM", data=dt, hue='Session', palette=dict(Morning="y", Evening="b"), legend = False, scatter_kws={"s": 50}) # 'lw':5
#plt.legend(bbox_to_anchor=(1, 0.9), loc='lower right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/TimeOfDay/ScatterPlot4.pdf') 

sns.lmplot(x="Assortativity1", y="AM", data=dt, hue='Session', palette=dict(Morning="y", Evening="b"), legend = False, scatter_kws={"s": 50}) # 'lw':5
#plt.legend(bbox_to_anchor=(1, 0.9), loc='lower right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/TimeOfDay/ScatterPlot5.pdf') 

sns.lmplot(x="Path Length2", y="ESS", data=dt, hue='Session', palette=dict(Morning="y", Evening="b"), legend = False, scatter_kws={"s": 50}) # 'lw':5
#plt.legend(bbox_to_anchor=(1, 0.9), loc='lower right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/TimeOfDay/ScatterPlot6.pdf') 

sns.lmplot(x="Global Efficiency2", y="ESS", data=dt, hue='Session', palette=dict(Morning="y", Evening="b"), legend = False, scatter_kws={"s": 50}) # 'lw':5
#plt.legend(bbox_to_anchor=(1, 0.9), loc='lower right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/TimeOfDay/ScatterPlot7.pdf') 

sns.lmplot(x="Assortativity2", y="ESS", data=dt, hue='Session', palette=dict(Morning="y", Evening="b"), legend = False, scatter_kws={"s": 50}) # 'lw':5
#plt.legend(bbox_to_anchor=(1, 0.9), loc='lower right', fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig('/Users/Farzad/Desktop/TimeOfDay/ScatterPlot8.pdf') 

