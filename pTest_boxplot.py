"""
Permutation Test
An implementation of a permutation test for hypothesis testing 
-- testing the null hypothesis that two different groups come from the same distribution.
============================================

Permutation tests (also called exact tests, randomization tests, or re-randomization tests) 
are nonparametric test procedures to test the null hypothesis that two different 
groups come from the same distribution. A permutation test can be used for significance 
or hypothesis testing (including A/B testing) without requiring to make any assumptions 
about the sampling distribution (e.g., it doesn't require the samples to be normal distributed). 

Moreover, a two-sampled t-test can help you decide if the mean of two normal
distributions are significantly different from each other. What about the 
median, mode, kurtosis, skewness, etc. of two distributions?       
https://towardsdatascience.com/how-to-assess-statistical-significance-in-your-data-with-permutation-tests-8bb925b2113d                 

"""

#%%
import os
import numpy as np
import scipy.io as sio
import statsmodels.stats.multitest as smm
import timeit

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
RichClub_list = []
SmallWorld_list = []
Synchronization_list = []

factor = 'time' # 'time' or 'type'

path = '/Volumes/Elements/TimeOfDay/Measures_' + factor
session_list = sorted(os.listdir(path), reverse=True)
for session in session_list:
    if session != '.DS_Store' and session != '._.DS_Store':
        measure_list = os.listdir(path + '/' + session)
        for measure in measure_list:
            if measure != '.DS_Store' and measure != '._.DS_Store':
                # use iterator as variable name in a loop
                exec(str(measure) + "= sio.loadmat('/Volumes/Elements/TimeOfDay/Measures_' + factor + '/' + session + '/' + measure + '/' + measure + '.mat')")
                for key in list(eval(measure)): # get the value of a variable given its name in a string
                    if key.startswith('_'):
                        del eval(measure)[key]
                    else:
                        eval(measure)[key] = np.squeeze(eval(measure)[key])
                exec(str(measure) + "_list.append(eval(measure))")
                
#%% Implementation of permutation test using monte-carlo method:
# Given the monte-carlo nature, you will not get exact same number on each run.

def perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

start = timeit.default_timer()

perm = 5000
p = 0.1
q = 0.1 # false discovery rate

from nilearn import datasets # Automatic atlas fetching
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
labels = atlas.labels

"""
`b`, `bonferroni` : one-step correction
`s`, `sidak` : one-step correction
`hs`, `holm-sidak` : step down method using Sidak adjustments
`h`, `holm` : step-down method using Bonferroni adjustments
`sh`, `simes-hochberg` : step-up method  (independent)
`hommel` : closed method based on Simes tests (non-negative)
`fdr_i`, `fdr_bh` : Benjamini/Hochberg  (non-negative)
`fdr_n`, `fdr_by` : Benjamini/Yekutieli (negative)
'fdr_tsbh' : two stage fdr correction (Benjamini/Hochberg)
'fdr_tsbky' : two stage fdr correction (Benjamini/Krieger/Yekutieli)
'fdr_gbs' : adaptive step-down fdr correction (Gavrilov, Benjamini, Sarkar)

"""  

myfile = open('/Users/Farzad/Desktop/TimeOfDay/'+factor+'_result.txt', 'w')    

for measure in measure_list:
    if measure != '.DS_Store':
        myfile.write('-----------------------------------------------------------------' + "\n")
        myfile.write('-------------------------' + measure + '-------------------------' + "\n")
        myfile.write('-----------------------------------------------------------------' + "\n")
        for key in list(eval(measure)): 
            if np.shape(eval(measure)[key]) == (62,): # participants
                # Return p-value under the null hypothesis
                pval = perm_test(eval(measure + '_list')[0][key], eval(measure + '_list')[1][key], perm)
                if pval <= p: # p-value < 10%
                    myfile.write(measure + '_' + key + ' = ' + "%0.4f\n" % pval)
            if np.shape(eval(measure)[key]) == (62, 10): # participants * thresholds
                for thld in range(10):
                    pval = perm_test(eval(measure + '_list')[0][key][:,thld], eval(measure + '_list')[1][key][:,thld], perm)
                    if pval <= p:
                        myfile.write(measure + '_' + key + '_' + str((thld+1)*5) + '%' + ' = ' + "%0.4f\n" % pval)
            if np.shape(eval(measure)[key]) == (62, 200): # participants * ROIs
                pval = np.zeros((200,)) 
                for roi in range(200):
                    pval[roi] = perm_test(eval(measure + '_list')[0][key][:,roi], eval(measure + '_list')[1][key][:,roi], perm)                   
                reject, pvals_corr, alphacSidak, alphacBonf = smm.multipletests(pval, q, method='fdr_bh')  
                for roi in range(200):
                    if pvals_corr[roi] <= q:
                        myfile.write(measure + '_' + key + '_' + labels[roi].decode("utf-8") + ' = ' + "%0.4f\n" % pvals_corr[roi])
            if np.shape(eval(measure)[key]) == (62, 200, 10): # participants * ROIs * thresholds
                pval = np.zeros((200,10)) 
                for thld in range(10):
                    for roi in range(200):
                        pval[roi,thld] = perm_test(eval(measure + '_list')[0][key][:,roi,thld], eval(measure + '_list')[1][key][:,roi,thld], perm)
                    reject, pvals_corr, alphacSidak, alphacBonf = smm.multipletests(pval[:,thld], q, method='fdr_bh')      
                    for roi in range(200):
                        if pvals_corr[roi] <= q:
                            myfile.write(measure + '_' + key + '_' + labels[roi].decode("utf-8") + '_' + str((thld+1)*5) + '%' + ' = ' + "%0.4f\n" % pvals_corr[roi])
                
myfile.close()  

# calculate program run time
stop = timeit.default_timer()
print('Run time:', stop - start)            
                
#%% BOXPLOT: Display every observations over the boxplot
# Note that violin plots can be an alternative if you have many many observations. 
# Another way to plot boxplot is via: seaborn -> https://seaborn.pydata.org/generated/seaborn.boxplot.html  or https://cmdlinetips.com/2019/03/how-to-make-grouped-boxplots-in-python-with-seaborn/     

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string

datasetA1 = SmallWorld_list[1]['Sigma']
datasetA2 = SmallWorld_list[0]['Sigma']
datasetB1 = Assortativity_list[0]['r']
datasetB2 = Assortativity_list[1]['r']
datasetC1 = Synchronization_list[0]['s']
datasetC2 = Synchronization_list[1]['s']

ticks = ['5%', '10%', '15%', '20%', '25%', '30%', '35%', '40%', '45%', '50%']

dfA1 = pd.DataFrame(datasetA1, columns=ticks)
dfA2 = pd.DataFrame(datasetA2, columns=ticks)
dfB1 = pd.DataFrame(datasetB1, columns=ticks)
dfB2 = pd.DataFrame(datasetB2, columns=ticks)
dfC1 = pd.DataFrame(datasetC1, columns=ticks)
dfC2 = pd.DataFrame(datasetC2, columns=ticks)

names = []
valsA1, xsA1, valsA2, xsA2 = [],[], [],[]
valsB1, xsB1, valsB2, xsB2 = [],[], [],[]
valsC1, xsC1, valsC2, xsC2 = [],[], [],[]

for i, col in enumerate(dfA1.columns):
    valsA1.append(dfA1[col].values)
    valsA2.append(dfA2[col].values)
    valsB1.append(dfB1[col].values)
    valsB2.append(dfB2[col].values)
    valsC1.append(dfC1[col].values)
    valsC2.append(dfC2[col].values)
    names.append(col)
    # Add some random "jitter" to the data points
    xsA1.append(np.random.normal(i*3-0.5, 0.07, dfA1[col].values.shape[0]))
    xsA2.append(np.random.normal(i*3+0.5, 0.07, dfA2[col].values.shape[0]))
    xsB1.append(np.random.normal(i*3-0.5, 0.07, dfB1[col].values.shape[0]))
    xsB2.append(np.random.normal(i*3+0.5, 0.07, dfB2[col].values.shape[0]))
    xsC1.append(np.random.normal(i*3-0.5, 0.07, dfC1[col].values.shape[0]))
    xsC2.append(np.random.normal(i*3+0.5, 0.07, dfC2[col].values.shape[0]))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=False, sharey=False, figsize=(15, 15))

#f = plt.figure(figsize=(15,5))

bpA1 = ax1.boxplot(valsA1, labels=names, positions=np.array(range(len(datasetA1[0])))*3-0.5, sym='', widths=0.7)
bpA2 = ax1.boxplot(valsA2, labels=names, positions=np.array(range(len(datasetA2[0])))*3+0.5, sym='', widths=0.7)
bpB1 = ax2.boxplot(valsB1, labels=names, positions=np.array(range(len(datasetB1[0])))*3-0.5, sym='', widths=0.7)
bpB2 = ax2.boxplot(valsB2, labels=names, positions=np.array(range(len(datasetB2[0])))*3+0.5, sym='', widths=0.7)
bpC1 = ax3.boxplot(valsC1, labels=names, positions=np.array(range(len(datasetC1[0])))*3-0.5, sym='', widths=0.7)
bpC2 = ax3.boxplot(valsC2, labels=names, positions=np.array(range(len(datasetC2[0])))*3+0.5, sym='', widths=0.7)
# Optional: change the color of 'boxes', 'whiskers', 'caps', 'medians', and 'fliers'
plt.setp(bpA1['medians'], color='r') # or color='#D7191C' ...
plt.setp(bpA2['medians'], linewidth=1, linestyle='-', color='r')
plt.setp(bpB1['medians'], color='r')
plt.setp(bpB2['medians'], linewidth=1, linestyle='-', color='r')
plt.setp(bpC1['medians'], color='r')
plt.setp(bpC2['medians'], linewidth=1, linestyle='-', color='r')

palette = ['r', 'g', 'b', 'y', 'm', 'c', 'k', 'tan', 'orchid', 'cyan', 'gold', 'crimson']

for xA1, xA2, valA1, valA2, c in zip(xsA1, xsA2, valsA1, valsA2, palette):
    ax1.scatter(xA1, valA1, alpha=0.4, color='y') # plt.plot(xA1, valA1, 'r.', alpha=0.4)
    ax1.scatter(xA2, valA2, alpha=0.4, color='b')
    
for xB1, xB2, valB1, valB2, c in zip(xsB1, xsB2, valsB1, valsB2, palette):
    ax2.scatter(xB1, valB1, alpha=0.4, color='y')
    ax2.scatter(xB2, valB2, alpha=0.4, color='b')   
    
for xC1, xC2, valC1, valC2, c in zip(xsC1, xsC2, valsC1, valsC2, palette):
    ax3.scatter(xC1, valC1, alpha=0.4, color='y')
    ax3.scatter(xC2, valC2, alpha=0.4, color='b') 

# Use the pyplot interface to customize any subplot...
# First subplot
plt.sca(ax1)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Small-worldness", fontweight='normal', fontsize=16)
plt.xlabel("Sparsity", fontweight='normal', fontsize=16)
plt.plot([], c='y', label='Morning Session', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='b', label='Evening Session', marker='o', linestyle='None', markersize=8)
# Statistical annotation
xs1 = np.array([-0.5, 2.5])
xs2 = np.array([0.5, 3.5])
for x1, x2 in zip(xs1, xs2):  # e.g., column 25%
    y, h, col = max(datasetA1[:,int((x1+x2)/6)].max(), datasetA2[:,int((x1+x2)/6)].max()) + 0.4, 0.12, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=14)      
# Create empty plot with blank marker containing the extra label
plt.text(20.81, 5.18, "*", ha='center', va='bottom', color=col, size=14, zorder=10) 
plt.plot([], [], " ", label='Significant Mean ($P\leq 0.05$)', color='black')    
plt.legend(prop={'size':16})
    
# Second subplot
plt.sca(ax2)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Assortativity", fontweight='normal', fontsize=16)
plt.xlabel("Sparsity", fontweight='normal', fontsize=16)
plt.plot([], c='y', label='Morning Session', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='b', label='Evening Session', marker='o', linestyle='None', markersize=8)
# Statistical annotation
xs1 = np.array([11.5, 14.5, 17.5, 20.5, 23.5, 26.5])
xs2 = np.array([12.5, 15.5, 18.5, 21.5, 24.5, 27.5])
for x1, x2 in zip(xs1, xs2):  # e.g., column 25%
    y, h, col = max(datasetB1[:,int((x1+x2)/6)].max(), datasetB2[:,int((x1+x2)/6)].max()) + 0.05, 0.015, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=14)
#plt.legend(prop={'size':14}, loc="lower left")

# Third subplot
plt.sca(ax3)
plt.xticks(range(0, len(ticks) * 3, 3), ticks)
plt.xlim(-1.5, len(ticks)*3-1.5)
plt.ylabel("Synchronization", fontweight='normal', fontsize=16)
plt.xlabel("Sparsity", fontweight='normal', fontsize=16)
plt.plot([], c='y', label='Morning Session', marker='o', linestyle='None', markersize=8) # e.g. of other colors, '#2C7BB6' https://htmlcolorcodes.com/ 
plt.plot([], c='b', label='Evening Session', marker='o', linestyle='None', markersize=8)
# Statistical annotation
xs1 = np.array([14.5, 17.5, 20.5, 23.5, 26.5])
xs2 = np.array([15.5, 18.5, 21.5, 24.5, 27.5])
for x1, x2 in zip(xs1, xs2):  # e.g., column 25%
    y, h, col = max(datasetC1[:,int((x1+x2)/6)].max(), datasetC2[:,int((x1+x2)/6)].max()) + 0.025, 0.00775, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col, size=14)  
#plt.legend(prop={'size':14})

# Unified legend  
#handles, labels = ax2.get_legend_handles_labels()
#fig.legend(handles, labels, loc='upper right')

# Annotate Subplots in a Figure with A, B, C 
for n, ax in enumerate((ax1, ax2, ax3)):
    ax.text(-0.05, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=18, weight='bold')

# If needed to add y-axis (threshold value or chance classification accuracy)
#plt.axhline(y=0.35, color='#ff3300', linestyle='--', linewidth=1, label='Threshold Value')
#plt.legend()

sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)

# Adjust the layout of the plot
plt.tight_layout()

plt.savefig('/Users/Farzad/Desktop/TimeOfDay/Boxplot.pdf') 

plt.show() 

#%% shaded ERROR BAR
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import seaborn as sns
import string

mean_A1 = np.mean(DegreeCentrality_list[0]['aDc'], axis=0)
std_A1 = np.std(DegreeCentrality_list[0]['aDc'], axis=0)
mean_A2 = np.mean(DegreeCentrality_list[1]['aDc'], axis=0)
std_A2 = np.std(DegreeCentrality_list[1]['aDc'], axis=0)

mean_B1 = np.mean(BetweennessCentrality_list[0]['aBc'], axis=0)
std_B1 = np.std(BetweennessCentrality_list[0]['aBc'], axis=0)
mean_B2 = np.mean(BetweennessCentrality_list[1]['aBc'], axis=0)
std_B2 = np.std(BetweennessCentrality_list[1]['aBc'], axis=0)

mean_C1 = np.mean(NodalClustCoeff_list[0]['aNCp'], axis=0)
std_C1 = np.std(NodalClustCoeff_list[0]['aNCp'], axis=0)
mean_C2 = np.mean(NodalClustCoeff_list[1]['aNCp'], axis=0)
std_C2 = np.std(NodalClustCoeff_list[1]['aNCp'], axis=0)

mean_D1 = np.mean(NodalEfficiency_list[0]['aNe'], axis=0)
std_D1 = np.std(NodalEfficiency_list[0]['aNe'], axis=0)
mean_D2 = np.mean(NodalEfficiency_list[1]['aNe'], axis=0)
std_D2 = np.std(NodalEfficiency_list[1]['aNe'], axis=0)

x = np.arange(len(mean_A1))

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1, sharex=False, sharey=False, figsize=(20, 20))

#g1 = 'Morning Session'; g2 = 'Evening Session'; c1 = 'y'; c2 = 'b'
g1 = 'Lark'; g2 = 'Owl'; c1 = 'cyan'; c2 = 'tomato'

plt.sca(ax1)
ebA1 = ax1.plot(x, mean_A1, '-ko', label=g1, markerfacecolor=c1)
ax1.fill_between(x, mean_A1 - std_A1, mean_A1 + std_A1, color=c1, alpha=0.3)
ebA2 = ax1.plot(x, mean_A2, '-ko', label=g2, markerfacecolor=c2)
ax1.fill_between(x, mean_A2 - std_A2, mean_A2 + std_A2, color=c2, alpha=0.3)
plt.ylabel("Degree Centrality", fontweight='normal', fontsize=16)
plt.axvline(x=99.5, color='k', linestyle='-', linewidth=1.5)
"""# time
plt.axvline(x=15-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=30-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=46-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=51-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=117-1, color='r', linestyle='--', linewidth=1.5)
"""# type
plt.axvline(x=48-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=90-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=180-1, color='r', linestyle='--', linewidth=1.5)

plt.xlim([-1, 200])
y_min, y_max = ax1.get_ylim()
h = (y_max-y_min)/15; i = y_min-h # intercept
# ticks and labels along the bottom edge are off
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
# Add rectangle objects as tick labels for better visualization
ax1.add_patch(patches.Rectangle((0, i), width=13.5-0, height=h, facecolor='purple', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((13.5, i), width=29.5-13.5, height=h, facecolor='blue', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((29.5, i), width=42.5-29.5, height=h, facecolor='green', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((42.5, i), width=53.5-42.5, height=h, facecolor='violet', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((53.5, i), width=59.5-53.5, height=h, facecolor='yellow', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((59.5, i), width=72.5-59.5, height=h, facecolor='orange', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((72.5, i), width=99.5-72.5, height=h, facecolor='red', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((99.5, i), width=114.5-99.5, height=h, facecolor='purple', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((114.5, i), width=133.5-114.5, height=h, facecolor='blue', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((133.5, i), width=146.5-133.5, height=h, facecolor='green', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((146.5, i), width=157.5-146.5, height=h, facecolor='violet', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((157.5, i), width=163.5-157.5, height=h, facecolor='yellow', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((163.5, i), width=180.5-163.5, height=h, facecolor='orange', clip_on=False, linewidth = 0))
ax1.add_patch(patches.Rectangle((180.5, i), width=199-180.5, height=h, facecolor='red', clip_on=False, linewidth = 0))

plt.sca(ax2)
ebB1 = ax2.plot(x, mean_B1, '-ko', label=g1, markerfacecolor=c1)
ax2.fill_between(x, mean_B1 - std_B1, mean_B1 + std_B1, color=c1, alpha=0.3)
ebB2 = ax2.plot(x, mean_B2, '-ko', label=g2, markerfacecolor=c2)
ax2.fill_between(x, mean_B2 - std_B2, mean_B2 + std_B2, color=c2, alpha=0.3)
plt.ylabel("Betweenness Centrality", fontweight='normal', fontsize=16)
plt.axvline(x=99.5, color='k', linestyle='-', linewidth=1.5)
"""# time
plt.axvline(x=99-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=138-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=150-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=200-1, color='r', linestyle='--', linewidth=1.5)
"""# type
plt.axvline(x=79-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=90-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=126-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=145-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=180-1, color='r', linestyle='--', linewidth=1.5)

plt.xlim([-1, 200])
y_min, y_max = ax2.get_ylim()
h = (y_max-y_min)/15; i = y_min-h
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax2.add_patch(patches.Rectangle((0, i), width=13.5-0, height=h, facecolor='purple', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((13.5, i), width=29.5-13.5, height=h, facecolor='blue', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((29.5, i), width=42.5-29.5, height=h, facecolor='green', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((42.5, i), width=53.5-42.5, height=h, facecolor='violet', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((53.5, i), width=59.5-53.5, height=h, facecolor='yellow', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((59.5, i), width=72.5-59.5, height=h, facecolor='orange', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((72.5, i), width=99.5-72.5, height=h, facecolor='red', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((99.5, i), width=114.5-99.5, height=h, facecolor='purple', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((114.5, i), width=133.5-114.5, height=h, facecolor='blue', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((133.5, i), width=146.5-133.5, height=h, facecolor='green', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((146.5, i), width=157.5-146.5, height=h, facecolor='violet', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((157.5, i), width=163.5-157.5, height=h, facecolor='yellow', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((163.5, i), width=180.5-163.5, height=h, facecolor='orange', clip_on=False, linewidth = 0))
ax2.add_patch(patches.Rectangle((180.5, i), width=199-180.5, height=h, facecolor='red', clip_on=False, linewidth = 0))

plt.sca(ax3)
ebC1 = ax3.plot(x, mean_C1, '-ko', label=g1, markerfacecolor=c1)
ax3.fill_between(x, mean_C1 - std_C1, mean_C1 + std_C1, color=c1, alpha=0.3)
ebC2 = ax3.plot(x, mean_C2, '-ko', label=g2, markerfacecolor=c2)
ax3.fill_between(x, mean_C2 - std_C2, mean_C2 + std_C2, color=c2, alpha=0.3)
plt.ylabel("Nodal Clustering Coefficient", fontweight='normal', fontsize=16)
plt.axvline(x=99.5, color='k', linestyle='-', linewidth=1.5)
"""# time
plt.axvline(x=94-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=117-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=138-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=187-1, color='r', linestyle='--', linewidth=1.5)
"""# type
plt.axvline(x=42-1, color='r', linestyle='--', linewidth=1.5)

plt.xlim([-1, 200])
y_min, y_max = ax3.get_ylim()
h = (y_max-y_min)/15; i = y_min-h
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax3.add_patch(patches.Rectangle((0, i), width=13.5-0, height=h, facecolor='purple', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((13.5, i), width=29.5-13.5, height=h, facecolor='blue', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((29.5, i), width=42.5-29.5, height=h, facecolor='green', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((42.5, i), width=53.5-42.5, height=h, facecolor='violet', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((53.5, i), width=59.5-53.5, height=h, facecolor='yellow', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((59.5, i), width=72.5-59.5, height=h, facecolor='orange', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((72.5, i), width=99.5-72.5, height=h, facecolor='red', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((99.5, i), width=114.5-99.5, height=h, facecolor='purple', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((114.5, i), width=133.5-114.5, height=h, facecolor='blue', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((133.5, i), width=146.5-133.5, height=h, facecolor='green', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((146.5, i), width=157.5-146.5, height=h, facecolor='violet', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((157.5, i), width=163.5-157.5, height=h, facecolor='yellow', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((163.5, i), width=180.5-163.5, height=h, facecolor='orange', clip_on=False, linewidth = 0))
ax3.add_patch(patches.Rectangle((180.5, i), width=199-180.5, height=h, facecolor='red', clip_on=False, linewidth = 0))

plt.sca(ax4)
ebD1 = ax4.plot(x, mean_D1, '-ko', label=g1, markerfacecolor=c1)
ax4.fill_between(x, mean_D1 - std_D1, mean_D1 + std_D1, color=c1, alpha=0.3)
ebD2 = ax4.plot(x, mean_D2, '-ko', label=g2, markerfacecolor=c2)
ax4.fill_between(x, mean_D2 - std_D2, mean_D2 + std_D2, color=c2, alpha=0.3)
plt.ylabel("Nodal Efficiency", fontweight='normal', fontsize=16)
plt.axvline(x=99.5, color='k', linestyle='-', linewidth=1.5, label='L-R Separator')
"""# time
plt.axvline(x=15-1, color='r', linestyle='--', linewidth=1.5, label='Significant Variation')
plt.axvline(x=30-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=51-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=63-1, color='r', linestyle='--', linewidth=1.5)
"""# type
plt.axvline(x=42-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=48-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=90-1, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=180-1, color='r', linestyle='--', linewidth=1.5)

plt.xlim([-1, 200])
y_min, y_max = ax4.get_ylim()
h = (y_max-y_min)/15; i = y_min-h
plt.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
ax4.add_patch(patches.Rectangle((0, i), width=13.5-0, height=h, facecolor='purple', clip_on=False, linewidth = 0, label='Visual'))
ax4.add_patch(patches.Rectangle((13.5, i), width=29.5-13.5, height=h, facecolor='blue', clip_on=False, linewidth = 0, label='Somatomotor'))
ax4.add_patch(patches.Rectangle((29.5, i), width=42.5-29.5, height=h, facecolor='green', clip_on=False, linewidth = 0, label='Dorsal Attention'))
ax4.add_patch(patches.Rectangle((42.5, i), width=53.5-42.5, height=h, facecolor='violet', clip_on=False, linewidth = 0, label='Ventral Attention'))
ax4.add_patch(patches.Rectangle((53.5, i), width=59.5-53.5, height=h, facecolor='yellow', clip_on=False, linewidth = 0, label='Limbic'))
ax4.add_patch(patches.Rectangle((59.5, i), width=72.5-59.5, height=h, facecolor='orange', clip_on=False, linewidth = 0, label='Frontoparietal'))
ax4.add_patch(patches.Rectangle((72.5, i), width=99.5-72.5, height=h, facecolor='red', clip_on=False, linewidth = 0, label='Default'))
ax4.add_patch(patches.Rectangle((99.5, i), width=114.5-99.5, height=h, facecolor='purple', clip_on=False, linewidth = 0))
ax4.add_patch(patches.Rectangle((114.5, i), width=133.5-114.5, height=h, facecolor='blue', clip_on=False, linewidth = 0))
ax4.add_patch(patches.Rectangle((133.5, i), width=146.5-133.5, height=h, facecolor='green', clip_on=False, linewidth = 0))
ax4.add_patch(patches.Rectangle((146.5, i), width=157.5-146.5, height=h, facecolor='violet', clip_on=False, linewidth = 0))
ax4.add_patch(patches.Rectangle((157.5, i), width=163.5-157.5, height=h, facecolor='yellow', clip_on=False, linewidth = 0))
ax4.add_patch(patches.Rectangle((163.5, i), width=180.5-163.5, height=h, facecolor='orange', clip_on=False, linewidth = 0))
ax4.add_patch(patches.Rectangle((180.5, i), width=199-180.5, height=h, facecolor='red', clip_on=False, linewidth = 0))

plt.legend(prop={'size':16}, ncol=6, frameon=False, bbox_to_anchor=(.48, -.06), loc='upper center')

# Annotate Subplots in a Figure with A, B, C, D (as well as L & R)
for n, ax in enumerate((ax1, ax2, ax3, ax4)):
    ax.text(-0.04, 1.05, string.ascii_uppercase[n], transform=ax.transAxes, 
            size=18, weight='bold')
    ax.text(0.258, 1.015, 'L', transform=ax.transAxes, 
            size=14, weight='regular')
    ax.text(0.731, 1.015, 'R', transform=ax.transAxes, 
            size=14, weight='regular')

sns.despine(right=True) # removes right and top axis lines (top, bottom, right, left)

# Adjust the layout of the plot
plt.tight_layout()

plt.savefig('/Users/Farzad/Desktop/TimeOfDay/ShadedErrorbar.pdf') 

plt.show()

