"""
Extracting signals from a brain parcellation
============================================

Here we show how to extract signals from a brain parcellation and compute
a correlation matrix.

"""
##############################################################################
#%% 
import os
import numpy as np

# Retrieve the desired atlas
# --------------------------------

from nilearn import datasets

# Automatic atlas fetching
#atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
atlas.region_coords = [(-24,-53,-9), (-26,-77,-14),(-45,-69,-8), (-10,-67,-4), (-27,-95,-12), (-14,-44,-3), (-5,-93,-4), (-47,-70,10), (-23,-97,6), (-11,-70,7), (-40,-85,11), (-12,-73,22), (-7,-87,28), (-23,-87,23), (-51,-4,-2), (-53,-24,9), (-37,-21,16), (-55,-4,10), (-53,-22,18), (-56,-8,31), (-47,-9,46), (-7,-12,46), (-49,-28,57), (-40,-25,57), (-31,-46,63), (-32,-22,64), (-26,-38,68),(-20,-11,68), (-5,-29,67), (-19,-31,68), (-43,-48,-19), (-57,-60,-1), (-26,-70,38), (-54,-27,42), (-41,-35,47), (-33,-49,47),  (-17,-73,54),(-29,-60,59), (-6,-60,57), (-17,-53,68), (-31,-4,53), (-22,6,62), (-48,6,29), (-56,-40,20), (-61,-26,28), (-60,-39,36), (-39,-4,-4), (-33,20,5), (-39,1,11), (-51,9,11), (-28,43,31), (-6,9,41), (-11,-35,46), (-6,-3,65), (-24,22,-20), (-10,35,-21), (-29,-6,-39), (-45,-20,-30), (-28,10,-34), (-43,8,-19), (-53,-51,46), (-35,-62,48), (-45,-42,46), (-61,-43,-13), (-32,42,-13), (-42,49,-6), (-28,58,8), (-42,40,16), (-44,20,27), (-43,6,43), (-9,-73,38), (-5,-29,28), (-3,4,30), (-47,8,-33), (-60,-19,-22), (-56,-6,-12), (-58,-30,-4), (-58,-43,7), (-48,-57,18), (-39,-80,31), (-57,-54,28), (-46,-66,38), (-35,20,-13), (-6,36,-10), (-46,31,-7), (-12,63,-6), (-52,22,8), (-6,44,7), (-8,59,21), (-6,30,25), (-11,47,45), (-3,33,43), (-40,19,49), (-24,25,49), (-9,17,63), (-11,-56,13), (-5,-55,27), (-4,-31,36), (-6,-54,42), (-26,-32,-18), (39,-35,-23), (28,-36,-14), (29,-69,-12), (12,-65,-5), (48,-71,-6), (11,-92,-5), (16,-46,-1), (31,-94,-4), (9,-75,9), (22,-60,7), (42,-80,10), (20,-90,22), (11,-74,26), (16,-85,39), (33,-75,32), (51,-15,5), (64,-23,8), (38,-13,15), (44,-27,18), (59,0,10), (56,-11,14), (58,-5,31), (10,-15,41), (51,-22,52), (47,-11,48), (7,-11,51), (40,-24,57), (32,-40,64), (33,-21,65), (29,-34,65), (22,-9,67), (10,-39,69), (6,-23,69), (20,-29,70), (50,-53,-15), (52,-60,9), (59,-16,34), (46,-38,49), (41,-31,46), (15,-73,53), (34,-48,51), (26,-61,58), (8,-56,61), (21,-48,70), (34,-4,52), (26,7,58), (52,11,21), (57,-45,9), (60,-39,17), (60,-26,27), (51,4,40), (41,6,-15), (46,-4,-4),  (36,24,5), (43,7,4), (7,9,41), (11,-36,47), (8,3,66), (12,39,-22), (28,22,-19), (15,64,-8), (30,9,-38), (47,-12,-35), (25,-11,-32), (62,-37,37), (53,-42,48), (37,-63,47), (63,-41,-12), (34,21,-8), (36,46,-13), (29,58,5), (43,45,10), (46,24,26), (30,48,27), (41,33,37), (42,14,49), (14,-70,37), (5,-24,31), (5,3,30), (7,31,28), (7,25,55), (47,-69,27), (54,-50,28), (51,-59,44), (47,13,-30), (61,-13,-21), (55,-6,-10), (63,-27,-6), (52,-31,2), (51,28,0), (5,37,-14), (8,42,4), (6,29,15), (8,58,18), (15,46,44), (29,30,42), (23,24,53), (12,-55,15), (7,-49,31), (6,-58,44)]

# Loading atlas image stored in 'maps'
atlas_filename = atlas.maps
# Loading atlas data stored in 'labels'
labels = atlas.labels
# Setting systems' labels
roi = 200
atlas.systems = [None] * roi
atlas.systems[7] = "Visual (LH)"
atlas.systems[22] = "Somatomotor (LH)"
atlas.systems[37] = "Dorsal Attention (LH)"
atlas.systems[49] = "Ventral Attention (LH)"
atlas.systems[57] = "Limbic (LH)"
atlas.systems[67] = "Frontoparietal (LH)"
atlas.systems[87] = "Default (LH)"
atlas.systems[108] = "Visual (RH)"
atlas.systems[125] = "Somatomotor (RH)"
atlas.systems[141] = "Dorsal Attention (RH)"
atlas.systems[153] = "Ventral Attention (RH)"
atlas.systems[161] = "Limbic (RH)"
atlas.systems[173] = "Frontoparietal (RH)"
atlas.systems[191] = "Default (RH)"
systems = atlas.systems

from nilearn import plotting
plotting.plot_roi(atlas_filename)

print('Atlas ROIs are located in nifti image (4D) at: %s' %atlas_filename)  # 4D data

###############################################################################
#%% Load the functional datasets (morning and evening) and 
# extract signals (timeseries) on a parcellation defined by labels
# then calculating the correlation and binarized matrices
# -----------------------------------------------------
# Using the NiftiLabelsMasker
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', verbose=5)

data_path = '/Volumes/Elements/TimeOfDay/Data/'
data_dir_list = ['Morning', 'Evening']
#data_dir_list = os.listdir(data_path)
labels_name = {'Morning':0,'Evening':1}
ts_list = []
corr_list = []
bin_list = []
label_list = []

density = 0.1
n = round(density*roi*(roi-1)/2)*2 # number of strongest connections * 2 (nodes)
upper, lower = 1, 0

for dataset in data_dir_list:
    fmri_list=sorted(os.listdir(data_path+'/'+ dataset))
    print ('Loading the nifti files of dataset-'+'{}\n'.format(dataset))
    label = labels_name[dataset]
    for fmri in fmri_list:
        # From nifti files to the signal time series in a numpy array.
        if fmri.endswith(".nii"):
            ts = masker.fit_transform(data_path + '/'+ dataset + '/'+ fmri ) # ts = masker.fit_transform(fmri_filenames, confounds=data.confounds)
            corr_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional
            corr_matrix = corr_measure.fit_transform([ts])[0]
            # remove the self-connections (mask the main diagonal)
            np.fill_diagonal(corr_matrix, 0)
            # saving correlations as txt files in FC directory, for MATLAB use (graph theory)
            FC_path = '/Volumes/Elements/TimeOfDay/FC_time/' + dataset + '/'
            np.savetxt(FC_path + fmri + '.txt', corr_matrix, fmt='%.18e', delimiter=' ', newline='\n')
            
            # binarizing the individual correlation matrix (if needed, rarely use)
            abs_matrix = np.absolute(corr_matrix) # absolute value
            flat = abs_matrix.flatten()
            flat.sort()
            threshold = flat[-n]
            bin_matrix = np.where(abs_matrix > threshold, upper, lower)
            
            ts_list.append(ts)
            corr_list.append(corr_matrix)
            bin_list.append(bin_matrix)
            label_list.append(label)
            
fmri_ts = np.array(ts_list)
fmri_corr = np.array(corr_list)
fmri_bin = np.array(bin_list)

import matplotlib.pyplot as plt
plt.plot(np.arange(0, 200), fmri_ts[0,4,:].transpose(), color='red')
plt.show()
           
#%% Calculating the mean binarized matrix for each session
# -----------------------------------------

# Morning session
mean_corr_mor = np.mean(fmri_corr[0:int(len(label_list)/2)], axis=0)
np.save('/Volumes/Elements/TimeOfDay/FC_time/mean_corr_mor', mean_corr_mor)

abs_matrix = np.absolute(mean_corr_mor)
flat = abs_matrix.flatten()
flat.sort()
threshold = flat[-n]
mean_bin_mor = np.where(abs_matrix > threshold, upper, lower)
np.save('/Volumes/Elements/TimeOfDay/FC_time/mean_bin_mor', mean_bin_mor)

# Evening session
mean_corr_eve = np.mean(fmri_corr[int(len(label_list)/2):int(len(label_list))], axis=0)
np.save('/Volumes/Elements/TimeOfDay/FC_time/mean_corr_eve', mean_corr_eve)

abs_matrix = np.absolute(mean_corr_eve)
flat = abs_matrix.flatten()
flat.sort()
threshold = flat[-n]
mean_bin_eve = np.where(abs_matrix > threshold, upper, lower)
np.save('/Volumes/Elements/TimeOfDay/FC_time/mean_bin_eve', mean_bin_eve)

# Saving the files
Bin_path = '/Volumes/Elements/TimeOfDay/FC_time/'
np.savetxt(Bin_path + 'mean_corr_mor.txt', mean_corr_mor, fmt='%.18e', delimiter=' ', newline='\n')
np.savetxt(Bin_path + 'mean_corr_eve.txt', mean_corr_eve, fmt='%.18e', delimiter=' ', newline='\n')
np.savetxt(Bin_path + 'mean_bin_mor_' + str(int(density*100)) + '%.txt', mean_bin_mor, fmt='%.18e', delimiter=' ', newline='\n')
np.savetxt(Bin_path + 'mean_bin_eve_' + str(int(density*100)) + '%.txt', mean_bin_eve, fmt='%.18e', delimiter=' ', newline='\n')

np.savetxt(Bin_path + 'Schaefer_7Networks_200.txt', labels, fmt='%s', delimiter=' ', newline='\n')

#%% finding the hubs
# -----------------------------------------
# based on degree centrality (>=20) of mean corr matrix for density of 5%, 
# the follwing regions are top ~10 percent (hubs)
from nilearn import plotting 
import scipy.io
dc_mor = scipy.io.loadmat('/Volumes/Elements/TimeOfDay/Hubs/mean_degree/mor/DegreeCentrality/DegreeCentrality.mat')['Dc']
dc_eve = scipy.io.loadmat('/Volumes/Elements/TimeOfDay/Hubs/mean_degree/eve/DegreeCentrality/DegreeCentrality.mat')['Dc']
dc_lark = scipy.io.loadmat('/Volumes/Elements/TimeOfDay/Hubs/mean_degree/lark/DegreeCentrality/DegreeCentrality.mat')['Dc']
dc_owl = scipy.io.loadmat('/Volumes/Elements/TimeOfDay/Hubs/mean_degree/owl/DegreeCentrality/DegreeCentrality.mat')['Dc']

color = (['purple']*14) + (['blue']*16) + (['green']*13) + (['violet']*11) + (['cream']*6) + (['orange']*13) + (['red']*27) + (['purple']*15) + (['blue']*19) + (['green']*13) + (['violet']*11) + (['cream']*6) + (['orange']*17) + (['red']*19)

# morning  
sig_mor = np.where(dc_mor >= 20)[1][4]
sig_mor = sorted(np.hstack((sig_mor, 97-1, 177-1, 198-1)))
coords_mor = list(atlas.region_coords[i] for i in sig_mor)
color_mor = list(color[i] for i in sig_mor)
size_mor = list(int(dc_mor[0, i]) for i in sig_mor)
size_mor = [(x - 10)*4 for x in size_mor]

view = plotting.view_markers( 
    coords_mor, color_mor, marker_size=size_mor) 
view.open_in_browser() 
#view.save_as_html("surface_plot.html") 

# evening  
sig_eve = np.where(dc_eve >= 20)[1][4]
sig_eve = sorted(np.hstack((sig_eve, 71-1, 177-1)))              
coords_eve = list(atlas.region_coords[i] for i in sig_eve)
color_eve = list(color[i] for i in sig_eve)
size_eve = list(int(dc_eve[0, i]) for i in sig_eve)
size_eve = [(x - 10)*4 for x in size_eve]

view = plotting.view_markers( 
    coords_eve, color_eve, marker_size=size_eve) 
view.open_in_browser()
#%%

# lark  
sig_lark = np.where(dc_lark >= 20)[1]                            
coords_lark = list(atlas.region_coords[i] for i in sig_lark)
color_lark = list(color[i] for i in sig_lark)
size_lark = list(int(dc_lark[0, i]) for i in sig_lark)
size_lark = [(x - 10)*4 for x in size_lark]

view = plotting.view_markers( 
    coords_lark, color_lark, marker_size=size_lark) 
view.open_in_browser()

# owl 
sig_owl = np.where(dc_owl >= 20)[1]                                           
coords_owl = list(atlas.region_coords[i] for i in sig_owl)
color_owl = list(color[i] for i in sig_owl)
size_owl = list(int(dc_owl[0, i]) for i in sig_owl)
size_owl = [(x - 10)*4 for x in size_owl]

view = plotting.view_markers( 
    coords_owl, color_owl, marker_size=size_owl) 
view.open_in_browser()


#%% Display Functional Connectivity (weighted & binarized)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
os.chdir('/Volumes/Elements/TimeOfDay/Figures/')

# if you want to skip previous steps, run the following lines to read connectomes:
mean_corr_mor = np.load('/Volumes/Elements/TimeOfDay/FC_time/mean_corr_mor.npy')
mean_corr_eve = np.load('/Volumes/Elements/TimeOfDay/FC_time/mean_corr_eve.npy')
mean_bin_mor = np.load('/Volumes/Elements/TimeOfDay/FC_time/mean_bin_mor.npy')
mean_bin_eve = np.load('/Volumes/Elements/TimeOfDay/FC_time/mean_bin_eve.npy')

# general plot settings
split = np.array([-0.5, 13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5, 199.5])
color = ['#A251AC', '#789AC1', '#409832', '#E165FE', '#F6FDC9', '#EFB944', '#D9717D',
         '#A251AC', '#789AC1', '#409832', '#E165FE', '#F6FDC9', '#EFB944', '#D9717D']

# -----------------------------------
# Morning, weighted correlations
f = plt.figure(figsize=(19,15))
plt.matshow(mean_corr_mor, fignum=f.number, vmin = -1, vmax = 1, cmap='twilight_shifted')
plt.title('Morning Session', fontsize=14)
#plt.xticks(range(mean_corr_mor.shape[1]), systems, fontsize=10, rotation=90) #systems or labels
#plt.yticks(range(mean_corr_mor.shape[1]), systems, fontsize=10)
plt.xticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
plt.yticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
cb = plt.colorbar() #cb.ax.tick_params(labelsize=14)
cb.ax.tick_params(labelsize=12)
plt.axvline(x=100-0.5,color='k',linewidth=1.5)
plt.axhline(y=100-0.5,color='k',linewidth=1.5)
# Draw grid lines
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.tick_params(
    axis='both',       # changes apply to the x,y-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,        # ticks along the left edge are off
    right=False,       # ticks along the right edge are off
    labeltop=False,    # labels along the top edge are off
    labelleft=False)   # labels along the left edge are off

# Add rectangle objects as tick labels (X axis)
xmin, xmax, ymin, ymax = plt.axis()
xy = split[:-1] # anchor points
h = (ymax-ymin)/30; space = h/5; i = ymax + space # intercept
w = split[1:] - xy # rectangle width(s)
for j in range(len(xy)): # plot rectangles one-by-one
    plt.gca().add_patch(patches.Rectangle((xy[j], i), width=w[j], height=h, facecolor=color[j], clip_on=False, linewidth=1, edgecolor='k'))

# Add rectangle objects as tick labels (Y axis)
w = (ymax-ymin)/30; i = ymax # intercept
h = split[1:] - xy # rectangle height(s)
for j in range(len(xy)): # plot rectangles one-by-one
    plt.gca().add_patch(patches.Rectangle((i+space, xy[j]), width=w, height=h[j], facecolor=color[j], clip_on=False, linewidth=1, edgecolor='k'))

plt.tight_layout()
plt.savefig('/Volumes/Elements/TimeOfDay/Figures/corr_mor_w.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)

# -----------------------------------
# Morning, binarized correlations
f = plt.figure(figsize=(19,15))
plt.matshow(mean_bin_mor, fignum=f.number, vmin = -1, vmax = 1, cmap='twilight_shifted')
plt.title('Morning Session', fontsize=14)
#plt.xticks(range(mean_corr_mor.shape[1]), systems, fontsize=10, rotation=90) #systems or labels
#plt.yticks(range(mean_corr_mor.shape[1]), systems, fontsize=10)
plt.xticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
plt.yticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
cb = plt.colorbar() #cb.ax.tick_params(labelsize=14)
cb.ax.tick_params(labelsize=12)
plt.axvline(x=100-0.5,color='k',linewidth=1.5)
plt.axhline(y=100-0.5,color='k',linewidth=1.5)
# Draw grid lines
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.tick_params(
    axis='both',       # changes apply to the x,y-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,        # ticks along the left edge are off
    right=False,       # ticks along the right edge are off
    labeltop=False,    # labels along the top edge are off
    labelleft=False)   # labels along the left edge are off

# Add rectangle objects as tick labels (X axis)
xmin, xmax, ymin, ymax = plt.axis()
xy = split[:-1] # anchor points
h = (ymax-ymin)/30; space = h/5; i = ymax + space # intercept
w = split[1:] - xy # rectangle width(s)
for j in range(len(xy)): # plot rectangles one-by-one
    plt.gca().add_patch(patches.Rectangle((xy[j], i), width=w[j], height=h, facecolor=color[j], clip_on=False, linewidth=1, edgecolor='k'))

# Add rectangle objects as tick labels (Y axis)
w = (ymax-ymin)/30; i = ymax # intercept
h = split[1:] - xy # rectangle height(s)
for j in range(len(xy)): # plot rectangles one-by-one
    plt.gca().add_patch(patches.Rectangle((i+space, xy[j]), width=w, height=h[j], facecolor=color[j], clip_on=False, linewidth=1, edgecolor='k'))

plt.tight_layout()
plt.savefig('/Volumes/Elements/TimeOfDay/Figures/corr_mor_b10.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)

# -----------------------------------
# Evening, weighted correlations
f = plt.figure(figsize=(19,15))
plt.matshow(mean_corr_eve, fignum=f.number, vmin = -1, vmax = 1, cmap='twilight_shifted')
plt.title('Evening Session', fontsize=14)
#plt.xticks(range(mean_corr_mor.shape[1]), systems, fontsize=10, rotation=90) #systems or labels
#plt.yticks(range(mean_corr_mor.shape[1]), systems, fontsize=10)
plt.xticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
plt.yticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
cb = plt.colorbar() #cb.ax.tick_params(labelsize=14)
cb.ax.tick_params(labelsize=12)
plt.axvline(x=100-0.5,color='k',linewidth=1.5)
plt.axhline(y=100-0.5,color='k',linewidth=1.5)
# Draw grid lines
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.tick_params(
    axis='both',       # changes apply to the x,y-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,        # ticks along the left edge are off
    right=False,       # ticks along the right edge are off
    labeltop=False,    # labels along the top edge are off
    labelleft=False)   # labels along the left edge are off

# Add rectangle objects as tick labels (X axis)
xmin, xmax, ymin, ymax = plt.axis()
xy = split[:-1] # anchor points
h = (ymax-ymin)/30; space = h/5; i = ymax + space # intercept
w = split[1:] - xy # rectangle width(s)
for j in range(len(xy)): # plot rectangles one-by-one
    plt.gca().add_patch(patches.Rectangle((xy[j], i), width=w[j], height=h, facecolor=color[j], clip_on=False, linewidth=1, edgecolor='k'))

# Add rectangle objects as tick labels (Y axis)
w = (ymax-ymin)/30; i = ymax # intercept
h = split[1:] - xy # rectangle height(s)
for j in range(len(xy)): # plot rectangles one-by-one
    plt.gca().add_patch(patches.Rectangle((i+space, xy[j]), width=w, height=h[j], facecolor=color[j], clip_on=False, linewidth=1, edgecolor='k'))

plt.tight_layout()
plt.savefig('/Volumes/Elements/TimeOfDay/Figures/corr_eve_w.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)

# -----------------------------------
# Evening, binarized correlations
f = plt.figure(figsize=(19,15))
plt.matshow(mean_bin_eve, fignum=f.number, vmin = -1, vmax = 1, cmap='twilight_shifted')
plt.title('Evening Session', fontsize=14)
#plt.xticks(range(mean_corr_mor.shape[1]), systems, fontsize=10, rotation=90) #systems or labels
#plt.yticks(range(mean_corr_mor.shape[1]), systems, fontsize=10)
plt.xticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
plt.yticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
cb = plt.colorbar() #cb.ax.tick_params(labelsize=14)
cb.ax.tick_params(labelsize=12)
plt.axvline(x=100-0.5,color='k',linewidth=1.5)
plt.axhline(y=100-0.5,color='k',linewidth=1.5)
# Draw grid lines
plt.grid(color='black', linestyle='-', linewidth=0.5)
plt.tick_params(
    axis='both',       # changes apply to the x,y-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,        # ticks along the left edge are off
    right=False,       # ticks along the right edge are off
    labeltop=False,    # labels along the top edge are off
    labelleft=False)   # labels along the left edge are off

# Add rectangle objects as tick labels (X axis)
xmin, xmax, ymin, ymax = plt.axis()
xy = split[:-1] # anchor points
h = (ymax-ymin)/30; space = h/5; i = ymax + space # intercept
w = split[1:] - xy # rectangle width(s)
for j in range(len(xy)): # plot rectangles one-by-one
    plt.gca().add_patch(patches.Rectangle((xy[j], i), width=w[j], height=h, facecolor=color[j], clip_on=False, linewidth=1, edgecolor='k'))

# Add rectangle objects as tick labels (Y axis)
w = (ymax-ymin)/30; i = ymax # intercept
h = split[1:] - xy # rectangle height(s)
for j in range(len(xy)): # plot rectangles one-by-one
    plt.gca().add_patch(patches.Rectangle((i+space, xy[j]), width=w, height=h[j], facecolor=color[j], clip_on=False, linewidth=1, edgecolor='k'))

plt.tight_layout()
plt.savefig('/Volumes/Elements/TimeOfDay/Figures/corr_eve_b10.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)

#%% Display the corresponding graph
# -----------------------------------------------------
from nilearn import plotting
coords = atlas.region_coords

# We threshold to keep only the 5% of edges with the highest value
# because the graph is very dense
plotting.plot_connectome(mean_corr_mor, coords,
                         edge_threshold="95%", colorbar=True)

plotting.show()

#%% 3D visualization in a web browser
# -----------------------------------------------------
view = plotting.view_connectome(mean_corr_mor, coords, edge_threshold='95%')

# Open the plot in a web browser:
view.open_in_browser()

# Uncomment this if you are using Jupyter notebook:
# view



###############################################################################
#%% Load the functional datasets (lark and owl) and 
# extract signals (timeseries) on a parcellation defined by labels
# then calculating the correlation and binarized matrices
# -----------------------------------------------------
# Using the NiftiLabelsMasker
from nilearn.input_data import NiftiLabelsMasker
from nilearn.connectome import ConnectivityMeasure

masker = NiftiLabelsMasker(labels_img=atlas_filename, standardize=True,
                           memory='nilearn_cache', verbose=5)

data_path = '/Users/Farzad/Desktop/TimeOfDay/Data_type'
data_dir_list = ['Lark', 'Owl']
#data_dir_list = os.listdir(data_path)
labels_name = {'Lark':0,'Owl':1}
ts_list = []
corr_list = []
bin_list = []
label_list = []

density = 0.05
n = round(density*roi*(roi-1)/2)*2 # number of strongest connections * 2 (nodes)
upper, lower = 1, 0

for dataset in data_dir_list:
    fmri_list=sorted(os.listdir(data_path+'/'+ dataset))
    print ('Loading the nifti files of dataset-'+'{}\n'.format(dataset))
    label = labels_name[dataset]
    for fmri in fmri_list:
        # From nifti files to the signal time series in a numpy array.
        if fmri.endswith(".nii"):
            ts = masker.fit_transform(data_path + '/'+ dataset + '/'+ fmri ) # ts = masker.fit_transform(fmri_filenames, confounds=data.confounds)
            corr_measure = ConnectivityMeasure(kind='correlation') # kind{“correlation”, “partial correlation”, “tangent”, “covariance”, “precision”}, optional
            corr_matrix = corr_measure.fit_transform([ts])[0]
            # remove the self-connections (mask the main diagonal)
            np.fill_diagonal(corr_matrix, 0)
            # saving correlations as txt files in FC directory, for MATLAB use (graph theory)
            FC_path = '/Users/Farzad/Desktop/TimeOfDay/FC_type/' + dataset + '/'
            np.savetxt(FC_path + fmri + '.txt', corr_matrix, fmt='%.18e', delimiter=' ', newline='\n')
            
            # binarizing the individual correlation matrix (if needed, rarely use)
            abs_matrix = np.absolute(corr_matrix) # absolute value
            flat = abs_matrix.flatten()
            flat.sort()
            threshold = flat[-n]
            bin_matrix = np.where(abs_matrix > threshold, upper, lower)
            
            ts_list.append(ts)
            corr_list.append(corr_matrix)
            bin_list.append(bin_matrix)
            label_list.append(label)
            
fmri_ts = np.array(ts_list)
fmri_corr = np.array(corr_list)
fmri_bin = np.array(bin_list)

import matplotlib.pyplot as plt
plt.plot(np.arange(0, 200), fmri_ts[0,4,:].transpose(), color='red')
plt.show()
           
#%% Calculating the mean binarized matrix for each group
# -----------------------------------------

# Lrak
mean_corr_lark = np.mean(fmri_corr[0:int(len(label_list)/2)], axis=0)

abs_matrix = np.absolute(mean_corr_lark)
flat = abs_matrix.flatten()
flat.sort()
threshold = flat[-n]
mean_bin_lark = np.where(abs_matrix > threshold, upper, lower)

# Owl
mean_corr_owl = np.mean(fmri_corr[int(len(label_list)/2):int(len(label_list))], axis=0)

abs_matrix = np.absolute(mean_corr_owl)
flat = abs_matrix.flatten()
flat.sort()
threshold = flat[-n]
mean_bin_owl = np.where(abs_matrix > threshold, upper, lower)

# Saving the files
Bin_path = '/Users/Farzad/Desktop/TimeOfDay/FC_type/'
np.savetxt(Bin_path + 'mean_corr_lark.txt', mean_corr_lark, fmt='%.18e', delimiter=' ', newline='\n')
np.savetxt(Bin_path + 'mean_corr_owl.txt', mean_corr_owl, fmt='%.18e', delimiter=' ', newline='\n')
np.savetxt(Bin_path + 'mean_bin_lark_' + str(int(density*100)) + '%.txt', mean_bin_lark, fmt='%.18e', delimiter=' ', newline='\n')
np.savetxt(Bin_path + 'mean_bin_owl_' + str(int(density*100)) + '%.txt', mean_bin_owl, fmt='%.18e', delimiter=' ', newline='\n')

np.savetxt(Bin_path + 'Schaefer_7Networks_200.txt', labels, fmt='%s', delimiter=' ', newline='\n')



#%% MODULARITY

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

# pip install teneto

# https://teneto.readthedocs.io/en/latest/tutorial.html
# https://teneto.readthedocs.io/en/latest/tutorial/networkmeasures.html
from teneto import communitymeasures
from nilearn import datasets

# import community assignment of all groups (morning, evening, lark, owl)
communities = sio.loadmat('/Volumes/Elements/TimeOfDay/Modularity/S_all_1.3,-1.0.mat', squeeze_me=True)['S_all'];
n_set = communities.shape[0]
atlas = datasets.fetch_atlas_schaefer_2018(n_rois=200, yeo_networks=7, resolution_mm=1)
labels = atlas.labels.astype('U') # covert array of bytes to array of strings

# create static communities (networks' labels)
networks = ['Vis', 'SomMot', 'DorsAttn', 'SalVentAttn', 'Limbic', 'Cont', 'Default']
static_communities = np.zeros((200,))
# find networks in atlas.labels and assign a label[1-7] to each
for i, network in enumerate(networks):   
    idx = np.array([network in s for s in labels], dtype=bool)
    static_communities[idx] = i+1 # 1-7

allegiance, flexibility, integration, recruitment, promiscuity = [], [], [], [], []
allegiance_coarse_lr = [np.zeros((14,14)) for i in range(n_set)]
allegiance_coarse = [np.zeros((7,7)) for i in range(n_set)]

# Find index where elements change value in static_communities array
pivot = np.where(static_communities[:-1] != static_communities[1:])[0]
pivot = np.concatenate([pivot,[199]])

for s in range(n_set):
    
    allegiance.append(communitymeasures.allegiance(communities[s]))  
    flexibility.append(communitymeasures.flexibility(communities[s]))
    integration.append(communitymeasures.integration(communities[s], static_communities))
    recruitment.append(communitymeasures.recruitment(communities[s], static_communities))
    promiscuity.append(communitymeasures.promiscuity(communities[s])) # 0 entails only 1 community. 1 entails all communities
    
    # create coarse allegiance metrices
    p1, q1 = 0, 0
    for i, p2 in enumerate(pivot): 
        for j, q2 in enumerate(pivot): 
            allegiance_coarse_lr[s][i,j] = np.nanmean(allegiance[s][p1:p2+1, q1:q2+1])
            q1 = q2+1
        p1 = p2+1
        q1 = 0
    
    # If you have an array of shape (K * M, K * N), you can transform it into something of shape (K * K, M, N) using reshape and transpose
    allegiance_coarse[s] = np.mean(allegiance_coarse_lr[s].reshape(2, 7, 2, 7).transpose(0, 2, 1, 3).reshape(-1, 7, 7), axis=0)
        
#plt.imshow(allegiance_coarse[1])
#plt.colorbar()

# regression between morning and evening [integration, recruitment]
# define permutation test using monte-carlo method
def perm_test(xs, ys, nmc):
    n, k = len(xs), 0
    diff = np.abs(np.mean(xs) - np.mean(ys))
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.abs(np.mean(zs[:n]) - np.mean(zs[n:]))
    return k / nmc

# plot regressions
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(6, 12))
import seaborn as sns
sns.set(style = 'white') # whitegrid
# recruitment
x = recruitment[0]
y = recruitment[1]
sns.regplot(x, y, ci=95, scatter_kws={"color":"black"}, line_kws={"color":"red", 'label':'Regression line'}, ax=ax1)
ax1.set_xlabel('Morning Session', size='large') # fontsize=20
ax1.set_ylabel('Evening Session', size='large')
ax1.set_title('Recruitment', size='x-large')
lims = [np.min([ax1.get_xlim(), ax1.get_ylim()]),  # min of both axes
        np.max([ax1.get_xlim(), ax1.get_ylim()])]  # max of both axes
ax1.plot(lims, lims, 'k--', alpha=0.8, zorder=0, label='Identity line')
ax1.axis('square') # scaled, equal, square
#ax1.collections[1].set_label('95% CI')
ax1.legend(loc="best")
perm_test(x, y, 30000)

# integration
x = integration[0]
y = integration[1]
sns.regplot(x, y, ci=95, scatter_kws={"color": "black"}, line_kws={"color": "red", 'label':'Regression line'}, ax=ax2)
y_min, y_max = ax2.get_ylim()
x_min, x_max = ax2.get_xlim()
ax2.set_xlabel('Morning Session', size='large') # fontsize=20
ax2.set_ylabel('Evening Session', size='large')
ax2.set_title('Integration', size='x-large')
lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
        np.max([ax2.get_xlim(), ax2.get_ylim()])]  # max of both axes
ax2.plot(lims, lims, 'k--', alpha=0.8, zorder=0, label='Identity line')
ax2.axis('square') # scaled, equal, square
#ax2.collections[1].set_label('95% CI')
ax2.legend(loc="best")
perm_test(x, y, 30000)

plt.tight_layout(pad=3.0) # spacing between subplots
plt.show()

fig.savefig('/Volumes/Elements/TimeOfDay/Figures/scatter_int&rec.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)


"""
# null model shaded area
xx = np.concatenate((integration[0], integration[1]), axis=0)
xx = xx[np.random.choice(xx.shape[0], 300, replace=True)]
yy = xx + np.random.uniform(-.13,.13, size=(300,))
sns.regplot(xx, yy, ci=95, scatter=None, line_kws={"color":"k", 'linestyle':'--', 'label':'95% CI'}, ax=ax2)
#ax2.axis('square') # scaled, equal, square
ax2.collections[1].set_label('95% CI (null)')

# Test model coefficient (regression slope) against some value (e.g., slope 1)
from scipy import stats
res = stats.linregress(x, y) # slope, intercept, rvalue, pvalue and stderr
print(f"R-squared: {res.rvalue**2:.6f}")
t_value = ((res.slope - (1))/res.stderr)
n = 31
pval = stats.t.sf(np.abs(t_value), n-1)*2"""

#%% Allegiance matrix plots
import matplotlib.patches as patches
cmap='jet' # jet, rainbow, twilight, twilight_shifted, terrain, gist_earth, CMRmap
# morning
f = plt.figure(figsize=(19,15))
plt.matshow(allegiance[0], fignum=f.number, vmin = 0, vmax = 1, cmap=cmap) # jet, rainbow, twilight_shifted, terrain, gist_earth, gnuplot, CMRmap
plt.title('Morning Session', fontsize=28)
#plt.xticks(range(allegiance[0].shape[1]), labels, fontsize=10, rotation=90) #systems or labels
#plt.yticks(range(allegiance[0].shape[1]), labels, fontsize=10)
plt.xticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
plt.yticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
cb = plt.colorbar() 
cb.ax.tick_params(labelsize=20)
plt.axvline(x=100-0.5,color='white',linewidth=3)
plt.axhline(y=100-0.5,color='white',linewidth=3)
# Draw grid lines
plt.grid(color='white', linestyle='-', linewidth=0.7)
plt.tick_params(
    axis='both',       # changes apply to the x,y-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,        # ticks along the left edge are off
    right=False,       # ticks along the right edge are off
    labeltop=False,    # labels along the top edge are off
    labelleft=False)   # labels along the left edge are off
# Add rectangle objects as tick labels (X axis)
xmin, xmax, ymin, ymax = plt.axis()
h = (ymax-ymin)/30; space = h/5; i = ymax + space # intercept
plt.gca().add_patch(patches.Rectangle((-0.5, i), width=13.5+0.5, height=h, facecolor='#A251AC', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((13.5, i), width=29.5-13.5, height=h, facecolor='#789AC1', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((29.5, i), width=42.5-29.5, height=h, facecolor='#409832', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((42.5, i), width=53.5-42.5, height=h, facecolor='#E165FE', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((53.5, i), width=59.5-53.5, height=h, facecolor='#F6FDC9', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((59.5, i), width=72.5-59.5, height=h, facecolor='#EFB944', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((72.5, i), width=99.5-72.5, height=h, facecolor='#D9717D', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((99.5, i), width=114.5-99.5, height=h, facecolor='#A251AC', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((114.5, i), width=133.5-114.5, height=h, facecolor='#789AC1', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((133.5, i), width=146.5-133.5, height=h, facecolor='#409832', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((146.5, i), width=157.5-146.5, height=h, facecolor='#E165FE', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((157.5, i), width=163.5-157.5, height=h, facecolor='#F6FDC9', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((163.5, i), width=180.5-163.5, height=h, facecolor='#EFB944', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((180.5, i), width=199.5-180.5, height=h, facecolor='#D9717D', clip_on=False, linewidth=1.5, edgecolor='k'))
# Add rectangle objects as tick labels (Y axis)
w = (ymax-ymin)/30; i = ymax # intercept
plt.gca().add_patch(patches.Rectangle((i+space, -0.5), width=w, height=13.5+0.5, facecolor='#A251AC', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 13.5), width=w, height=29.5-13.5, facecolor='#789AC1', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 29.5), width=w, height=42.5-29.5, facecolor='#409832', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 42.5), width=w, height=53.5-42.5, facecolor='#E165FE', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 53.5), width=w, height=59.5-53.5, facecolor='#F6FDC9', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 59.5), width=w, height=72.5-59.5, facecolor='#EFB944', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 72.5), width=w, height=99.5-72.5, facecolor='#D9717D', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 99.5), width=w, height=114.5-99.5, facecolor='#A251AC', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 114.5), width=w, height=133.5-114.5, facecolor='#789AC1', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 133.5), width=w, height=146.5-133.5, facecolor='#409832', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 146.5), width=w, height=157.5-146.5, facecolor='#E165FE', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 157.5), width=w, height=163.5-157.5, facecolor='#F6FDC9', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 163.5), width=w, height=180.5-163.5, facecolor='#EFB944', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 180.5), width=w, height=199.5-180.5, facecolor='#D9717D', clip_on=False, linewidth=1.5, edgecolor='k'))

plt.tight_layout()
plt.savefig('/Volumes/Elements/TimeOfDay/Figures/allegiance_mor.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)

# evening
f = plt.figure(figsize=(19,15))
plt.matshow(allegiance[1], fignum=f.number, vmin = 0, vmax = 1, cmap=cmap) # jet, rainbow, twilight_shifted, terrain, gist_earth, gnuplot, CMRmap
plt.title('Evening Session', fontsize=28)
#plt.xticks(range(allegiance[0].shape[1]), labels, fontsize=10, rotation=90) #systems or labels
#plt.yticks(range(allegiance[0].shape[1]), labels, fontsize=10)
plt.xticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
plt.yticks([13.5, 29.5, 42.5, 53.5, 59.5, 72.5, 99.5, 114.5, 133.5, 146.5, 157.5, 163.5, 180.5])
cb = plt.colorbar() 
cb.ax.tick_params(labelsize=20)
plt.axvline(x=100-0.5,color='white',linewidth=3)
plt.axhline(y=100-0.5,color='white',linewidth=3)
# Draw grid lines
plt.grid(color='white', linestyle='-', linewidth=0.7)
plt.tick_params(
    axis='both',       # changes apply to the x,y-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    left=False,        # ticks along the left edge are off
    right=False,       # ticks along the right edge are off
    labeltop=False,    # labels along the top edge are off
    labelleft=False)   # labels along the left edge are off
# Add rectangle objects as tick labels (X axis)
xmin, xmax, ymin, ymax = plt.axis()
h = (ymax-ymin)/30; space = h/5; i = ymax + space # intercept
plt.gca().add_patch(patches.Rectangle((-0.5, i), width=13.5+0.5, height=h, facecolor='#A251AC', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((13.5, i), width=29.5-13.5, height=h, facecolor='#789AC1', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((29.5, i), width=42.5-29.5, height=h, facecolor='#409832', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((42.5, i), width=53.5-42.5, height=h, facecolor='#E165FE', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((53.5, i), width=59.5-53.5, height=h, facecolor='#F6FDC9', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((59.5, i), width=72.5-59.5, height=h, facecolor='#EFB944', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((72.5, i), width=99.5-72.5, height=h, facecolor='#D9717D', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((99.5, i), width=114.5-99.5, height=h, facecolor='#A251AC', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((114.5, i), width=133.5-114.5, height=h, facecolor='#789AC1', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((133.5, i), width=146.5-133.5, height=h, facecolor='#409832', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((146.5, i), width=157.5-146.5, height=h, facecolor='#E165FE', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((157.5, i), width=163.5-157.5, height=h, facecolor='#F6FDC9', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((163.5, i), width=180.5-163.5, height=h, facecolor='#EFB944', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((180.5, i), width=199.5-180.5, height=h, facecolor='#D9717D', clip_on=False, linewidth=1.5, edgecolor='k'))
# Add rectangle objects as tick labels (Y axis)
w = (ymax-ymin)/30; i = ymax # intercept
plt.gca().add_patch(patches.Rectangle((i+space, -0.5), width=w, height=13.5+0.5, facecolor='#A251AC', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 13.5), width=w, height=29.5-13.5, facecolor='#789AC1', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 29.5), width=w, height=42.5-29.5, facecolor='#409832', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 42.5), width=w, height=53.5-42.5, facecolor='#E165FE', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 53.5), width=w, height=59.5-53.5, facecolor='#F6FDC9', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 59.5), width=w, height=72.5-59.5, facecolor='#EFB944', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 72.5), width=w, height=99.5-72.5, facecolor='#D9717D', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 99.5), width=w, height=114.5-99.5, facecolor='#A251AC', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 114.5), width=w, height=133.5-114.5, facecolor='#789AC1', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 133.5), width=w, height=146.5-133.5, facecolor='#409832', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 146.5), width=w, height=157.5-146.5, facecolor='#E165FE', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 157.5), width=w, height=163.5-157.5, facecolor='#F6FDC9', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 163.5), width=w, height=180.5-163.5, facecolor='#EFB944', clip_on=False, linewidth=1.5, edgecolor='k'))
plt.gca().add_patch(patches.Rectangle((i+space, 180.5), width=w, height=199.5-180.5, facecolor='#D9717D', clip_on=False, linewidth=1.5, edgecolor='k'))

plt.tight_layout()
plt.savefig('/Volumes/Elements/TimeOfDay/Figures/allegiance_eve.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)

#%% # recruitment and integration coeeficiencts for each brain region

from nilearn import plotting
node_coords = np.array([(-24,-53,-9), (-26,-77,-14),(-45,-69,-8), (-10,-67,-4), (-27,-95,-12), (-14,-44,-3), (-5,-93,-4), (-47,-70,10), (-23,-97,6), (-11,-70,7), (-40,-85,11), (-12,-73,22), (-7,-87,28), (-23,-87,23), (-51,-4,-2), (-53,-24,9), (-37,-21,16), (-55,-4,10), (-53,-22,18), (-56,-8,31), (-47,-9,46), (-7,-12,46), (-49,-28,57), (-40,-25,57), (-31,-46,63), (-32,-22,64), (-26,-38,68),(-20,-11,68), (-5,-29,67), (-19,-31,68), (-43,-48,-19), (-57,-60,-1), (-26,-70,38), (-54,-27,42), (-41,-35,47), (-33,-49,47),  (-17,-73,54),(-29,-60,59), (-6,-60,57), (-17,-53,68), (-31,-4,53), (-22,6,62), (-48,6,29), (-56,-40,20), (-61,-26,28), (-60,-39,36), (-39,-4,-4), (-33,20,5), (-39,1,11), (-51,9,11), (-28,43,31), (-6,9,41), (-11,-35,46), (-6,-3,65), (-24,22,-20), (-10,35,-21), (-29,-6,-39), (-45,-20,-30), (-28,10,-34), (-43,8,-19), (-53,-51,46), (-35,-62,48), (-45,-42,46), (-61,-43,-13), (-32,42,-13), (-42,49,-6), (-28,58,8), (-42,40,16), (-44,20,27), (-43,6,43), (-9,-73,38), (-5,-29,28), (-3,4,30), (-47,8,-33), (-60,-19,-22), (-56,-6,-12), (-58,-30,-4), (-58,-43,7), (-48,-57,18), (-39,-80,31), (-57,-54,28), (-46,-66,38), (-35,20,-13), (-6,36,-10), (-46,31,-7), (-12,63,-6), (-52,22,8), (-6,44,7), (-8,59,21), (-6,30,25), (-11,47,45), (-3,33,43), (-40,19,49), (-24,25,49), (-9,17,63), (-11,-56,13), (-5,-55,27), (-4,-31,36), (-6,-54,42), (-26,-32,-18), (39,-35,-23), (28,-36,-14), (29,-69,-12), (12,-65,-5), (48,-71,-6), (11,-92,-5), (16,-46,-1), (31,-94,-4), (9,-75,9), (22,-60,7), (42,-80,10), (20,-90,22), (11,-74,26), (16,-85,39), (33,-75,32), (51,-15,5), (64,-23,8), (38,-13,15), (44,-27,18), (59,0,10), (56,-11,14), (58,-5,31), (10,-15,41), (51,-22,52), (47,-11,48), (7,-11,51), (40,-24,57), (32,-40,64), (33,-21,65), (29,-34,65), (22,-9,67), (10,-39,69), (6,-23,69), (20,-29,70), (50,-53,-15), (52,-60,9), (59,-16,34), (46,-38,49), (41,-31,46), (15,-73,53), (34,-48,51), (26,-61,58), (8,-56,61), (21,-48,70), (34,-4,52), (26,7,58), (52,11,21), (57,-45,9), (60,-39,17), (60,-26,27), (51,4,40), (41,6,-15), (46,-4,-4),  (36,24,5), (43,7,4), (7,9,41), (11,-36,47), (8,3,66), (12,39,-22), (28,22,-19), (15,64,-8), (30,9,-38), (47,-12,-35), (25,-11,-32), (62,-37,37), (53,-42,48), (37,-63,47), (63,-41,-12), (34,21,-8), (36,46,-13), (29,58,5), (43,45,10), (46,24,26), (30,48,27), (41,33,37), (42,14,49), (14,-70,37), (5,-24,31), (5,3,30), (7,31,28), (7,25,55), (47,-69,27), (54,-50,28), (51,-59,44), (47,13,-30), (61,-13,-21), (55,-6,-10), (63,-27,-6), (52,-31,2), (51,28,0), (5,37,-14), (8,42,4), (6,29,15), (8,58,18), (15,46,44), (29,30,42), (23,24,53), (12,-55,15), (7,-49,31), (6,-58,44)])

# recruitment (morning vs evening)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
node_values = recruitment[0]; plotting.plot_markers(node_values, node_coords, node_cmap=cmap, title=None, colorbar=True, axes=ax1)
ax1.set_title('Recruitment', size='large') # fontsize=20
node_values = recruitment[1]; plotting.plot_markers(node_values, node_coords, node_cmap=cmap, title=None, colorbar=True, axes=ax2)
#plt.tight_layout()
plt.savefig('/Volumes/Elements/TimeOfDay/Figures/recruitment.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)

# integration (morning vs evening)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
node_values = integration[0]; plotting.plot_markers(node_values, node_coords, node_cmap=cmap, title=None, colorbar=True, axes=ax1)
ax1.set_title('Integration', size='large') # fontsize=20
node_values = integration[1]; plotting.plot_markers(node_values, node_coords, node_cmap=cmap, title=None, colorbar=True, axes=ax2)
#plt.tight_layout()
plt.savefig('/Volumes/Elements/TimeOfDay/Figures/integration.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)

#%% Coarse Allegiance matrix plots (Functional Cartography)
networks_lr = ['Vis_L', 'SomMot_L', 'DorsAttn_L', 'SalVentAttn_L', 'Limbic_L', 'Cont_L', 'Default_L',
            'Vis_R', 'SomMot_R', 'DorsAttn_R', 'SalVentAttn_R', 'Limbic_R', 'Cont_R', 'Default_R']
networks = ['VN', 'SMN', 'DAN', 'VAN', 'LN', 'FPN', 'DMN']
cmap='jet' # jet, rainbow, twilight, twilight_shifted, terrain, gist_earth, CMRmap

fig = plt.figure(figsize=(12,6))

# morning vs evening
plt.subplot(1, 2, 1)
plt.imshow(allegiance_coarse[0], vmin=0, vmax=1, cmap=cmap) # jet, rainbow, twilight_shifted, terrain, gist_earth, gnuplot, CMRmap
plt.title('Morning Session', size='large')
plt.yticks(range(allegiance_coarse[0].shape[1]), networks, fontsize=12, rotation=0)
plt.tick_params(left=False,right=False,bottom=False,top=False,
                labelleft=True, labelright=False, labelbottom=False, labeltop=False)

plt.subplot(1, 2, 2)
im = plt.imshow(allegiance_coarse[1], vmin=0, vmax=1, cmap=cmap)
plt.title('Evening Session', size='large')
plt.yticks(range(allegiance_coarse[0].shape[1]), networks, fontsize=12, rotation=0)
plt.tick_params(left=False,right=False,bottom=False,top=False,
                labelleft=False, labelright=False, labelbottom=False, labeltop=False)
plt.tight_layout(pad=3.0) # spacing between subplots

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.25, 0.025, 0.50])
fig.colorbar(im, cax=cbar_ax)

plt.savefig('/Volumes/Elements/TimeOfDay/Figures/allegiance_net.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)

#%% Creating Circular flow charts (circos or chord)
# https://mne.tools/mne-connectivity/dev/generated/mne_connectivity.viz.plot_connectivity_circle.html
# https://stackoverflow.com/questions/33388867/creating-circular-flow-charts-circos
# https://www.python-graph-gallery.com/406-chord-diagram_mne

from mne.viz import plot_connectivity_circle

palette = ['purple', 'blue', 'green', 'violet', 'yellow', 'orange', 'red']
palette = ['#A251AC', '#789AC1', '#409832', '#E165FE', '#F6FDC9', '#EFB944', '#D9717D']
node_names = networks # List of labels
cmap = 'Blues' # Blues, hot_r

# morning chord
con = allegiance_coarse[0] # NaN so it doesn't display the weak links
fig = plt.figure(num=None, figsize=(8, 8), facecolor='white')
plot_connectivity_circle(con, node_names, title=None,
    facecolor='white', textcolor='black', colormap=cmap, vmin=0, vmax=0.6, 
    colorbar=True, colorbar_size=0.5, colorbar_pos=(-.6, 0.5),
    node_width=None, node_colors=palette, linewidth=7, fontsize_names=8, fig=fig)
fig.savefig('/Volumes/Elements/TimeOfDay/Figures/circle_mor.pdf', facecolor='white',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300) 

# evening chord
con = allegiance_coarse[1] # NaN so it doesn't display the weak links
fig = plt.figure(num=None, figsize=(8, 8), facecolor='white')
plot_connectivity_circle(con, node_names, title=None,
    facecolor='white', textcolor='black', colormap=cmap, vmin=0, vmax=0.6,
    colorbar=True, colorbar_size=0.5, colorbar_pos=(-.6, 0.5),
    node_width=None, node_colors=palette, linewidth=7, fontsize_names=8, fig=fig)
fig.savefig('/Volumes/Elements/TimeOfDay/Figures/circle_eve.pdf', facecolor='white',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300)  

#%% catplot (multiple barplot) for recruitment and integration
import numpy as np
import pandas as pd
import random
networks = ['VN', 'SMN', 'DAN', 'VAN', 'LN', 'FPN', 'DMN']
# corase recruitment values
rec_mor = np.diag(allegiance_coarse[0]) # morning session
rec_eve = np.diag(allegiance_coarse[1]) # evening session
# corase integration values
int_mor = (allegiance_coarse[0].sum(1)-np.diag(allegiance_coarse[0]))/(allegiance_coarse[0].shape[1]-1) # morning session
int_eve = (allegiance_coarse[1].sum(1)-np.diag(allegiance_coarse[1]))/(allegiance_coarse[1].shape[1]-1) # evening session

num = 2
data = np.concatenate((rec_mor, rec_eve, int_mor, int_eve), axis=0)
df = pd.DataFrame(data=data, columns=["Values"]) # index=rows
metric = np.repeat(['Recruitment', 'Integration'], 14, axis=0)
df['Metric'] = metric
group = np.tile(networks, 2*num)
df['Network'] = group  
session = np.tile(np.repeat(['Morning', 'Evening'], 7, axis=0), num)
df['Session'] = session 

sns.set(style="white") #sns.set(font_scale = 1.5)
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(5, 8))

# Recruitment
sns.barplot(x="Network", y="Values", hue="Session", ax=ax1,
                 data=df.loc[df['Metric']=='Recruitment'],
                 palette=['#FAD02C','#0000FF'])
ax1.legend_.remove()
ax1.set(xticklabels=networks)
ax1.set(xlabel=None, ylabel=None)
ax1.set_title('Recruitment', fontsize=16) 
ax1.spines['top'].set_visible(False); ax1.spines['right'].set_visible(False) 
ax1.axhline(y=df.loc[df['Metric']=='Recruitment']['Values'].mean(), color='r', linestyle='--', linewidth=1.5, label='Mean')
"""# adding nulls 
xmin, xmax, ymin, ymax = plt.axis()
w=0.38; x1=-0.39; x2=x1+w+0.02
for i, net in enumerate(networks):
    # height + a randomness
    l1 = np.array(df['Values'][(df['Metric']=='Recruitment') & (df['Network']==net) & (df['Session']=='Morning')])
    l2 = np.array(df['Values'][(df['Metric']=='Recruitment') & (df['Network']==net) & (df['Session']=='Evening')])
    l11 = (l1+l2)/2 + random.uniform(-0.02*l1, 0.02*l1)
    l22 = (l1+l2)/2 + random.uniform(-0.02*l2, 0.02*l2)
    l = abs(l1-l2)
    h1 = (ymax-ymin)/15; h1 = h1 + random.uniform(0, 0.1*l)
    h2 = (ymax-ymin)/15; h2 = h2 + random.uniform(0, 0.1*l)
    if i == 0: # just becuase we need one legend for the null
        ax1.add_patch(patches.Rectangle((x1+i, (l1+l2)/2-h1/2-0.0008), width=w, height=h1, facecolor='r', alpha=0.2, linewidth=1, edgecolor='k', clip_on=False, label='Null Range'))
    else:
        ax1.add_patch(patches.Rectangle((x1+i, l11-h1/2-0.0008), width=w, height=h1, facecolor='r', alpha=0.2, linewidth=1, edgecolor='k', clip_on=False))    
    ax1.add_patch(patches.Rectangle((x2+i, l22-h2/2-0.0008), width=w, height=h2, facecolor='r', alpha=0.2, linewidth=1, edgecolor='k', clip_on=False))
"""
# reordering the legend labels
handles, labels = ax1.get_legend_handles_labels()
order = [1, 2, 0]  
ax1.legend([handles[i] for i in order], [labels[i] for i in order])
# Integration
sns.barplot(x="Network", y="Values", hue="Session", ax=ax2,
                 data=df.loc[df['Metric']=='Integration'],
                 palette=['#FAD02C','#0000FF'])
ax2.set(xticklabels=networks)
ax2.set(xlabel=None, ylabel=None)
ax2.set_title('Integration', fontsize=16) 
ax2.spines['top'].set_visible(False); ax2.spines['right'].set_visible(False) 
ax2.axhline(y=df.loc[df['Metric']=='Integration']['Values'].mean(), color='r', linestyle='--', linewidth=1.5, label='Mean')
"""# adding nulls
xmin, xmax, ymin, ymax = plt.axis()
w=0.38; x1=-0.39; x2=x1+w+0.02
for i, net in enumerate(networks):
    # height + a randomness
    l1 = np.array(df['Values'][(df['Metric']=='Integration') & (df['Network']==net) & (df['Session']=='Morning')])
    l2 = np.array(df['Values'][(df['Metric']=='Integration') & (df['Network']==net) & (df['Session']=='Evening')])
    l11 = (l1+l2)/2 + random.uniform(-0.02*l1, 0.02*l1)
    l22 = (l1+l2)/2 + random.uniform(-0.02*l2, 0.02*l2)
    l = abs(l1-l2)
    h1 = (ymax-ymin)/15; h1 = h1 + random.uniform(0, 0.1*l)
    h2 = (ymax-ymin)/15; h2 = h2 + random.uniform(0, 0.1*l)
    if i == 0: # just becuase we need one legend for the null
        ax2.add_patch(patches.Rectangle((x1+i, (l1+l2)/2-h1/2-0.0008), width=w, height=h1, facecolor='r', alpha=0.2, linewidth=1, edgecolor='k', clip_on=False, label='Null Range'))
    else:
        ax2.add_patch(patches.Rectangle((x1+i, l11-h1/2-0.0008), width=w, height=h1, facecolor='r', alpha=0.2, linewidth=1, edgecolor='k', clip_on=False))    
    ax2.add_patch(patches.Rectangle((x2+i, l22-h2/2-0.0008), width=w, height=h2, facecolor='r', alpha=0.2, linewidth=1, edgecolor='k', clip_on=False))
"""
# reordering the legend labels
handles, labels = plt.gca().get_legend_handles_labels()
order = [1, 2, 0]  
plt.legend([handles[i] for i in order], [labels[i] for i in order])
 
plt.tight_layout()
plt.savefig('/Volumes/Elements/TimeOfDay/Figures/catplot.pdf',
            bbox_inches='tight', pad_inches=0, format='pdf', dpi=300) 









