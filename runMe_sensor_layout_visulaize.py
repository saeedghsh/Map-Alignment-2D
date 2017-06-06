from __future__ import print_function

import sys
sys.path.append( u'../arrangement/' )
from core import map_alignment as mapali 

import cv2
import numpy as np
import matplotlib.pyplot as plt

################################################################################
################################################################################ 
################################################################################
key_pairs = { 1: ['HIH_layout', 'HIH_01_tango'], 
              2: ['HIH_layout', 'HIH_02_tango'],
              3: ['HIH_layout', 'HIH_03_tango'],
              4: ['HIH_layout', 'HIH_04_tango'],

              5: ['kpt4a_layout', 'kpt4a_f_tango'],
              6: ['kpt4a_layout', 'kpt4a_kb_tango'],
              7: ['kpt4a_layout', 'kpt4a_kl_tango'],
              8: ['kpt4a_layout', 'kpt4a_lb_tango'],
              
              9: ['E5_layout', 'E5_01_tango'],
              10: ['E5_layout', 'E5_02_tango'],
              11: ['E5_layout', 'E5_03_tango'],
              12: ['E5_layout', 'E5_04_tango'],
              13: ['E5_layout', 'E5_05_tango'],
              14: ['E5_layout', 'E5_06_tango'],
              15: ['E5_layout', 'E5_07_tango'],
              16: ['E5_layout', 'E5_08_tango'],
              17: ['E5_layout', 'E5_09_tango'],
              18: ['E5_layout', 'E5_10_tango'],
              19: ['E5_layout', 'E5_11_tango'],
              20: ['E5_layout', 'E5_12_tango'],
              21: ['E5_layout', 'E5_13_tango'],
              22: ['E5_layout', 'E5_14_tango'],

              23: ['F5_layout', 'F5_01_tango'],
              24: ['F5_layout', 'F5_02_tango'],
              25: ['F5_layout', 'F5_03_tango'],
              26: ['F5_layout', 'F5_04_tango'],
              27: ['F5_layout', 'F5_05_tango'],
              28: ['F5_layout', 'F5_06_tango'],
              29: ['F5_layout', 'F5_07_tango'],
              30: ['F5_layout', 'F5_08_tango'],
              31: ['F5_layout', 'F5_09_tango'],
              32: ['F5_layout', 'F5_10_tango'],
              33: ['F5_layout', 'F5_11_tango'],
              34: ['F5_layout', 'F5_12_tango'],
              35: ['F5_layout', 'F5_13_tango'],
              36: ['F5_layout', 'F5_14_tango'] }

################################################################################
arr_config = {'multi_processing':4, 'end_point':False, 'timing':False}

prun_dis_neighborhood = 2
prun_dis_threshold = .15 # for home environment - distance image 2
# prun_dis_threshold = .075 # for lab environment - distance image 2

con_map_neighborhood = 3 #1
con_map_cross_thr = 9 #3

scale_mismatch_ratio_threshold = .2 # .5 #.1
scale_bounds = [.5, 2] # [.3,3] [.1, 10]

too_many_tforms = 1000 #30000
dbscan_eps = .051
dbscan_min_samples = 2


### loading match scores of all hypotheses for box-plot
arr_ms = []
labels = {}
kpt4_idx = 1
for idx in key_pairs: #[24,32]
    keys = key_pairs[idx]

    label = keys[1][:len(keys[1])-6]
    if label[:4] == 'kpt4':
        label = 'KPT4A_0'+str(kpt4_idx)
        kpt4_idx += 1
        
    labels[idx] = label
    data = np.atleast_1d(np.load('arr_match_score_'+'_'.join(keys)+'.npy'))[0]
    data = [data[k] for k in data]
    
    arr_ms.append(data)

# ### loading match scores of winning hypotheses for marking
# import re
# scores = np.ones((36,36))
adr = '/home/saesha/Dropbox/myGits/orebro_visit/map_alignment_paper/figures/sensor_sensor_result/'
fle = 'sensos_sensor_matchscore'

# f = open(adr + fle + '.txt', 'r')
# for line in f:
#     spl = re.findall(r"[\w']+", line)
#     if len(spl) > 0:
#         idx1, idx2, score = int(spl[0]), int(spl[1]), float(spl[-1])/10000.
#         if idx1 == idx2: print (idx1)
#         scores[idx1, idx2] = score
#         scores[idx2, idx1] = score



######################################## plotting
fig, ax = plt.subplots(figsize=(10, 4))

ax.boxplot(arr_ms)

# for idx in range(36):
#     ax.plot(idx+1, scores[idx,idx], 'r.')

for idx,ams in enumerate(arr_ms):
    ax.plot(idx+1, max(ams), 'r.')



# plt.xticks( [1.5, 5.5, 14.5, 28.5], ['HIH','KPT4A', 'E5','F5'], rotation='vertical', horizontalalignment='center')
plt.xticks( [y+1 for y in range(len(arr_ms))],
            [labels[idx] for idx in labels],
            rotation='vertical', horizontalalignment='center')

# ax.set_xlabel('maps')
ax.set_ylabel('alignment match score')

# # add x-tick labels
# plt.setp(ax,
#          xticks=[y+1 for y in range(len(arr_ms))],
#          xticklabels=[labels[idx] for idx in labels],)


if 1:
    plt.savefig(adr + fle + '_boxplot.png', bbox_inches='tight')
    plt.close(fig)
else:
    plt.tight_layout()
    plt.show()

