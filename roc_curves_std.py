import numpy as np
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('ieee')
from sklearn.metrics import roc_curve, auc
from scipy import interpolate
import hmog_data
import touch_data


double_scores_hmog = hmog_data.fpr_tpr_scores.Scores.double_scores_hmog()
half_scores_hmog = hmog_data.fpr_tpr_scores.Scores.half_scores_hmog()

double_scores_touch = touch_data.fpr_tpr_scores.Scores.double_scores_hmog()
half_scores_touch = touch_data.fpr_tpr_scores.Scores.half_scores_hmog()

# Load the fpr and tpr scores for SSCL model on HMOG dataset

# 1) Scores on STD = 1
fpr_std_1_load_hmog = double_scores_hmog[0]
tpr_std_1_load_hmog = double_scores_hmog[1]



'''
--------------------------------------------------- A) Load the fpr and tpr scores for SSCL model on HMOG dataset
'''

# 2) Scores on STD = 2
fpr_std_2_load_hmog = double_scores_hmog[2]
tpr_std_2_load_hmog = double_scores_hmog[3]

# Load the fpr and tpr scores for SSCL model on HMOG dataset

# 3) Scores on STD = 4
fpr_std_4_load_hmog = double_scores_hmog[4]
tpr_std_4_load_hmog = double_scores_hmog[5]

# Load the fpr and tpr scores for SSCL model on HMOG dataset

# 4) Scores on STD = 8
fpr_std_8_load_hmog = double_scores_hmog[6]
tpr_std_8_load_hmog = double_scores_hmog[7]

# Load the fpr and tpr scores for SSCL model on HMOG dataset

# 5) Scores on STD = 16
fpr_std_16_load_hmog = double_scores_hmog[8]
tpr_std_16_load_hmog = double_scores_hmog[9]

# Load the fpr and tpr scores for SSCL model on HMOG dataset

# 6) Scores on STD = 32
fpr_std_32_load_hmog = double_scores_hmog[10]
tpr_std_32_load_hmog = double_scores_hmog[11]

# 7) Scores on STD = 1/2
fpr_std_half_load_hmog = half_scores_hmog[0]
tpr_std_half_load_hmog = half_scores_hmog[1]

# 8) Scores on STD = 1/4
fpr_std_fourth_load_hmog = half_scores_hmog[2]
tpr_std_fourth_load_hmog = half_scores_hmog[3]
# 6) Scores on STD = 1/8
fpr_std_eight_load_hmog = half_scores_hmog[4]
tpr_std_eight_load_hmog = half_scores_hmog[5]

# 10) Scores on STD = 1/16
fpr_std_sixteen_load_hmog = half_scores_hmog[6]
tpr_std_sixteen_load_hmog = half_scores_hmog[7]

# 11) Scores on STD = 1/32
fpr_std_thirty_two_load_hmog = half_scores_hmog[8]
tpr_std_thirty_two_load_hmog = half_scores_hmog[9]

'''
---------------------------------ROC curves on HMOG dataset-------------------------------------
'''
plt.figure(figsize=(8,6), dpi=60)
plt.rcParams['legend.fontsize']=15
#plt.rcParams.update({'figure.dpi': '500'})
plt.rcParams['font.size'] = 20
plt.plot(fpr_std_1_load_hmog, tpr_std_1_load_hmog, color='magenta', label=' $\sigma$ = 1 (auc=%0.3f)' %auc(fpr_std_1_load_hmog, tpr_std_1_load_hmog), linewidth=5)
plt.plot(fpr_std_2_load_hmog, tpr_std_2_load_hmog, color='green', label=' $\sigma$ = 2 (auc=%0.3f)' %auc(fpr_std_2_load_hmog, tpr_std_2_load_hmog), linewidth=2.5)
plt.plot(fpr_std_4_load_hmog, tpr_std_4_load_hmog, color='red', label= ' $\sigma$ = 4 (auc=%0.3f)' %auc(fpr_std_4_load_hmog, tpr_std_4_load_hmog), linewidth=2.5)
plt.plot(fpr_std_8_load_hmog, tpr_std_8_load_hmog, color='blue', label= ' $\sigma$ = 8 (auc=%0.3f)' %auc(fpr_std_8_load_hmog, tpr_std_8_load_hmog), linewidth=2.5)
plt.plot(fpr_std_16_load_hmog, tpr_std_16_load_hmog, color='cyan', label= ' $\sigma$ = 16 (auc=%0.3f)' %auc(fpr_std_16_load_hmog, tpr_std_16_load_hmog), linewidth=2.5)
plt.plot(fpr_std_32_load_hmog, tpr_std_32_load_hmog, color='orange', label= ' $\sigma$ = 32 (auc=%0.3f)' %auc(fpr_std_32_load_hmog, tpr_std_32_load_hmog), linewidth=2.5)


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve by varying standard deviation $\sigma$. \n Dataset: HMOG. $\sigma$ = [1, 2, 4, 8, 16, 32]')
plt.legend()
plt.show()

plt.figure(figsize=(8,6), dpi=60)
plt.rcParams['legend.fontsize']=15
#plt.rcParams.update({'figure.dpi': '500'})
plt.rcParams['font.size'] = 20
plt.plot(fpr_std_1_load_hmog, tpr_std_1_load_hmog, color='magenta', label=' $\sigma$ = 1 (auc=%0.3f)' %auc(fpr_std_1_load_hmog, tpr_std_1_load_hmog), linewidth=5)
plt.plot(fpr_std_half_load_hmog, tpr_std_half_load_hmog, color='teal', label=' $\sigma$ = 1/2 (auc=%0.3f)' %auc(fpr_std_half_load_hmog, tpr_std_half_load_hmog), linewidth=2.5)
plt.plot(fpr_std_fourth_load_hmog, tpr_std_fourth_load_hmog, color='olive', label= ' $\sigma$= 1/4 (auc=%0.3f)' %auc(fpr_std_fourth_load_hmog, tpr_std_fourth_load_hmog), linewidth=2.5)
plt.plot(fpr_std_eight_load_hmog, tpr_std_eight_load_hmog, color='maroon', label= ' $\sigma$= 1/8 (auc=%0.3f)' %auc(fpr_std_eight_load_hmog, tpr_std_eight_load_hmog), linewidth=2.5)
plt.plot(fpr_std_sixteen_load_hmog, tpr_std_sixteen_load_hmog, color='indigo', label= ' $\sigma$= 1/16 (auc=%0.3f)' %auc(fpr_std_sixteen_load_hmog, tpr_std_sixteen_load_hmog), linewidth=2.5)
plt.plot(fpr_std_thirty_two_load_hmog, tpr_std_thirty_two_load_hmog, color='#36454F', label= ' $\sigma$= 1/32 (auc=%0.3f)' %auc(fpr_std_thirty_two_load_hmog, tpr_std_thirty_two_load_hmog), linewidth=2.5)


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC curve by varying standard deviation $\sigma$. \n Dataset: HMOG. $\sigma$ = [1, 1/2, 1/4, 1/8, 1/16, 1/32]')
plt.legend()
plt.show()

'''
----------------------------------------B) Load the fpr and tpr scores for SSCL model on Touch dataset-------------------
'''

# 1) Scores on STD = 1
fpr_std_1_load_touch = double_scores_touch[0]
tpr_std_1_load_touch = double_scores_touch[1]

# Load the fpr and tpr scores for SSCL model on HMOG dataset

# 2) Scores on STD = 2
fpr_std_2_load_touch = double_scores_touch[2]
tpr_std_2_load_touch = double_scores_touch[3]

# Load the fpr and tpr scores for SSCL model on HMOG dataset

# 3) Scores on STD = 4
fpr_std_4_load_touch = double_scores_touch[4]
tpr_std_4_load_touch = double_scores_touch[5]

# Load the fpr and tpr scores for SSCL model on HMOG dataset

# 4) Scores on STD = 8
fpr_std_8_load_touch = double_scores_touch[6]
tpr_std_8_load_touch = double_scores_touch[7]

# Load the fpr and tpr scores for SSCL model on HMOG dataset

# 5) Scores on STD = 16
fpr_std_16_load_touch = double_scores_touch[8]
tpr_std_16_load_touch = double_scores_touch[9]

# Load the fpr and tpr scores for SSCL model on HMOG dataset

# 6) Scores on STD = 32
fpr_std_32_load_touch = double_scores_touch[10]
tpr_std_32_load_touch = double_scores_touch[11]

# 7) Scores on STD = 1/2
fpr_std_half_load_touch = half_scores_touch[0]
tpr_std_half_load_touch = half_scores_touch[1]

# 8) Scores on STD = 1/4
fpr_std_fourth_load_touch = half_scores_touch[2]
tpr_std_fourth_load_touch = half_scores_touch[3]
# 6) Scores on STD = 1/8
fpr_std_eight_load_touch = half_scores_touch[4]
tpr_std_eight_load_touch = half_scores_touch[5]

# 10) Scores on STD = 1/16
fpr_std_sixteen_load_touch = half_scores_touch[6]
tpr_std_sixteen_load_touch = half_scores_touch[7]

# 11) Scores on STD = 1/32
fpr_std_thirty_two_load_touch = half_scores_touch[8]
tpr_std_thirty_two_load_touch = half_scores_touch[9]


'''
------------------------------------------------------------------------ROC curve on Touch dataset-------------------------------------------------------------------------------
'''

plt.figure(figsize=(8,6), dpi=60)
plt.rcParams['legend.fontsize']=15
#plt.rcParams.update({'figure.dpi': '500'})
plt.rcParams['font.size'] = 20
plt.plot(fpr_std_1_load_touch, tpr_std_1_load_touch, color='magenta', label=' $\sigma$ = 1 (auc=%0.3f)' %auc(fpr_std_1_load_touch, tpr_std_1_load_touch), linewidth=5)
plt.plot(fpr_std_2_load_touch, tpr_std_2_load_touch, color='green', label=' $\sigma$ = 2 (auc=%0.3f)' %auc(fpr_std_2_load_touch, tpr_std_2_load_touch), linewidth=2.5)
plt.plot(fpr_std_4_load_touch, tpr_std_4_load_touch, color='red', label= ' $\sigma$ = 4 (auc=%0.3f)' %auc(fpr_std_4_load_touch, tpr_std_4_load_touch), linewidth=2.5)
plt.plot(fpr_std_8_load_touch, tpr_std_8_load_touch, color='blue', label= ' $\sigma$ = 8 (auc=%0.3f)' %auc(fpr_std_8_load_touch, tpr_std_8_load_touch), linewidth=2.5)
plt.plot(fpr_std_16_load_touch, tpr_std_16_load_touch, color='cyan', label= ' $\sigma$ = 16 (auc=%0.3f)' %auc(fpr_std_16_load_touch, tpr_std_16_load_touch), linewidth=2.5)
plt.plot(fpr_std_32_load_touch, tpr_std_32_load_touch, color='orange', label= ' $\sigma$ = 32 (auc=%0.3f)' %auc(fpr_std_32_load_touch, tpr_std_32_load_touch), linewidth=2.5)


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve by varying standard deviation $\sigma$. \n Dataset: HMOG. $\sigma$ = [1, 2, 4, 8, 16, 32]')
plt.legend()
plt.show()

plt.figure(figsize=(8,6), dpi=60)
plt.rcParams['legend.fontsize']=15
#plt.rcParams.update({'figure.dpi': '500'})
plt.rcParams['font.size'] = 20
plt.plot(fpr_std_1_load_touch, tpr_std_1_load_touch, color='magenta', label=' $\sigma$ = 1 (auc=%0.3f)' %auc(fpr_std_1_load_touch, tpr_std_1_load_touch), linewidth=5)
plt.plot(fpr_std_half_load_touch, tpr_std_half_load_touch, color='teal', label=' $\sigma$ = 1/2 (auc=%0.3f)' %auc(fpr_std_half_load_touch, tpr_std_half_load_touch), linewidth=2.5)
plt.plot(fpr_std_fourth_load_touch, tpr_std_fourth_load_touch, color='olive', label= ' $\sigma$= 1/4 (auc=%0.3f)' %auc(fpr_std_fourth_load_touch, tpr_std_fourth_load_touch), linewidth=2.5)
plt.plot(fpr_std_eight_load_touch, tpr_std_eight_load_touch, color='maroon', label= ' $\sigma$= 1/8 (auc=%0.3f)' %auc(fpr_std_eight_load_touch, tpr_std_eight_load_touch), linewidth=2.5)
plt.plot(fpr_std_sixteen_load_touch, tpr_std_sixteen_load_touch, color='indigo', label= ' $\sigma$= 1/16 (auc=%0.3f)' %auc(fpr_std_sixteen_load_touch, tpr_std_sixteen_load_touch), linewidth=2.5)
plt.plot(fpr_std_thirty_two_load_touch, tpr_std_thirty_two_load_touch, color='#36454F', label= ' $\sigma$= 1/32 (auc=%0.3f)' %auc(fpr_std_thirty_two_load_touch, tpr_std_thirty_two_load_touch), linewidth=2.5)


plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('ROC curve by varying standard deviation $\sigma$. \n Dataset: HMOG. $\sigma$ = [1, 1/2, 1/4, 1/8, 1/16, 1/32]')
plt.legend()
plt.show()


