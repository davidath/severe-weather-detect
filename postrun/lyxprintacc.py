#! /usr/bin/env python

import os
import re
import sys
import numpy as np

ls = os.listdir(sys.argv[1])
d = {'acc_tr': [],
     'acc_te': [],
     'nmi_tr': [],
     'nmi_te': [],
     'sil_tr': [],
     'sil_te': [],
     'chs_tr': [],
     'chs_te': []
    }
for num in sys.argv[3:]:
    for l in ls:
        if l == sys.argv[2] + 'test' + num + '.txt':
            f = open(sys.argv[1] + '/' + l, 'r')
            lines = f.readlines()
            for li in lines:
                if 'COND - ACC FULL' in li:
                    c = re.findall("\d+\.\d+", li)[1]
                    d['acc_te'].append(float(c)*100)
                if 'COND - NMI FULL' in li:
                    c = re.findall("\d+\.\d+", li)[1]
                    d['nmi_te'].append(float(c)*100)
                if 'COND - CHS FULL' in li:
                    c = re.findall("\d+\.\d+", li)[1]
                    d['chs_te'].append(float(c))
                if 'PX - ACC FULL' in li:
                    c = re.findall("\d+\.\d+", li)[1]
                    acc_te = float(c)*100
                if 'PX - NMI FULL' in li:
                    c = re.findall("\d+\.\d+", li)[1]
                    nmi_te = float(c)*100
                if 'PX - CHS FULL' in li:
                    c = re.findall("\d+\.\d+", li)[1]
                    chs_te = float(c)

mean_te_acc =  np.asarray(d['acc_te']).mean()
std_te_acc =  np.asarray(d['acc_te']).std()
mean_te_nmi = np.asarray(d['nmi_te']).mean()
std_te_nmi = np.asarray(d['nmi_te']).std()


mean_te_chs =  np.asarray(d['chs_te']).mean()
std_te_chs =  np.asarray(d['chs_te']).std()

ste_acc_te = "%.2f" % round(mean_te_acc, 2)
if round((mean_te_acc - acc_te), 2) > 0:
    ste_diff_acc_te = "+%.2f" % round((mean_te_acc - acc_te), 2)
else:
    ste_diff_acc_te = "%.2f" % round((mean_te_acc - acc_te), 2)

ste_std_acc_te =  "%.2f" % std_te_acc

ste_nmi_te = "%.2f" % round(mean_te_nmi, 2)
if round((mean_te_nmi - nmi_te), 2) > 0:
    ste_diff_nmi_te = "+%.2f" % round((mean_te_nmi - nmi_te), 2)
else:
    ste_diff_nmi_te = "%.2f" % round((mean_te_nmi - nmi_te), 2)

ste_std_nmi_te =  "%.2f" % std_te_nmi

ste_chs_te = "%.1f" % round(mean_te_chs, 1)
te_percent = ((mean_te_chs - chs_te) / float(chs_te))*100
ste_diff_chs_te = "%.1f" % te_percent


#  print str_acc_tr + ' (' + str_diff_acc_tr + ')' + '\t' + str_std_acc_tr + '\t' + str_nmi_tr + '(' + str_diff_nmi_tr + ')' + '\t' + str_std_nmi_tr + '\t' + ste_acc_te + ' (' + ste_diff_acc_te + ')' + '\t' + ste_std_acc_te + '\t' + ste_nmi_te + '(' + ste_diff_nmi_te + ')' + '\t' + ste_std_nmi_te + '\t'
#  print str_acc_tr + ' (' + str_diff_acc_tr + ')' + '\t' + str_nmi_tr + '(' + str_diff_nmi_tr + ')' + '\t' + ste_acc_te + ' (' + ste_diff_acc_te + ')' + '\t' + ste_nmi_te + '(' + ste_diff_nmi_te + ')'
#  print ste_acc_te + ' (' + ste_diff_acc_te + ')' + '\t' + ste_nmi_te + '(' + ste_diff_nmi_te + ')'

#  print  str_acc_tr + ' (' + str_diff_acc_tr + ')' + '\t' + str_nmi_tr + '(' + str_diff_nmi_tr + ')'
#  print  ste_acc_te + ' (' + ste_diff_acc_te + ')' + '\t' + ste_nmi_te + ' (' + ste_diff_nmi_te + ')'

# print ste_acc_te + ' (' + ste_diff_acc_te + ')' + '\t' + ste_nmi_te + ' (' + ste_diff_nmi_te + ')' + '\t' + ste_chs_te + ' (' + ste_diff_chs_te + ')'

print ste_acc_te + ' (' + ste_diff_acc_te + ')' + '\t' + ste_nmi_te + ' (' + ste_diff_nmi_te + ')'
