#! /usr/bin/env python

import os
import re
import sys

ls = os.listdir(sys.argv[1])
for l in ls:
    if l == sys.argv[2]:
        f = open(sys.argv[1] + '/' + l, 'r')
        lines = f.readlines()
        ret = ""
        for li in lines:
            if 'PX - ACC FULL' in li:
                c = re.findall("\d+\.\d+", li)[1]
                acc_te = c
            if 'PX - NMI FULL' in li:
                c = re.findall("\d+\.\d+", li)[1]
                nmi_te = c
            if 'PX - CHS FULL' in li:
                c = re.findall("\d+\.\d+", li)[1]
                chs_te = c
        print acc_te + '\t' + nmi_te + '\t'  + chs_te
        for li in lines:
            if 'COND - ACC FULL' in li:
                c = re.findall("\d+\.\d+", li)[1]
                acc_te = c
            if 'COND - NMI FULL' in li:
                c = re.findall("\d+\.\d+", li)[1]
                nmi_te = c
            if 'COND - CHS FULL' in li:
                c = re.findall("\d+\.\d+", li)[1]
                chs_te = c
        print acc_te + '\t' + nmi_te + '\t'  + chs_te
