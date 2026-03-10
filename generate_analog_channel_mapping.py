#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 15:24:19 2026

@author: christophpfeiffer
"""
import json
from mne._fiff.constants import FIFF

eog=dict(kind=FIFF.FIFFV_EOG_CH, unit=FIFF.FIFF_UNIT_V)
emg=dict(kind=FIFF.FIFFV_EMG_CH, unit=FIFF.FIFF_UNIT_V)
ecg=dict(kind=FIFF.FIFFV_ECG_CH, unit=FIFF.FIFF_UNIT_V)
resp=dict(kind=FIFF.FIFFV_RESP_CH, unit=FIFF.FIFF_UNIT_V)
bio=dict(kind=FIFF.FIFFV_BIO_CH, unit=FIFF.FIFF_UNIT_V)
misc=dict(kind=FIFF.FIFFV_MISC_CH, unit=FIFF.FIFF_UNIT_V)


mapping ={
    "ai1":  {"newname": "Acc2X", "type": misc}, #Card1
    "ai2":  {"newname": "Acc2Y", "type": misc},
    "ai3":  {"newname": "Acc2Z", "tptypeye": misc},
    #"ai4":  {"newname": "-", "tpye": misc},
    "ai5":  {"newname": "ECG", "type": ecg}, #Card2
    "ai6":  {"newname": "EOG1", "type": eog},
    "ai7":  {"newname": "EOG2", "type": eog},
    "ai8":  {"newname": "RESP", "type": resp},
    "ai9":  {"newname": "Acc1X", "type": misc}, #Card3
    "ai10":  {"newname": "Acc1Y", "type": misc},
    "ai11":  {"newname": "Acc1Z", "type": misc},
    #"ai12":  {"newname": "-", "tpye": misc},
    #"ai13":  {"newname": "-", "tpye": misc}, #Card4
    #"ai14":  {"newname": "-", "tpye": misc},
    "ai15":  {"newname": "EyeLX", "type": misc},
    "ai16":  {"newname": "EyeLY", "type": misc},
    "ai17":  {"newname": "EyeLP", "type": misc}, #Card5
    "ai18":  {"newname": "EyeRX", "type": misc},
    "ai19":  {"newname": "EyeRY", "type": misc},
    "ai20":  {"newname": "EyeRP", "type": misc},
    #"ai21":  {"newname": "-", "tpye": misc}, #Card6
    #"ai22":  {"newname": "-", "tpye": misc},
    #"ai23":  {"newname": "-", "tpye": misc},
    #"ai24":  {"newname": "-", "tpye": misc}
    }

with open('analog_channel_mapping.json', "w") as f:
        json.dump(mapping, f, indent=4)