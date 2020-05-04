file = "CE1_BMYK_IIM_SCI_N_20081204035456_20081204060237_4456_A.2C"
import pds3
lbl = PDS3Label(file)
r = lbl['RECORD_LENGTH']
shape = lbl['IMAGE']['LINES'], lbl['IMAGE']['LINE_SAMPLES']
print(lbl)