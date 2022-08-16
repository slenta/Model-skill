
#script to concatenate all historical files into one for each ensemble member

import config as cfg
from preprocessing import concat_hist
import os

import tempfile
tempfile.tempdir ='/mnt/lustre02/work/uo1075/u301617/tmp'

cfg.set_args()

hist_tmp_path = cfg.tmp_path + 'hist/'

if not os.path.exists(cfg.plot_path):
    os.makedirs(cfg.plot_path)
if not os.path.exists(cfg.tmp_path):
    os.makedirs(cfg.tmp_path)
if not os.path.exists(hist_tmp_path):
    os.makedirs(hist_tmp_path)


#concatenate and save historical simulations in a single file
concate = concat_hist(cfg.hist_start_years, cfg.hist_end_years, 10, cfg.scenario_path, cfg.scenario)
concate.concat()


#concatenate and save aviro data
#first concatenate daily data to monthly and save at tmp
#for i in range(1993, 2019):
#    print(i)
#    for j in range(1, 13):
#        month = str(j).zfill(2)
#        con = concat(cfg.aviso_path + str(i) + '/' + month + '/', 'Aviso_Ssh', variable='adt', start=str(i) + '-' + month, end=str(i + 1) + '-' + month)
#        con.concat()
#then: concatenate all monthly data over the whole timeframe
#con = concat(cfg.tmp_path + 'Aviro_Ssh/', name = 'Aviro_Ssh_full', variable='var', start='2000-01', end='2001-12')
#con.concat()
