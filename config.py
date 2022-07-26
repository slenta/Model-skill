import argparse
from numpy import array

name = None
historical_path = None
hindcast_path = None
observation_path = None
residual_path = None
scenario_path = None
assi_path = None
aviso_path = None
ohc_path = None
start_year = None
end_year = None
hist_start_years = None
hist_end_years = None
start_month_hind = None
ensemble_member = None
model_specifics = None
tmp_path = None
lead_years = None
scenario = None
lonlats = None
lead_year = None
hist_name = None
hind_name = None
hind_mod = None
ssh_mod = None
plot_path = None


def set_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--name', type=str, default='ModelNameEnsembleJahr')
    arg_parser.add_argument('--historical_path', type=str, default='/pool/data/CMIP6/data/CMIP/MIROC/MIROC6/historical/r')
    arg_parser.add_argument('--hindcast_path', type=str, default='/pool/data/CMIP6/data/DCPP/MIROC/MIROC6/dcppA-hindcast/s')
    arg_parser.add_argument('--observation_path', type=str, default='/pool/data/ICDC/ocean/hadisst1/DATA/HadISST_sst.nc')
    arg_parser.add_argument('--residual_path', type=str, default='/work/uo1075/u301617/HiWi_Vimal/Code/tmp/residuals/residual')
    arg_parser.add_argument('--scenario_path', type=str, default='/pool/data/CMIP6/data/ScenarioMIP/MIROC/MIROC6/ssp245/r')
    arg_parser.add_argument('--assi_path', type=str, default='/pool/data/CMIP6/data/DCPP/MIROC/MIROC6/dcppA-assim/r')
    arg_parser.add_argument('--aviso_path', type=str, default='/pool/data/ICDC/ocean/aviso_ssh/DATA/')
    arg_parser.add_argument('--ohc_path', type=str, default='/work/uo1075/u241265/obs/ohc/IAP_ohc700m_mm_1960_2016.nc')
    arg_parser.add_argument('--hist_name', type=str, default='_historical_r')
    arg_parser.add_argument('--hind_name', type=str, default='_dcppA-hindcast_s')
    arg_parser.add_argument('--hind_mod', type=str, default='v20190821/')
    arg_parser.add_argument('--ssh_mod', type=str, default='v20190821/')    
    arg_parser.add_argument('--hist_start_years', type=list, default=[1850, 1950, 2015])
    arg_parser.add_argument('--hist_end_years', type=list, default=[1949, 2014, 2100])
    arg_parser.add_argument('--start_year', type=int, default=1960)
    arg_parser.add_argument('--end_year', type=int, default=2011)
    arg_parser.add_argument('--start_month_hind', type=str, default='11')
    arg_parser.add_argument('--ensemble_member', type=list, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    arg_parser.add_argument('--model_specifics', type=str, default='MIROC6')
    arg_parser.add_argument('--tmp_path', type=str, default='./tmp/')
    arg_parser.add_argument('--plot_path', type=str, default='./plots/')
    arg_parser.add_argument('--lead_year', type=int, default=1)
    arg_parser.add_argument('--scenario', type=str, default='ssp245')
    arg_parser.add_argument('--lonlats', type=list, default=[0, 360, -90, 90])
    arg_parser.add_argument('--lead_years', type=str)



    args = arg_parser.parse_args()

    global name
    global historical_path
    global hindcast_path
    global observation_path
    global residual_path
    global scenario_path
    global assi_path
    global aviso_path
    global ohc_path
    global start_year
    global end_year
    global hist_start_years
    global hist_end_years
    global start_month_hind
    global ensemble_member
    global model_specifics
    global tmp_path
    global lead_years
    global scenario
    global lonlats
    global lead_year
    global hist_name
    global hind_name
    global hind_mod
    global ssh_mod
    global plot_path


    name = args.name
    historical_path = args.historical_path
    hindcast_path = args.hindcast_path
    observation_path = args.observation_path
    residual_path = args.residual_path
    scenario_path = args.scenario_path
    assi_path = args.assi_path
    aviso_path = args.aviso_path
    ohc_path = args.ohc_path
    start_year = args.start_year
    end_year = args.end_year
    hist_start_years = args.hist_start_years
    hist_end_years = args.hist_end_years
    start_month_hind = args.start_month_hind
    ensemble_member = args.ensemble_member
    model_specifics = args.model_specifics
    plot_path = args.plot_path + args.model_specifics + '/'
    tmp_path = args.tmp_path
    lead_years = args.lead_years
    scenario = args.scenario
    lonlats = args.lonlats
    lead_year = args.lead_year
    hist_name = args.hist_name
    hind_name = args.hind_name
    hind_mod = args.hind_mod
    ssh_mod = args.ssh_mod

