import argparse
from numpy import array

historical_path = None
hindcast_path = None
observation_path = None
residual_path = None
scenario_path = None
assi_path = None
aviso_path = None
pi_path = None
ohc_path = None
start_year = None
end_year = None
hist_start_years = None
hist_end_years = None
start_month_hind = None
ensemble_member = None
model_specifics_hist = None
model_specifics_hind = None
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
hist_mod = None
scenario_mod = None
data_path = None
hind_length = None
ensemble_member_hist = None
variable = None
region = None
data_path = None


def set_args():
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument(
        "--observation_path",
        type=str,
        default="/pool/data/ICDC/ocean/hadisst1/DATA/HadISST_sst.nc",
    )
    arg_parser.add_argument(
        "--aviso_path", type=str, default="/pool/data/ICDC/ocean/aviso_ssh/DATA/"
    )
    arg_parser.add_argument(
        "--ohc_path",
        type=str,
        default="/work/uo1075/u241265/obs/ohc/IAP_ohc700m_mm_1960_2016.nc",
    )
    arg_parser.add_argument("--hist_name", type=str, default="_historical_r")
    arg_parser.add_argument("--hind_name", type=str, default="_dcppA-hindcast_s")
    arg_parser.add_argument("--data_path", type=str, default="./tmp/")
    arg_parser.add_argument("--ssh_mod", type=str, default="v20190821/")
    arg_parser.add_argument("--plot_path", type=str, default="./plots/")
    arg_parser.add_argument("--lead_year", type=int, default=1)
    arg_parser.add_argument("--lead_years", type=str)
    arg_parser.add_argument("--scenario", type=str, default="ssp245")
    arg_parser.add_argument("--region", type=str, default="global")
    arg_parser.add_argument("--start_year", type=int, default=1970)
    arg_parser.add_argument("--end_year", type=int, default=2018)

    model_parsers = arg_parser.add_subparsers(help="model types")

    MIROC_parser = model_parsers.add_parser("MIROC6")

    MIROC_parser.add_argument(
        "--historical_path",
        type=str,
        default="/pool/data/CMIP6/data/CMIP/MIROC/MIROC6/historical/r",
    )
    MIROC_parser.add_argument(
        "--hindcast_path",
        type=str,
        default="/pool/data/CMIP6/data/DCPP/MIROC/MIROC6/dcppA-hindcast/s",
    )
    MIROC_parser.add_argument(
        "--scenario_path",
        type=str,
        default="/pool/data/CMIP6/data/ScenarioMIP/MIROC/MIROC6/ssp245/r",
    )
    MIROC_parser.add_argument(
        "--assi_path",
        type=str,
        default="/pool/data/CMIP6/data/DCPP/MIROC/MIROC6/dcppA-assim/r",
    )
    MIROC_parser.add_argument("--hind_mod", type=str, default="v20190821")
    MIROC_parser.add_argument(
        "--scenario_mod", type=str, default="v20190627"
    )  # 2:v20200623
    MIROC_parser.add_argument(
        "--hist_mod", type=str, default="v20181212"
    )  # 2:v20210901
    MIROC_parser.add_argument(
        "--hist_start_years", type=list, default=[1850, 1950, 2015]
    )
    MIROC_parser.add_argument("--hist_end_years", type=list, default=[1949, 2014, 2100])
    MIROC_parser.add_argument("--start_month_hind", type=str, default="11")
    MIROC_parser.add_argument("--ensemble_member", type=int, default=10)
    MIROC_parser.add_argument("--ensemble_member_hist", type=int, default=20)
    MIROC_parser.add_argument("--hind_length", type=int, default=10)
    MIROC_parser.add_argument("--model_specifics_hist", type=str, default="MIROC6")
    MIROC_parser.add_argument("--model_specifics_hind", type=str, default="MIROC6")
    MIROC_parser.add_argument("--variable", type=str, default="tos")
    MIROC_parser.add_argument("--assi", type=str, default=True)
    MIROC_parser.add_argument(
        "--pi_path",
        type=str,
        default="/work/uo1075/u301617/HiWi_Vimal/Code/tmp/MIROC6/pi_control_thetao_320001_399912.nc",
    )

    MPI_parser = model_parsers.add_parser("MPI")

    MPI_parser.add_argument(
        "--historical_path",
        type=str,
        default="/pool/data/CMIP6/data/CMIP/MPI-M/MPI-ESM1-2-LR/historical/r",
    )
    MPI_parser.add_argument(
        "--hindcast_path",
        type=str,
        default="/pool/data/CMIP6/data/DCPP/MPI-M/MPI-ESM1-2-HR/dcppA-hindcast/s",
    )
    MPI_parser.add_argument(
        "--scenario_path",
        type=str,
        default="/pool/data/CMIP6/data/ScenarioMIP/MPI-M/MPI-ESM1-2-LR/ssp245/r",
    )
    MPI_parser.add_argument(
        "--assi_path",
        type=str,
        default="/work/uo1075/u301617/HiWi_Vimal/Code/tmp/MPI-ESM1-2-HR/assimilation_MPI-ESM1-2-HR_1958_2017.nc",
    )
    MPI_parser.add_argument(
        "--pi_path",
        type=str,
        default="/work/uo1075/u301617/HiWi_Vimal/Code/tmp/MPI-ESM1-2-HR/picontrol_MPI_thetao_1850_2349.nc",
    )
    MPI_parser.add_argument(
        "--hind_mod", type=str, default="v20190917"
    )  # v20200909 bei thetao
    MPI_parser.add_argument(
        "--scenario_mod", type=str, default="v20190710"
    )  # 2:v20210901
    MPI_parser.add_argument("--hist_mod", type=str, default="v20190710")  # 2:v20210901
    MPI_parser.add_argument(
        "--hist_start_years",
        type=list,
        default=[1850, 1870, 1890, 1910, 1930, 1950, 1970, 1990, 2010, 2015],
    )
    MPI_parser.add_argument(
        "--hist_end_years",
        type=list,
        default=[1869, 1889, 1909, 1929, 1949, 1969, 1989, 2009, 2014, 2034],
    )
    MPI_parser.add_argument("--start_month_hind", type=str, default="11")
    MPI_parser.add_argument("--ensemble_member", type=int, default=5)
    MPI_parser.add_argument("--ensemble_member_hist", type=int, default=20)
    MPI_parser.add_argument("--hind_length", type=int, default=10)
    MPI_parser.add_argument("--model_specifics_hist", type=str, default="MPI-ESM1-2-LR")
    MPI_parser.add_argument("--model_specifics_hind", type=str, default="MPI-ESM1-2-HR")
    MPI_parser.add_argument("--variable", type=str, default="tos")
    MPI_parser.add_argument("--assi", type=str, default=True)

    BCC_parser = model_parsers.add_parser("BCC")

    BCC_parser.add_argument(
        "--historical_path",
        type=str,
        default="/pool/data/CMIP6/data/CMIP/BCC/BCC-CSM2-MR/historical/r",
    )
    BCC_parser.add_argument(
        "--hindcast_path",
        type=str,
        default="/pool/data/CMIP6/data/DCPP/BCC/BCC-CSM2-MR/dcppA-hindcast/s",
    )
    BCC_parser.add_argument(
        "--scenario_path",
        type=str,
        default="/pool/data/CMIP6/data/ScenarioMIP/BCC/BCC-CSM2-MR/ssp245/r",
    )
    BCC_parser.add_argument(
        "--assi_path",
        type=str,
        default="/work/uo1075/u301617/HiWi_Vimal/Code/tmp/MPI-ESM1-2-HR/assimilation_MPI-ESM1-2-HR_1958_2017.nc",
    )
    BCC_parser.add_argument(
        "--pi_path",
        type=str,
        default="/work/uo1075/u301617/HiWi_Vimal/Code/tmp/BCC-CSM2-MR/picontrol_thetao_BCC_185001_244912.nc",
    )
    BCC_parser.add_argument(
        "--hind_mod", type=str, default="v20200117"
    )  # v20190917 bei tos
    BCC_parser.add_argument(
        "--scenario_mod", type=str, default="v20190319"
    )  # 2:v20210901
    BCC_parser.add_argument("--hist_mod", type=str, default="v20181126")  # 2:v20210901
    BCC_parser.add_argument(
        "--hist_start_years",
        type=list,
        default=[
            1850,
            1860,
            1870,
            1880,
            1890,
            1900,
            1910,
            1920,
            1930,
            1940,
            1950,
            1960,
            1970,
            1980,
            1990,
            2000,
            2010,
            2010,
            2015,
            2025,
        ],
    )
    BCC_parser.add_argument(
        "--hist_end_years",
        type=list,
        default=[
            1859,
            1869,
            1879,
            1889,
            1899,
            1909,
            1919,
            1929,
            1939,
            1949,
            1959,
            1969,
            1979,
            1989,
            1999,
            2009,
            2014,
            2024,
            2034,
        ],
    )
    BCC_parser.add_argument("--start_month_hind", type=str, default="01")
    BCC_parser.add_argument("--assi", type=str, default=False)
    BCC_parser.add_argument("--ensemble_member", type=int, default=8)
    BCC_parser.add_argument("--ensemble_member_hist", type=int, default=3)
    BCC_parser.add_argument("--hind_length", type=int, default=9)
    BCC_parser.add_argument("--model_specifics_hist", type=str, default="BCC-CSM2-MR")
    BCC_parser.add_argument("--model_specifics_hind", type=str, default="BCC-CSM2-MR")
    BCC_parser.add_argument("--variable", type=str, default="thetao")

    # arg_parser.add_argument('--historical_path', type=str, default='/pool/data/CMIP6/data/CMIP/CCCma/CanESM5/historical/r')
    # arg_parser.add_argument('--hindcast_path', type=str, default='/pool/data/CMIP6/data/DCPP/CCCma/CanESM5/dcppA-hindcast/s')
    # arg_parser.add_argument('--scenario_path', type=str, default='/pool/data/CMIP6/data/ScenarioMIP/CCCma/CanESM5/ssp245/r')
    # arg_parser.add_argument('--assi_path', type=str, default='/pool/data/CMIP6/data/DCPP/CCCma/CanESM5/dcppA-assim/r')
    # arg_parser.add_argument('--hind_mod', type=str, default='v20190429')
    # arg_parser.add_argument('--scenario_mod', type=str, default='v20190429') #2:v20190429
    # arg_parser.add_argument('--hist_mod', type=str, default='v20190429') #2:v20190429
    # arg_parser.add_argument('--hist_start_years', type=list, default=[1850, 1861, 1871, 1881, 1891, 1901, 1911, 1921, 1931, 1941, 1951, 1961, 1971, 1981, 1991, 2001, 2011, 2015, 2021])
    # arg_parser.add_argument('--hist_end_years', type=list, default=[1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2014, 2020, 2030])
    # arg_parser.add_argument('--start_year', type=int, default=1966)
    # arg_parser.add_argument('--end_year', type=int, default=2008)
    # arg_parser.add_argument('--start_month_hind', type=str, default='01')
    # arg_parser.add_argument('--ensemble_member', type=int, default=2)
    # arg_parser.add_argument('--ensemble_member_hist', type=int, default=25)
    # arg_parser.add_argument('--hind_length', type=int, default=9)
    # arg_parser.add_argument('--model_specifics_hist', type=str, default='CanESM5')
    # arg_parser.add_argument('--model_specifics_hind', type=str, default='CanESM5')
    # arg_parser.add_argument('--variable', type=str, default='thetao')

    # arg_parser.add_argument('--historical_path', type=str, default='/pool/data/CMIP6/data/CMIP/NCC/NorCPM1/historical/r')
    # arg_parser.add_argument('--hindcast_path', type=str, default='/pool/data/CMIP6/data/DCPP/NCC/NorCPM1/dcppA-hindcast/s')
    # arg_parser.add_argument('--scenario_path', type=str, default='/pool/data/CMIP6/data/ScenarioMIP/NCC/NorCPM1/ssp245/r')
    # arg_parser.add_argument('--assi_path', type=str, default='/pool/data/CMIP6/data/DCPP/NCC/NorCPM1/dcppA-assim/r')
    # arg_parser.add_argument('--hind_mod', type=str, default='v20190914')
    # arg_parser.add_argument('--scenario_mod', type=str, default='v20190429') #2:v20190429
    # arg_parser.add_argument('--hist_mod', type=str, default='v20200724') #2:v20190429
    # arg_parser.add_argument('--hist_start_years', type=list, default=[1850])
    # arg_parser.add_argument('--hist_end_years', type=list, default=[2014])
    # arg_parser.add_argument('--start_year', type=int, default=1970)
    # arg_parser.add_argument('--end_year', type=int, default=2013)
    # arg_parser.add_argument('--start_month_hind', type=str, default='10')
    # arg_parser.add_argument('--ensemble_member', type=int, default=3)
    # arg_parser.add_argument('--ensemble_member_hist', type=int, default=25)
    # arg_parser.add_argument('--hind_length', type=int, default=10)
    # arg_parser.add_argument('--model_specifics_hist', type=str, default='NorCPM1')
    # arg_parser.add_argument('--model_specifics_hind', type=str, default='NorCPM1')
    # arg_parser.add_argument('--variable', type=str, default='tos')

    # arg_parser.add_argument('--historical_path', type=str, default='/pool/data/CMIP6/data/CMIP/MRI/MRI-ESM2-0/historical/r')
    # arg_parser.add_argument('--hindcast_path', type=str, default='/pool/data/CMIP6/data/DCPP/MRI/MRI-ESM2-0/dcppA-hindcast/s')
    # arg_parser.add_argument('--scenario_path', type=str, default='/pool/data/CMIP6/data/ScenarioMIP/MRI/MRI-ESM2-0/ssp245/r')
    # arg_parser.add_argument('--assi_path', type=str, default='/pool/data/CMIP6/data/DCPP/MRI/MRI-ESM2-0/dcppA-assim/r')
    # arg_parser.add_argument('--hind_mod', type=str, default='v20201016/')
    # arg_parser.add_argument('--scenario_mod', type=str, default='v20190904/') #2:v20200623
    # arg_parser.add_argument('--hist_mod', type=str, default='v20190904/') #2:v20210901
    # arg_parser.add_argument('--hist_start_years', type=list, default=[1850, 2015])
    # arg_parser.add_argument('--hist_end_years', type=list, default=[2014, 2100])
    # arg_parser.add_argument('--start_year', type=int, default=1960)
    # arg_parser.add_argument('--end_year', type=int, default=2011)
    # arg_parser.add_argument('--start_month_hind', type=str, default='11')
    # arg_parser.add_argument('--ensemble_member', type=int, default=10)
    # arg_parser.add_argument('--ensemble_member_hist', type=int, default=1)
    # arg_parser.add_argument('--hind_length', type=int, default=5)
    # arg_parser.add_argument('--model_specifics_hist', type=str, default='MRI-ESM2-0')
    # arg_parser.add_argument('--model_specifics_hind', type=str, default='MRI-ESM2-0')
    # arg_parser.add_argument('--variable', type=str, default='tos')

    args = arg_parser.parse_args()

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
    global model_specifics_hist
    global model_specifics_hind
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
    global hist_mod
    global scenario_mod
    global data_path
    global hind_length
    global ensemble_member_hist
    global variable
    global region
    global pi_path
    global data_path

    historical_path = args.historical_path
    hindcast_path = args.hindcast_path
    observation_path = args.observation_path
    residual_path = (
        f"{args.data_path}/{args.model_specifics_hind}/{args.region}/residuals/residual"
    )
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
    ensemble_member_hist = args.ensemble_member_hist
    model_specifics_hist = args.model_specifics_hist
    model_specifics_hind = args.model_specifics_hind
    plot_path = f"{args.plot_path}{args.model_specifics_hind}/{args.region}/"
    tmp_path = f"{args.data_path}/{args.model_specifics_hind}/{args.region}/"
    data_path = args.data_path + "/tmp/"
    lead_years = args.lead_years
    scenario = args.scenario
    lead_year = args.lead_year
    hist_name = args.hist_name
    hind_name = args.hind_name
    hind_mod = args.hind_mod
    ssh_mod = args.ssh_mod
    hist_mod = args.hist_mod
    scenario_mod = args.scenario_mod
    data_path = args.data_path
    hind_length = args.hind_length
    variable = args.variable
    region = args.region
    pi_path = args.pi_path
    if region == "global":
        lonlats = [0, 360, -90, 90]
    elif region == "indian_ocean":
        lonlats = [50, 110, -30, 26]
    elif region == "SPG":
        lonlats = [-60, -10, 50, 65]
