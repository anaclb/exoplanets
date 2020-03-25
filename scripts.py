import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from astropy.units import jupiterMass, jupiterRad, a, day, earthRad, earthMass, solMass, solRad, AU
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
scaler_norm = MinMaxScaler()
scaler_stnd = StandardScaler()

def read_file(file_exo, params=0, removeNaN=True):

    "Reads exoplanet file, returning a DataFrame with the asked parameters\
    \for and without NaN values. file: file path (.rdb in this case), params: list with parameters wanted\
    \removeNaN: True/False, removes planet from the DataFrame if any value in the parameters asked are NaN"

    database = np.genfromtxt(file_exo, comments="--", skip_header=4,
                            dtype=None, delimiter="\t", encoding=None)
    df_exo = pd.DataFrame(database)
    with open(file_exo) as f:
        columns = f.readline()
    columns = columns.replace("\n", "")
    columns = columns.split('\t')
    df_exo.columns = columns
    df_exo = pd.DataFrame.set_index(df_exo,keys='obj_id_catname')
    if params != 0:
        df_exo = df_exo[params]
    if removeNaN == True:
        df_exo = pd.DataFrame.dropna(df_exo,axis=0, how='any')
    return df_exo

def sol_pd(file_solar,params):

    "returns DataFrame with asked parameters for the solar system planets\
    \file_solar: file path (.csv in this case), params: list of parameters wanted"

    solar=pd.DataFrame.transpose(pd.read_csv(file_solar, index_col=0))
    solar['temp_eq'] = solar.T_C+274.15
    solar['obj_phys_mass_mjup'] = solar.M*earthMass.to(jupiterMass)
    solar['obj_phys_radius_rjup'] = solar.D/2*earthRad.to(jupiterRad)
    solar['obj_orb_period_day'] = solar.P*a.to(day)
    solar['obj_orb_ecc'] = solar.e*0.0167
    solar['obj_orb_a_au'] = solar.a
    solar = solar[params]
    return solar

def exo_sol(file_exo,file_solar,params,stnd=False,norm=False,removeNaN=True):

    "Reads exoplanet and solar files, returns DataFrame containing both data for the wanted parameters,\
    \with different normalization options.\
    \file_exo: exoplanet file path (.rdb).\
    \file_solar: solar system planets file path (.csv),\
    \params: list of parameters wanted.\
    \stnd: True/False, True for data standardization, centers the distribution in 0 and scales it to unit variance.\
    \norm: True/False, True for normalization, scales the distribution's range to [0,1]."

    df_solar = sol_pd(file_solar, params)
    df_exo = read_file(file_exo, params, removeNaN)
    data = pd.concat([df_exo, df_solar])
    if stnd == True:
        data = scaler_stnd.fit_transform(data)
    if norm == True:
        data = scaler_norm.fit_transform(data)
    return data


def getTeqpl(Teffst, aR, ecc, A=0, f=1/4.): 
    return Teffst * (f * (1 - A))**(1 / 4.) * np.sqrt(1 / aR) / (1 - ecc**2)**(1/8.)

def add_temp_eq_dataset(dataset):
    semi_major_axis = dataset['obj_orb_a_au'] * AU.to('solRad')
    teq_planet = [getTeqpl(teff, a/rad, ecc)
                  for teff, a, rad, ecc,
                  in zip(dataset['obj_parent_phys_teff_k'], semi_major_axis,
                         dataset['obj_parent_phys_radius_rsun'], dataset['obj_orb_ecc'])]
    dataset.insert(2, 'temp_eq', teq_planet)
    return dataset

file_US="/home/bolacha/University/Project/code/data-example/all_data_US.rdb"
file_EU="/home/bolacha/University/Project/code/data-example/all_data_EU.rdb"
cat_solar="/home/bolacha/University/Project/code/data-example/solar_data.csv"
