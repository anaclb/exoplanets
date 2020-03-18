import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from astropy.units import jupiterMass, jupiterRad, solMass, solRad, AU

def read_file(file, removeNaN=False, params=0,standard=False):
    database = np.genfromtxt(file_EU, comments="--", skip_header=4,
                            dtype=None, delimiter="\t", encoding=None)
    df = pd.DataFrame(database)
    with open(file) as f:
        columns = f.readline()
    columns = columns.replace("\n", "")
    columns = columns.split('\t')
    n = len(columns)
    df = df.iloc[:,:n]
    df.columns = columns
    df = pd.DataFrame.set_index(df,keys='obj_id_catname')
    if params != 0:
        df = df[params]
    if removeNaN == True:
        df = pd.DataFrame.dropna(df,axis=0, how='any')
    return df

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
