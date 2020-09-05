import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from astropy.units import jupiterMass, jupiterRad, a, day, earthRad, earthMass, solMass, solRad, AU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

def read_file(file_exo, params=0):

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
    df_exo = pd.DataFrame.dropna(df_exo,axis=0, how='any')
    return df_exo

def sol_pd(file_solar,params):

    "returns DataFrame with asked parameters for the solar system planets\
    \file_solar: file path (.csv in this case), params: list of parameters wanted"

    solar=pd.DataFrame.transpose(pd.read_csv(file_solar, index_col=0))
    solar['temp_eq'] = solar.T_C+274.15
    solar['obj_phys_mass_mjup'] = solar.M*earthMass.to(jupiterMass)
    solar['obj_phys_radius_rjup'] = solar.D*earthRad.to(jupiterRad)
    solar['obj_orb_period_day'] = solar.P*a.to(day)
    solar['obj_orb_ecc'] = solar.e*0.0167
    solar['obj_orb_a_au'] = solar.a
    solar['obj_parent_phys_teff_k'] = pd.DataFrame(np.ones(len(solar))*5778, index = solar.index)
    solar['obj_parent_phys_radius_rsun'] = pd.DataFrame(np.ones(len(solar)), index = solar.index)
    solar['obj_parent_phys_mass_msun'] = pd.DataFrame(np.ones(len(solar)), index = solar.index)
    solar['obj_parent_phys_feh'] = pd.DataFrame(np.zeros(len(solar)), index = solar.index)
    solar = solar[params]
    return solar


def sol_pd2(file_solar,params):

    "returns DataFrame with asked parameters for the solar system planets\
    \file_solar: file path (.csv in this case), params: list of parameters wanted"

    solar=pd.DataFrame.transpose(pd.read_csv(file_solar, index_col=0))
    solar['temp_eq'] = solar.T_C+274.15
    solar['mass'] = solar.M*earthMass.to(jupiterMass)
    solar['radius'] = solar.D*earthRad.to(jupiterRad)
    solar['orbital_period'] = solar.P*a.to(day)
    solar['star_teff'] = pd.DataFrame(np.ones(len(solar))*5778, index = solar.index)
    solar['star_radius'] = pd.DataFrame(np.ones(len(solar)), index = solar.index)
    solar['star_mass'] = pd.DataFrame(np.ones(len(solar)), index = solar.index)
    solar['star_metallicity'] = pd.DataFrame(np.zeros(len(solar)), index = solar.index)
    solar = solar[params]
    return solar




def exo_sol(file_exo, file_solar, params):

    "Reads exoplanet and solar files, returns DataFrame containing both data for the wanted parameters,\
    \with different normalization options.\
    \file_exo: exoplanet file path (.rdb).\
    \file_solar: solar system planets file path (.csv),\
    \params: list of parameters wanted."

    df_solar = sol_pd(file_solar, params)
    df_exo = read_file(file_exo, params)
    data = pd.concat([df_exo, df_solar])

    data = data.drop(['PLUTO','MOON'])
    data = data.drop(['K2-22 b','K2-77 b'])


  #  data = data.drop(['Kepler-58 c'])

    #corrections to the database - updated values from NASA table at https://exoplanetarchive.ipac.caltech.edu/

    data.loc['Kepler-11 g'].obj_phys_mass_mjup = 0.0791*earthMass.to(jupiterMass)
    data.loc['Kepler-57 c'].obj_phys_mass_mjup = 5.68*earthMass.to(jupiterMass)
    data.loc['Kepler-59 c'].obj_phys_mass_mjup = 0.0082
    data.loc['Kepler-59 c'].obj_phys_radius_rjup = .196
    data.loc['Kepler-28 c'].obj_phys_mass_mjup = 10.9*earthMass.to(jupiterMass)
    data.loc['Kepler-28 c'].obj_phys_radius_rjup = 2.77*earthRad.to(jupiterRad)
    data.loc['Kepler-53 c'].obj_phys_mass_mjup = 35.5*earthMass.to(jupiterMass)
    data.loc['Kepler-53 c'].obj_phys_radius_rjup = 3.12*earthRad.to(jupiterRad)
    data.loc['Kepler-57 b'].obj_phys_mass_mjup = 118.1*earthMass.to(jupiterMass)
    data.loc['Kepler-57 b'].obj_phys_radius_rjup = 2.12*earthRad.to(jupiterRad)
    data.loc['Kepler-28 b'].obj_phys_mass_mjup = 8.8*earthMass.to(jupiterMass)
    data.loc['Kepler-28 b'].obj_phys_radius_rjup = 1.971*earthRad.to(jupiterRad)
    data.loc['Kepler-10 c'].obj_phys_mass_mjup = 7.37*earthMass.to(jupiterMass)
    data.loc['Kepler-10 c'].obj_phys_radius_rjup = 2.351*earthRad.to(jupiterRad)
    data.loc['Kepler-52 b'].obj_phys_mass_mjup = 79.6*earthMass.to(jupiterMass)
    data.loc['Kepler-52 b'].obj_phys_radius_rjup = 2.176*earthRad.to(jupiterRad)
    data.loc['Kepler-53 b'].obj_phys_mass_mjup = 103.1*earthMass.to(jupiterMass)
    data.loc['Kepler-53 b'].obj_phys_radius_rjup = 3.225*earthRad.to(jupiterRad)
    data.loc['Kepler-52 c'].obj_phys_mass_mjup = 62.9*earthMass.to(jupiterMass)
    data.loc['Kepler-52 c'].obj_phys_radius_rjup = 2.196*earthRad.to(jupiterRad)
    data.loc['Kepler-24 c'].obj_phys_mass_mjup = 33.6*earthMass.to(jupiterMass)
    data.loc['Kepler-24 c'].obj_phys_radius_rjup = 3.689*earthRad.to(jupiterRad)
    data.loc['K2-19 b'].obj_phys_mass_mjup = 32.4*earthMass.to(jupiterMass)
    data.loc['Kepler-58 b'].obj_phys_mass_mjup = 34.9*earthMass.to(jupiterMass)
    data.loc['Kepler-24 b'].obj_phys_mass_mjup = 33.3*earthMass.to(jupiterMass)
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

def add_lumi(data):
    data['luminosity'] = data.obj_parent_phys_radius_rsun**2*(data.obj_parent_phys_teff_k/5778)**4
    return data

def add_insolation(data, log=False):
    data['insolation'] = data.luminosity*(1/data.obj_orb_a_au)**2
    if log==True:
        data['insolation']=np.log10(data.insolation)
    return data

def sil(data):
    K = range(2,10)
    s = []
    for k in K:
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(data)
        labels, centroids = kmeans.labels_,kmeans.cluster_centers_
        s.append(silhouette_score(data, labels, metric='euclidean'))
    plt.plot(K,s,zorder=0)
    plt.ylabel("Silhouette coefficient")
    plt.xlabel("k")
    plt.savefig("Sil.pdf",dpi=1000,transparent=True)
    plt.show()
   
def elbow(data):
    K = range(1,10)
    inertia = np.zeros(len(K))
    for i,k in enumerate(K):
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(data)
        inertia[i] = kmeanModel.inertia_
    plt.plot(K, inertia, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Inertia')
    plt.show()
    plt.scatter(data[:,0],data[:,1])
    plt.title("Distribution")
    plt.show()

def earthUnits(data):
    data['obj_phys_mass_mearth'] = data.obj_phys_mass_mjup*jupiterMass.to(earthMass)
    data['obj_phys_radius_rearth'] = data.obj_phys_radius_rjup*jupiterRad.to(earthRad)
    data=data.drop(columns=['obj_phys_mass_mjup','obj_phys_radius_rjup'])
    return data

file_US="/home/bolacha/University/Project/code/data-example/all_data_US.rdb"
file_EU="/home/bolacha/University/Project/code/data-example/all_data_EU.rdb"
cat_solar="/home/bolacha/University/Project/code/data-example/solar_data.csv"
