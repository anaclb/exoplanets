import pandas as pd
import os
from astropy.io import ascii

file_US="/home/bolacha/University/Project/code/database/selection_US.rdb"
file_EU="/home/bolacha/University/Project/code/database/selection_EU.rdb"

file1=pd.read_csv(file_US, sep = '\t', header=[1,2])
file2=pd.read_csv(file_EU, sep = '\t', header=[1,2])
df1, df2 = pd.DataFrame(file1), pd.DataFrame(file2)

print(df1)
