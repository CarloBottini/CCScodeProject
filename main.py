import pandas as pd
import os

print("LOADING REAL DATA")

csv_path = "data/METLIN_IMS_dimers_rmTM.csv"

if os.path.exists(csv_path):
    print(f" File '{csv_path}' located successfully")
    
    #reading the first 500 
    df = pd.read_csv(csv_path, nrows=500)
    
    print(f"\n DATASET SHAPE:")
    print(f"Total molecules loaded (rows): {df.shape[0]}")
    print(f"Total features (columns): {df.shape[1]}")
    
    print(" COLUMN INSPECTION ")
    #printing columns to see the data
    columns = df.columns.tolist()
    for col in columns:
        print(f" - {col}")
    
    print("\n FIRST 3 ROWS")
    #displaying the first 3 rows to understand the format of the data.
    pd.set_option('display.max_columns', None) #ensures Pandas prints all columns in the terminal
    print(df.head(3).to_string())
    
else:
    print(f" ERROR: Cannot find '{csv_path}'. Check if the file name is exactly correct.")



'''
It printed this structure:

LOADING REAL DATA
 File 'data/METLIN_IMS_dimers_rmTM.csv' located successfully

DATASET SHAPE:
Total molecules loaded (rows): 500  
Total features (columns): 21
COLUMN INSPECTION:
 - Molecule Name
 - Molecular Formula
 - METLIN ID
 - Precursor Adduct
 - CCS1
 - CCS2
 - CCS3
 - CCS_AVG
 - % CV
 - m/z
 - Adduct
 - m/z.1
 - Dimer
 - Dimer.1
 - dimer line
 - CCS
 - m/z.2
 - pubChem
 - inchi
 - smiles
 - InChIKEY

 FIRST 3 ROWS
Molecule Name                                                                                                           Molecular Formula   METLIN ID   Precursor Adduct    CCS1    CCS2    CCS3        CCS_AVG      % CV       m/z         Adduct       m/z.1            Dimer         Dimer.1         dimer line            CCS           m/z.2       pubChem         inchi                                                                                                                                                                                         smiles                                                    InChIKEY
0  ({[(2,4,6-trimethylphenyl)carbamoyl]methyl}carbamoyl)methyl 1-[(tert-butylcarbamoyl)methyl]cyclopentane-1-carboxylate        C25H37N3O5    1133361    460.2806[M+H]      214.67  214.67  214.29       214.54     0.102262    460.2806     [M+H]       460.2806        245.447538       Monomer         NaN               135.0           50.0        16384698        InChI=1S/C25H37N3O5/c1-16-11-17(2)22(18(3)12-16)27-20(30)14-26-21(31)15-33-23(32)25(9-7-8-10-25)13-19(29)28-24(4,5)6/h11-12H,7-10,13-15H2,1-6H3,(H,26,31)(H,27,30)(H,28,29)      O=C(COC(=O)C1(CCCC1)CC(=O)NC(C)(C)C)NCC(=O)Nc1c(C)cc(cc1C)C         FEGOCYXICMRAQQ-UHFFFAOYSA-N


There are some NaN values in some rows
'''









