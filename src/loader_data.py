import pandas as pd
import pathlib as pl

def load_csv ():
    
    path_init = pl.Path(__file__).parent
    path_csv = path_init.parent / 'data' / 'dados_consumo_energia.csv'

    if not path_csv.exists():
        raise FileNotFoundError("Arquivo não encontrado")
    
    df = pd.read_csv(path_csv)

    return df

