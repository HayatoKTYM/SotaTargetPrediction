import numpy as np
import pandas as pd
import glob
import os

def np_to_dataframe(np_list) -> pd.DataFrame:
    """
    np -> pd.DataFrame
    """
    if type(np_list) == type('hello'): # str 
        np_list = np.load(np_list)
        np_list = np_list[:len(np_list)//2*2] #奇数個なら偶数個に
        np_list = np_list.reshape(-1,256) #20fps > 10fps
        return pd.DataFrame(np_list)
    else: #np.load 済みなら
        return pd.DataFrame(np_list)
        
def setup(PATH='/mnt/aoni04/katayama/DATA2020/',dense_flag=False):
    lld_files = sorted(glob.glob(os.path.join(PATH,'lld_all/*csv')))
    feature_files = sorted(glob.glob(os.path.join(PATH,'feature/*csv')))
    gaze_files = sorted(glob.glob(os.path.join(PATH, 'img_middle64/*npy')))

    print(f'file length is {len(lld_files)} and {len(feature_files)}')
    df_list = []
    lld_list = []
    for i in range(len(feature_files)):
        df = pd.read_csv(feature_files[i])
        lld = pd.read_csv(lld_files[i])
        gaze = pd.DataFrame(np.load(gaze_files[i]))

        length = min([len(gaze), len(df), len(lld)//10])
        gaze = gaze[:length]
        df = df[:length]
        df = pd.concat([df, gaze], axis=1)
        df = df.fillna(0)
        df_list.append(df)

        lld = lld[:length*10]
        lld_list.append(lld)
    return df_list, lld_list
