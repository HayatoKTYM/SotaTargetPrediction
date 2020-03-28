import numpy as np
import pandas as pd
import glob

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
        
def setup(dense_flag=False):
    img_middle_feature_files = sorted(glob.glob('/mnt/aoni04/katayama/DATA2020/lld_all/*csv'))
    feature_files = sorted(glob.glob('/mnt/aoni04/katayama/DATA2020/feature/*csv'))
    print(f'file length is {len(img_middle_feature_files)} and {len(feature_files)}')
    df_list = []
    lld_list = []
    for i in range(len(feature_files)-94):
        df = pd.read_csv(feature_files[i])
        try:
            lld = pd.read_csv(img_middle_feature_files[i])
        except:
            print(img_middle_feature_files[i])
            continue
        #df = pd.concat([df,img,imgB],axis=1)
        #if not dense_flag:
        #    reset_array = [-1] * len(df.columns)
        #    df.loc['reset'] = reset_array 
        #    df_list.append(df)
        #else:
        df_list.append(df)
        lld_list.append(lld)
    return df_list, lld_list
