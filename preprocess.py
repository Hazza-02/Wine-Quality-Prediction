import pandas as pd

"""
Add one hot encoding here instead of in the data loader
Check for missing values
Nomarlise values so all values are on the same scale stopping certain values overpowering the training process
"""

class preprocess():
    
        def encode_wine_type(df, drop_first=True, inplace=True):
        
            df_encoded = pd.get_dummies(df, columns=['wine_type'], drop_first=drop_first)
            df_encoded = df_encoded.astype(int)
            
            if inplace:
                df = df_encoded
                return df
            else:
                return df_encoded
            