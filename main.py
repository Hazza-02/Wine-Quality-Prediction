from dataloader import WineDataHandler
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Define the URLs
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Instantiate the data loader
wine_loader = WineDataHandler(red_wine_url, white_wine_url)

# Get the combined DataFrame
combined_wine_df = wine_loader.combined_df

"""
Graph plots
"""
#wine_loader.plot_correlation_matrix()
#wine_loader.plot_scatter_quality()

# Encode wine type and get the encoded DataFrame
def encode_wine_type(df, drop_first=True, inplace=True):
        
        df_encoded = pd.get_dummies(df, columns=['wine_type'], drop_first=drop_first)
        df_encoded = df_encoded.astype(int)
            
        if inplace:
            df = df_encoded
            return df
        else:
            return df_encoded
    
combined_wine_df = encode_wine_type(combined_wine_df)       
#display(combined_wine_df)

X = combined_wine_df.drop("quality", axis=1) # All columns except 'quality'
Y = combined_wine_df["quality"] # The target variable
X_train, X_test, y_train, y_test = train_test_split(X, Y ,test_size=0.2)

norm = MinMaxScaler().fit(X_train) # Fit min-max scaler on training data
X_train_norm = norm.transform(X_train) # Transform the training data

display(X_train_norm)
 

"""
Check for missing values
"""
#print("\nMissing values per column:")
#print(combined_wine_df.isnull().sum())