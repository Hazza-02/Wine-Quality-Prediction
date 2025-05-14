from dataloader import WineDataHandler
from preprocess import preprocess
import pandas as pd
from IPython.display import display

# Define the URLs
red_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
white_wine_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"

# Instantiate the data loader
wine_loader = WineDataHandler(red_wine_url, white_wine_url)

# Get the combined DataFrame
combined_wine_df = wine_loader.combined_df

#wine_loader.plot_correlation_matrix()
#wine_loader.plot_scatter_quality()
# Encode wine type and get the encoded DataFrame

combined_wine_df = preprocess.encode_wine_type(combined_wine_df)

display(combined_wine_df)