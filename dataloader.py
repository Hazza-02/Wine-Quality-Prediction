
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class WineDataHandler:

    """
    Initializes the WineDataHandler with URLs for red and white wine datasets.
    Loads and combines the datasets into a pandas DataFrame.
    """
    def __init__(self, red_wine_url, white_wine_url):

        self.red_wine_url = red_wine_url
        self.white_wine_url = white_wine_url
        self.red_df = self._load_data(self.red_wine_url, "red")
        self.white_df = self._load_data(self.white_wine_url, "white")
        self.combined_df = self._combine_datasets()

    """
    Loads data from a given URL into a pandas DataFrame.
    """
    def _load_data(self, url, wine_type):

        try:
            df = pd.read_csv(url, sep=';')
            print(f"{wine_type.capitalize()} wine data loaded successfully!")
            return df
        except Exception as e:
            print(f"Error loading {wine_type} wine data: {e}")
            return None

    """
    Combines the red and white wine DataFrames into a single DataFrame
    and adds a 'wine_type' column.
    """
    def _combine_datasets(self):
        
        if self.red_df is not None and self.white_df is not None:
            self.red_df['wine_type'] = 'red'
            self.white_df['wine_type'] = 'white'
            combined_df = pd.concat([self.red_df, self.white_df], ignore_index=True)
            print("\nRed and white wine datasets combined successfully!")
            return combined_df
        else:
            print("\nError: One or both base datasets were not loaded. Cannot combine.")
            return None

    """
    Inspects the first few rows and structure of the red and white wine DataFrames.
    """
    def inspect_base_datasets(self):

        print("--- Red Wine Data ---")
        print("First 5 rows:")
        print(self.red_df.head())
        print("\nStructure of the DataFrame:")
        self.red_df.info()

        print("\n--- White Wine Data ---")
        print("First 5 rows:")
        print(self.white_df.head())
        print("\nStructure of the DataFrame:")
        self.white_df.info()

    """
    Inspects the first few rows and structure of the combined DataFrame.
    """
    def inspect_combined_dataset(self):

        if self.combined_df is not None:
            print("\n--- Combined Wine Data ---")
            print("First 5 rows:")
            print(self.combined_df.head())
            print("\nInformation about the combined DataFrame:")
            self.combined_df.info()

    """
    Plots the correlation matrix of the numerical features in the given DataFrame.
    Uses the internal combined dataframe if none is provided.
    """
    def plot_correlation_matrix(self, dataframe=None, figsize=(12, 10)):
        
        if dataframe is None:
            if self.combined_df is not None:
                correlation_matrix = self.combined_df.corr(numeric_only=True)
                plt.figure(figsize=figsize)
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                plt.title('Correlation Matrix of Combined Wine Features')
                plt.show()
            else:
                print("Error: Combined dataset is not available in the WineDataHandler.")
        else:
            correlation_matrix = dataframe.corr(numeric_only=True)
            plt.figure(figsize=figsize)
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix of Wine Features')
            plt.show()
            
    """
    Plots scatter plots of each numerical feature against the 'quality' score
    in the given DataFrame (or the combined one if None).
    """        
    def plot_scatter_quality(self, dataframe=None, figsize=(8, 6)):

        if dataframe is None:
            if self.combined_df is not None:
                for column in self.combined_df.select_dtypes(include=['number']).columns:
                    if column != 'quality':
                        plt.figure(figsize=figsize)
                        sns.scatterplot(x=column, y='quality', data=self.combined_df, hue='wine_type')
                        plt.title(f'{column} vs Quality')
                        plt.xlabel(column)
                        plt.ylabel('Quality')
                        plt.show()
            else:
                print("\nError: Combined dataset is not available.")
        else:
            for column in dataframe.select_dtypes(include=['number']).columns:
                if column != 'quality':
                    plt.figure(figsize=figsize)
                    sns.scatterplot(x=column, y='quality', data=dataframe)
                    plt.title(f'{column} vs Quality')
                    plt.xlabel(column)
                    plt.ylabel('Quality')
                    plt.show()
                    
    def one_hot_encode(df, column, drop_first=True):

        df_encoded = pd.get_dummies(df, columns=[column], drop_first=drop_first)
        return df_encoded
