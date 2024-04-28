
METHODS = [['q', 'qiskit'], ['p', 'pennylane'], ['c', '']]

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from pandas.api.types import is_numeric_dtype



class Data: 
    def __init__(self, data: pd.DataFrame) -> None:
        """
        Initialize the data from a given DataFrame.

        Parameters
        ----------
        data: pandas.DataFrame
            The data we want to use to pre-process as a pandas DataFrame
            
    
        Saves
        -------
        df : pandas.DataFrame
            The data.
        """
        try:
            if isinstance(data, pd.DataFrame):
                self.df = data
            else:
                raise ValueError("Invalid data type. Please provide a pandas DataFrame")
            self.columns_names = self.df.get_columns()
        except Exception as e:
            print(e)
            
    
    def __str__(self, n=5) -> str:
        """
        Print the first 5 rows of the dataframe by default.

        If the training testing split is done we will print out those dataframes instead of the original data. 

        Parameters
        ----------
        n : int
            The number of rows to print (has to be a number between 0 and the number of rows in the dataframe).
            Default is 5.
            If n is greater than the number of rows in the dataframe, we will print all the rows.
            If n is less than 0, we will print an error message.
        """
        if n > len(self.df):
            n = len(self.df)
        elif n < 0:
            raise ValueError("Invalid number of rows. Please provide a number between 0 and the number of rows in the dataframe.")

        if hasattr(self, 'train_df'):
            all_prints = [
                self.train_df.head(n).to_string(),
                self.test_df.head(n).to_string()
            ]
            return '\n\n'.join(all_prints)
        else:
            return self.df.head(n).to_string()

    def __len__(self) -> int:
        """
        Return the number of rows of the dataframe.

        If the training testing split is done we will return the length of the training dataframe instead of the original data.

        Returns
        -------
        int: 
            The number of rows of the dataframe. 
        tuple: (int, int)
            The number of rows of the training and testing dataframes.
            The first return is the number of rows of the training dataframe.
            The second return is the number of rows of the testing dataframe.
        """
        return len(self.df)
        
    def __getitem__(self, col: str) -> pd.Series:
        """
        Return the column of the dataframe.

        Parameters
        ----------
        col : str
            The column name.

        Returns
        -------
        pandas.Series: 
            The column of the dataframe.
        """
        return self.df[col]
    
    # I do not have a setitem method because I do not want to change the data in the dataframe.
    
    def get_data(self):
        """
        Get the orginal dataframe.

        Returns
        -------
        pandas.DataFrame: 
            The original dataframe.
        """
        return self.df
    
    def get_train_test_data(self):
        """
        Get the training and testing dataframes.

        Returns 
        -------
        A tuple of dataframes.
        tuple: (pandas.DataFrame, pandas.DataFrame)
            The first return is the training dataframe.
            The second return is the testing dataframe.
        """
        if hasattr(self, 'train_df'):
            return (self.train_df, self.test_df)
        else:
            raise ValueError("Train test split has not been done yet.")

    def split_train_test(self, test_size=0.2, y_col=None, method='c'):
        """
        Split the data into training and testing sets.

        Parameters
        ----------
        test_size : float
            The size of the testing set. Default is 0.2.
            This is the proportion/percentage of number of rows in the dataset to include in the test split.
        y_col : str, optional
            The column name of the target variable. Default is None.
            If None, the data is split into training and testing sets.
        method : str, optional
            The method to use for preprocessing.
            Options are ['q', 'qiskit', 'p', 'pennylane'] for quantum pre-processing or ['c', 'classical'] for classical pre-processing.
            Default is 'c' or 'classical'.

            
        Saves
        -------
        If we have a target variable:
            train_X : pandas.DataFrame
                The training data without the target variable.
            test_X : pandas.DataFrame
                The testing data without the target variable.
            train_y : pandas.Series
                The training target variable.
            test_y : pandas.Series
                The testing target variable.
        If we do not have a target variable:
            train_df : pandas.DataFrame
                The training data.
            test_df : pandas.DataFrame
                The testing data.
        """
        self.pre_process(method=method)
        if y_col:
            y = self.df[y_col]
            X = self.df.drop(columns=[y_col])
            train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size)
            self.train_X = train_X
            self.test_X = test_X
            self.train_y = train_y
            self.test_y = test_y
        else:
            print(self.df.head())
            train_df, test_df = train_test_split(self.df, test_size=test_size)
            self.train_df = train_df
            self.test_df = test_df
    
    def pre_process(self, method='c'):
        """
        Preprocess the data in either classical or quantum way.

        Parameters
        ----------
        method : str, optional
            The method to use for preprocessing. 
            Options are ['q', 'qiskit', 'p', 'pennylane'] for quantum pre-processing or ['c', 'classical'] for classical pre-processing.
            Default is 'c' or 'classical'.


        Saves
        -------
        method: str
            The method used for preprocessing the data.
            Helps keep the methods consistent for encoding the data.
        df : pandas.DataFrame
            The preprocessed data.
        """
        self.method = method
        if method in METHODS[0] or method in METHODS[1]:
            self.df = self.q_preprocess_data(self.df)
        elif method in METHODS[2]:
            self.df = self.c_preprocess_data(self.df)
        else:
            raise ValueError("Invalid method. Choose from ['q', 'qiskit', 'p', 'pennylane'] for quantum pre-processing or ['c', 'classical'] for classical pre-processing.")
        
    def q_preprocess_data(self):
        """
        Preprocess the data in a quantum way.

        Returns
        -------
        df : pandas.DataFrame
            The preprocessed data.
        """
        pass
    
    def get_columns(self):
        """
        Get the columns of the dataframe.

        Returns
        -------
        list: 
            The columns of the dataframe.
        """
        self.columns_names = self.df.columns.tolist()
    
    
    def c_preprocess_data(self, method='c'):
        """
        Preprocess the data in a classical way.

        Takes the object columns and converts them to categorical data which is then One-Hot Encoded which is a binary encoding method.
        Takes any numeric columns and scales with min-max scaling method which will have a range of 0 to 1.

        Please do not use any date time columns as it will not be preprocessed.

        Saves
        -------
        df : pandas.DataFrame
            Updates the dataframe with the preprocessed data.
        """
        if not hasattr(self, 'columns_names'):
            self.get_columns()
            
        for col in self.columns_names:
            if pd.api.types.is_object_dtype(self.df[col]):
                self.df[col] = pd.Categorical(self.df[col])
                self.df[col] = self.df[col].cat.codes
                self.df
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.df[col] = StandardScaler().fit_transform(self.df[col].values.reshape(-1, 1))
    