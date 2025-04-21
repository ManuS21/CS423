from __future__ import annotations  #must be first line in your library!
import pandas as pd
import numpy as np
import types
from typing import Dict, Any, Optional, Union, List, Set, Hashable, Literal, Tuple, Self, Iterable
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import sklearn
import warnings
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices


class CustomMappingTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that maps values in a specified column according to a provided dictionary.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies value substitution to a specified column using
    a mapping dictionary, which can be useful for encoding categorical variables or
    transforming numeric values.

    Parameters
    ----------
    mapping_column : str or int
        The name (str) or position (int) of the column to which the mapping will be applied.
    mapping_dict : dict
        A dictionary defining the mapping from existing values to new values.
        Keys should be values present in the mapping_column, and values should
        be their desired replacements.

    Attributes
    ----------
    mapping_dict : dict
        The dictionary used for mapping values.
    mapping_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> mapper = CustomMappingTransformer('category', {'A': 1, 'B': 2, 'C': 3})
    >>> transformed_df = mapper.fit_transform(df)
    >>> transformed_df
       category
    0        1
    1        2
    2        3
    3        1
    """

    def __init__(self, mapping_column: Union[str, int], mapping_dict: Dict[Hashable, Any]) -> None:
        """
        Initialize the CustomMappingTransformer.

        Parameters
        ----------
        mapping_column : str or int
            The name (str) or position (int) of the column to apply the mapping to.
        mapping_dict : Dict[Hashable, Any]
            A dictionary defining the mapping from existing values to new values.

        Raises
        ------
        AssertionError
            If mapping_dict is not a dictionary.
        """
        assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
        self.mapping_dict: Dict[Hashable, Any] = mapping_dict
        self.mapping_column: Union[str, int] = mapping_column  #column to focus on

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomMappingTransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  #always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the mapping to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if mapping_column is not in X.

        Notes
        -----
        This method provides warnings if:
        1. Keys in mapping_dict are not found in the column values
        2. Values in the column don't have corresponding keys in mapping_dict
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?
        warnings.filterwarnings('ignore', message='.*downcasting.*')  #squash warning in replace method below

        #now check to see if all keys are contained in column
        column_set: Set[Any] = set(X[self.mapping_column].unique())
        keys_not_found: Set[Any] = set(self.mapping_dict.keys()) - column_set
        if keys_not_found:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain these keys as values {keys_not_found}\n")

        #now check to see if some keys are absent
        keys_absent: Set[Any] = column_set - set(self.mapping_dict.keys())
        if keys_absent:
            print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] does not contain keys for these values {keys_absent}\n")

        X_: pd.DataFrame = X.copy()
        X_[self.mapping_column] = X_[self.mapping_column].replace(self.mapping_dict)
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with mapping applied to the specified column.
        """
        #self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result

class CustomOHETransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that performs one-hot encoding on specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It applies one-hot encoding to specified categorical
    columns, creating new binary columns for each unique category.

    Parameters
    ----------
    target_column : str or int
        The name (str) or position (int) of the column to which one-hot encoding will be applied.
    dummy_na : bool, default False
        Whether to create a dummy column for NaN values.
    drop_first : bool, default False
        Whether to drop the first category in each encoded column.

    Attributes
    ----------
    target_column : str or int
        The column (by name or position) that will be transformed.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'category': ['A', 'B', 'C', 'A']})
    >>> ohe = CustomOHETransformer('category')
    >>> transformed_df = ohe.fit_transform(df)
    >>> transformed_df
       category_A  category_B  category_C
    0           1           0           0
    1           0           1           0
    2           0           0           1
    3           1           0           0
    """

    def __init__(self, target_column: Union[str, int], dummy_na: bool = False, drop_first: bool = False) -> None:
        """
        Initialize the CustomOHETransformer.

        Parameters
        ----------
        target_column : str or int
            The name (str) or position (int) of the column to apply one-hot encoding to.
        dummy_na : bool, default False
            Whether to create a dummy column for NaN values.
        drop_first : bool, default False
            Whether to drop the first category in each encoded column.

        Raises
        ------
        AssertionError
            If target_column is not a string or integer.
        """
        assert isinstance(target_column, (str, int)), f'{self.__class__.__name__} constructor expected string or int but got {type(target_column)} instead.'
        self.target_column: Union[str, int] = target_column
        self.dummy_na: bool = dummy_na
        self.drop_first: bool = drop_first

    def fit(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> Self:
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        self : instance of CustomOHETransformer
            Returns self to allow method chaining.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self  # always the return value of fit

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply one-hot encoding to the specified column in the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with one-hot encoding applied to the specified column.

        Raises
        ------
        AssertionError
            If X is not a pandas DataFrame or if target_column is not in X.
        """
        assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
        assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  # column legit?

        X_ = pd.get_dummies(X,
                            prefix=self.target_column,  # your choice
                            prefix_sep='_',  # your choice
                            columns=[self.target_column],
                            dummy_na=self.dummy_na,  # will try to impute later so leave NaNs in place
                            drop_first=self.drop_first,
                            dtype=int
                            )
        return X_

    def fit_transform(self, X: pd.DataFrame, y: Optional[Iterable] = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Combines fit() and transform() methods for convenience.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame containing the column to transform.
        y : array-like, default=None
            Ignored. Present for compatibility with scikit-learn interface.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with one-hot encoding applied to the specified column.
        """
        # self.fit(X,y)  #commented out to avoid warning message in fit
        result: pd.DataFrame = self.transform(X)
        return result
    
class CustomDropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    A transformer that either drops or keeps specified columns in a DataFrame.

    This transformer follows the scikit-learn transformer interface and can be used in
    a scikit-learn pipeline. It allows for selectively keeping or dropping columns
    from a DataFrame based on a provided list.

    Parameters
    ----------
    column_list : List[str]
        List of column names to either drop or keep, depending on the action parameter.
    action : str, default='drop'
        The action to perform on the specified columns. Must be one of:
        - 'drop': Remove the specified columns from the DataFrame
        - 'keep': Keep only the specified columns in the DataFrame

    Attributes
    ----------
    column_list : List[str]
        The list of column names to operate on.
    action : str
        The action to perform ('drop' or 'keep').

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
    >>>
    >>> # Drop columns example
    >>> dropper = CustomDropColumnsTransformer(column_list=['A', 'B'], action='drop')
    >>> dropped_df = dropper.fit_transform(df)
    >>> dropped_df.columns.tolist()
    ['C']
    >>>
    >>> # Keep columns example
    >>> keeper = CustomDropColumnsTransformer(column_list=['A', 'C'], action='keep')
    >>> kept_df = keeper.fit_transform(df)
    >>> kept_df.columns.tolist()
    ['A', 'C']
    """

    def __init__(self, column_list: List[str], action: Literal['drop', 'keep'] = 'drop') -> None:
        """
        Initialize the CustomDropColumnsTransformer.

        Parameters
        ----------
        column_list : List[str]
            List of column names to either drop or keep.
        action : str, default='drop'
            The action to perform on the specified columns.
            Must be either 'drop' or 'keep'.

        Raises
        ------
        AssertionError
            If action is not 'drop' or 'keep', or if column_list is not a list.
        """
        assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
        assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
        self.column_list: List[str] = column_list
        self.action: Literal['drop', 'keep'] = action

        self.column_list = column_list
        self.action = action

    def fit(self, X: pd.DataFrame, y: None = None) -> 'DropColumnsTransformer':
        """
        Fit method - performs no actual fitting operation.

        This method is implemented to adhere to the scikit-learn transformer interface
        but doesn't perform any computation.

        Parameters
        ----------
        X : pandas.DataFrame
            The input data to fit.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns self.
        """
        print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the dropping or keeping of columns to the input DataFrame.

        Parameters
        ----------
        X : pandas.DataFrame
            The DataFrame to transform.

        Returns
        -------
        pandas.DataFrame
            A copy of the input DataFrame with columns dropped or kept.
        """
        assert isinstance(X, pd.DataFrame), f"{self.__class__.__name__}.transform expected DataFrame but got {type(X)} instead."

        if self.action == 'keep':
            # Error if column_list contains columns not in the dataframe
            missing_cols = set(self.column_list) - set(X.columns)
            if missing_cols:
                raise AssertionError(f"{self.__class__.__name__}.transform columns to keep not found: {missing_cols}")
            X_ = X[self.column_list].copy()

        elif self.action == 'drop':
            # Warning if column_list contains columns not in the dataframe
            missing_cols = set(self.column_list) - set(X.columns)
            if missing_cols:
                print(f"\nWarning: {self.__class__.__name__}.transform columns to drop not found: {missing_cols}\n")
            # Drop columns with errors='ignore' to avoid errors for missing columns
            X_ = X.drop(columns=self.column_list, errors='ignore').copy()

        else:
            raise ValueError(f"{self.__class__.__name__}.transform unknown action: {self.action}")

        return X_

    def fit_transform(self, X: pd.DataFrame, y: None = None) -> pd.DataFrame:
        """
        Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : pandas.DataFrame
            Input samples.
        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : pandas.DataFrame
            Transformed array.
        """
        # self.fit(X, y)  # uncomment if you want
        result = self.transform(X)
        return result


titanic_transformer = Pipeline(steps=[
    ('gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    #add your new ohe step below
    ('joined', CustomOHETransformer(target_column='Joined'))

    ('fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),

    ], verbose=True)



customer_transformer = Pipeline(steps=[
    # Step 1: Drop ID and Rating columns
    ('drop_columns', CustomDropColumnsTransformer(column_list=['ID', 'Rating'], action='drop')),

    # Step 2: Map Gender values (Female → 1, Male → 0)
    ('map_gender', CustomMappingTransformer(
        mapping_column='Gender',
        mapping_dict={'Female': 1, 'Male': 0}
    )),

    # Step 3: Map Experience Level values (low → 0, medium → 1, high → 2)
    ('map_experience', CustomMappingTransformer(
        mapping_column='Experience Level',
        mapping_dict={'low': 0, 'medium': 1, 'high': 2}
    )),

    # Step 4: One-hot encode OS
    ('encode_os', CustomOHETransformer(target_column='OS')),

    # Step 5: One-hot encode ISP
    ('encode_isp', CustomOHETransformer(target_column='ISP')),

    ('time spent', CustomTukeyTransformer('Time Spent', 'inner')),
], verbose=True)
