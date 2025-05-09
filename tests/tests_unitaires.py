import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from category_encoders import CountEncoder
import old.factorized_functions as ff

class TestFactorizedFunctions:

    def __init__(self):
        self.success_count = 0
        self.total_tests = 4

    def test_identify_column_types(self):
        df = pd.DataFrame({
            'num1': [1.0, 2.0],
            'cat1': ['a', 'b']
        })
        num_cols, cat_cols = ff.identify_column_types(df)
        assert num_cols == ['num1']
        assert cat_cols == ['cat1']
        self.success_count += 1
        print("✅ test_identify_column_types passed")

    def test_fill_missing_values_with_invalid_data(self):
        df1 = pd.DataFrame({'num': [1, 'a'], 'cat': ['a', None]})
        df2 = pd.DataFrame({'num': [np.nan, 2], 'cat': [None, 'b']})
        num_cols, cat_cols = ['num'], ['cat']
        filled_df1, filled_df2 = ff.fill_missing_values(df1.copy(), df2.copy(), num_cols, cat_cols)
        assert filled_df1['num'].iloc[1] == 0  # 'a' should be coerced to NaN and then replaced by 0
        assert filled_df2['cat'].iloc[0] == "Inconnu"
        print("✅ test_fill_missing_values_with_invalid_data passed")

    def test_encode_categorical_features(self):
        df1 = pd.DataFrame({'A': ['x', 'y', 'x']})
        df2 = pd.DataFrame({'A': ['y', 'x']})
        df1_encoded, df2_encoded = ff.encode_categorical_features(df1.copy(), df2.copy(), ['A'])
        assert df1_encoded.shape == df1.shape
        assert df2_encoded.shape == df2.shape
        self.success_count += 1
        print("✅ test_encode_categorical_features passed")

    def test_drop_columns_with_high_missing_or_low_variance(self):
        df = pd.DataFrame({
            'num1': [0, 0, 0, 0, 1],
            'num2': [1, 2, 3, 4, 5],
            'cat1': ['a', 'a', 'a', 'a', 'a']
        })
        df['cat1'] = df['cat1'].astype('object')
        train_df, test_df = df.copy(), df.copy()
        num_cols, cat_cols = ff.identify_column_types(df)
        cleaned_train_df, cleaned_test_df, dropped = ff.drop_columns_with_high_missing_or_low_variance(
            train_df, test_df, num_cols, threshold=0.6, var_threshold=0.01
        )
        assert 'num1' in dropped or 'cat1' in dropped
        self.success_count += 1
        print("✅ test_drop_columns_with_high_missing_or_low_variance passed")

    def run_all_tests(self):
        self.test_identify_column_types()
        self.test_fill_missing_values_with_invalid_data()
        self.test_encode_categorical_features()
        self.test_drop_columns_with_high_missing_or_low_variance()

        print(f"\n{self.success_count}/{self.total_tests} tests unitaires passent ✅")

if __name__ == "__main__":
    tester = TestFactorizedFunctions()
    tester.run_all_tests()
