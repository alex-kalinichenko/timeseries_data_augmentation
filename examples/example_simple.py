import sys
import warnings

import numpy as np


sys.path.append("src")
from sklearn.ensemble import RandomForestRegressor


warnings.filterwarnings("ignore")

from data_generators import generate_test_data_3
from data_generators import get_data_from_file
from tools import run_model_for_raw_and_augmented_data
from tools import smape


if __name__ == "__main__":
    """
    run only one experiment using generate_test_data_1 dataset, collect result into res dataframe and print it
    method generate_test_data_1 could be replaced (generate_test_data_2, generate_test_data_3) or you own data set
    """

    df, train_test_split = generate_test_data_3()
    # df, train_test_split = get_data_from_file("data/df_710.csv")
    res = run_model_for_raw_and_augmented_data(
        model=RandomForestRegressor(n_estimators=100, random_state=42),
        df=df,
        train_test_split=train_test_split,
        tabgan=True,
    )

    # visualize result in console
    r = res[~np.isnan(res.y)]

    print("smape for raw data:", smape(r.y, r.pred_raw))
    print("smape for augmented data:", smape(r.y, r.pred_augm))
    print("smape for tabgan data:", smape(r.y, r.pred_gan))
