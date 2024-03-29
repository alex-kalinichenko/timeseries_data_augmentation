import os
import sys
import warnings

from datetime import datetime

import numpy as np
import pandas as pd

from tabgan.sampler import GANGenerator


sys.path.insert(0, "src")
from augmentation import TransformData


warnings.filterwarnings("ignore")


def smape(a, f):
    return np.sum(np.abs(f - a)) / np.sum(f + a)


def run_model_for_raw_and_augmented_data(
    model,
    df: pd.DataFrame,
    train_test_split: datetime,
    N=15,
    K=10,
    augm=True,
    tabgan=True,
    tabgan_params=None,
) -> pd.DataFrame:
    """
    df dataFrame will be splitted into two dataFrames using train_test_split
    then we try to train and predict y column using two approaches.
    In the first one we use only raw data and in second one we try to augment data
    and predict y column using it.

    result will be returned as pd.DataFrame with 2 result columns like this:

    time                     y       pred_raw      pred_augm
    2021-01-20 00:00:00,    nan,     102.72,       102.78
    2021-01-20 01:00:00,    nan,     102.44,       102.29
    2021-01-20 02:00:00,    nan,     100.52,       100.34
    2021-01-20 03:00:00,    nan,     100.52,       100.34
    2021-01-20 04:00:00,    95.53,   100.23,       100.29
    2021-01-20 05:00:00,    nan,      98.03,        98.14

    where
      y - initial y column,
      pred_raw - prediction based on raw data,
      pred_augm - prediction based on augmented data
    """

    d = df.copy()

    train_df = d[(d.time < train_test_split) & (~np.isnan(d.y))]
    test_df = d[d.time >= train_test_split]
    train_df.fillna(train_df.mean(), inplace=True)
    train_df.fillna(0, inplace=True)
    q_liq_original_col = test_df.y.copy()
    test_df.fillna(train_df.mean(), inplace=True)
    test_df.fillna(0, inplace=True)
    test_df.y = q_liq_original_col

    model.fit(train_df.drop(columns=["y", "time"]), train_df.y)
    res1 = model.predict(test_df.drop(columns=["y", "time"]))

    tr = TransformData(target_column="y", time_column="time", N=N, k=K)

    r = pd.DataFrame({"time": test_df.time, "y": test_df.y, "pred_raw": res1})

    if augm:
        augmented_data = tr.augment_data(train_df)
        model.fit(augmented_data.drop(columns=["time", "y"]), augmented_data.y)
        res2 = model.predict(test_df.drop(columns=["time", "y"]))
        r["pred_augm"] = res2

    if tabgan:
        try:
            if tabgan_params is not None:
                tg = GANGenerator(**tabgan_params)
            else:
                tg = GANGenerator()

            gan_train, gan_y = tg.generate_data_pipe(
                train_df=train_df.drop(columns=["y", "time"]),
                target=train_df[["y"]],
                test_df=test_df.drop(columns=["y", "time"]),
            )
            model.fit(gan_train, gan_y)
            gan_pred = model.predict(test_df.drop(columns=["y", "time"]))
            r["pred_gan"] = gan_pred
        except Exception as ex:
            r["pred_gan"] = -1
            print(f"Tabgan error: {ex}. Tabgun prediction = -1")
            print()

    return r


def experiment(
    model,
    df: pd.DataFrame,
    train_test_split: datetime,
    N=10,
    K=15,
    augm=True,
    tabgan=True,
    tabgan_params=None,
):
    """
    hold one experiment using df dataframe
    dataframe will be stlitted into train and test datatet using train_test_split timestamp
    """

    experiment_result = run_model_for_raw_and_augmented_data(
        model,
        df,
        train_test_split,
        N=N,
        K=K,
        augm=augm,
        tabgan=tabgan,
        tabgan_params=tabgan_params,
    )
    e = experiment_result[~np.isnan(experiment_result.y)]
    return smape(e.y, e.pred_raw), smape(e.y, e.pred_augm)
