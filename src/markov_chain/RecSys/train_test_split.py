import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def split_by_field(df, field, test_size=0.2, random_state=42):
    """
    Разбивает DataFrame на train и test по уникальным значениям указанного поля.

    Параметры:
    ----------
    df : pandas.DataFrame
        Исходный DataFrame
    field : str
        Название поля, по которому происходит разделение
    test_size : float, optional (default=0.2)
        Доля данных для тестовой выборки (0.0-1.0)
    random_state : int, optional (default=42)
        Seed для воспроизводимости результатов

    Возвращает:
    -----------
    train_df, test_df : tuple of pandas.DataFrame
        Разделенные DataFrame
    """
    unique_values = df[field].unique()

    train_values, test_values = train_test_split(
        unique_values,
        test_size=test_size,
        random_state=random_state
    )

    train_mask = df[field].isin(train_values)
    test_mask = df[field].isin(test_values)

    return df[train_mask].copy(), df[test_mask].copy()
