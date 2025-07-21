import pandas as pd
def flat_split_train_val_test_pd(
    df: pd.DataFrame,
    test_timestamp: int,
    val_size: int = 0,
    gap_size: int = 0,
    drop_non_train_items: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame]:
    """
    Делит pandas DataFrame на train/val/test по временному признаку с фильтрацией по пользователям и товарам.
    """

    # Вычисляем временные границы
    train_end = test_timestamp - gap_size - val_size - (gap_size if val_size > 0 else 0)
    val_start = test_timestamp - val_size - gap_size
    val_end = test_timestamp - gap_size

    # --- ТРЕНИРОВОЧНЫЕ ДАННЫЕ ---
    train = df[df['timestamp'] < train_end]
    train_uids = set(train['uid'].unique())
    train_item_ids = set(train['item_id'].unique())

    # --- ВАЛИДАЦИЯ (если нужно) ---
    validation = None
    if val_size > 0:
        validation = df[
            (df['timestamp'] >= val_start) &
            (df['timestamp'] < val_end) &
            (df['uid'].isin(train_uids))
        ]
        if drop_non_train_items:
            validation = validation[validation['item_id'].isin(train_item_ids)]

    # --- ТЕСТ ---
    test = df[
        (df['timestamp'] >= test_timestamp) &
        (df['uid'].isin(train_uids))
    ]
    if drop_non_train_items:
        test = test[test['item_id'].isin(train_item_ids)]

    return train, validation, test
