from typing import Literal
import random
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from MarkovChain import MarkovChain


# steps = ["Listen", "Like", "Dislike", "Unlike", "Undislike"]
#
# # Параметры генерации
# num_rows = 100
# max_time = 600
# num_users = 10
# num_items = 20
#
# # Генерация датасета
# data = []
# for _ in range(num_rows):
#     user_id = random.randint(1, num_users)
#     item_id = random.randint(100, 100 + num_items - 1)
#     time = random.randint(0, max_time)
#     step = random.choice(steps)
#     data.append((user_id, item_id, time, step))
#
# # Создание DataFrame
# df = pd.DataFrame(data, columns=["user_id", "item_id", "time", "step"])
# print(df.head(10))
# # name=None, data=None, struct=None, time_column=None, users_id_column=None

df = pd.DataFrame(
    {
        'user_id': [1, 1, 1, 1, 2, 2, 2, 3, 3, 4],
        'item_id': [5, 5, 6, 6, 4, 10, 10, 3, 3, 7],
        'time': [1, 2, 3, 4, 1, 2, 4, 7, 8, 60],
        'step': ['listen', 'like', 'listen', 'dislike', 'listen', 'listen', 'dislike', 'listen', 'like', 'listen']
    }
)

mh = MarkovChain(
    name='Example',
    data=df,
    struct=('step', 'item_id'),
    time_column='time',
    users_id_column='user_id'
)
mh.preprocessing_data()
mh.build_markov_chain()
mh.show_markov_chain(visualize=True, top_n=10)