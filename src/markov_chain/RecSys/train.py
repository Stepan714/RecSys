import numpy as np
import pandas as pd

class Train:
    def __init__(self, data=None, struct=None, main_column_name=None, main_column_value=None, time_column=None, users_id_column=None):
        self.data = data  # pandas DataFrame
        self.struct = struct  # tuple
        self.main_column_name = main_column_name # string
        self.main_column_value = main_column_value  # string
        self.time_column = time_column  # string
        self.users_id_column = users_id_column  # string
        self.count_nodes = 0

    def find_main_state(self):
        self.data = self.data.sort_values(by=[self.users_id_column, self.time_column], ascending=True)
        self.data['row_num'] = self.data.groupby(self.users_id_column).cumcount() # нумерация событий для каждого пользователя в порядке возрастания времени

        def main_state(row):
            if row[self.main_column_name] == self.main_column_value:
                return 1
            return 0

        self.data['flg_main_state'] = self.data.apply(main_state, axis=1)

    def processing_depth(self, depth=3):
        self.main_state: dict[str, set()] = {}
        grouped = self.data.groupby(self.users_id_column)

        for user_id, user_data in grouped:
            user_data = user_data.sort_values(self.time_column)
            target_indices = user_data[user_data['flg_main_state'] == 1].index

            for target_idx in target_indices:
                target_pos = user_data.loc[target_idx, 'row_num'] # Находим позицию целевого события в последовательности пользователя

                target_start = target_pos - depth
                if target_start < 1: # Проверяем, есть ли depth предыдущих событий
                    target_start = 1
                if user_id in self.main_state.keys():
                    self.main_state[user_id].add((target_start, target_pos))
                else:
                    self.main_state[user_id] = {(target_start, target_pos)}

        return self.main_state

