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

    def filter_by_time_range(self, start_time: int, end_time: int):
        """
        Удаляет строки из self.data, где время не входит в указанный интервал.
        Время указывается в секундах (целые числа).
        
        :param start_time: нижняя граница времени (включительно)
        :param end_time: верхняя граница времени (включительно)
        """
        if self.data is None or self.time_column not in self.data.columns:
            raise ValueError("Нет данных или отсутствует колонка времени.")
    
        before_count = len(self.data)
        self.data = self.data[
            (self.data[self.time_column] >= start_time) &
            (self.data[self.time_column] <= end_time)
        ]
        after_count = len(self.data)
    
        print(f"Отфильтровано по времени: оставлено {after_count} из {before_count} записей "
              f"(границы: {start_time} - {end_time})")

        

    def build_sequences_dataframe(self, depth=3):
        rows = []
    
        for user_id, segments in self.main_state.items():
            user_data = self.data[self.data[self.users_id_column] == user_id]
    
            for start, end in segments:
                if end - start != depth:
                    continue  # Пропускаем, если длина меньше depth
    
                # Получаем события из start до end - 1
                seq_rows = user_data[user_data['row_num'].between(start, end - 1)]
    
                # Получаем target-событие (строка с row_num == end)
                target_row = user_data[user_data['row_num'] == end]
                if target_row.empty:
                    continue  # на случай, если target_pos отсутствует (редко, но может быть)

                target_row = target_row.iloc[0]    
    
                # Сборка строки
                row = [str(user_id)]
                for _, r in seq_rows.iterrows():
                    row.append(r['item_id'])
                    row.append(r[self.main_column_name])
    
                # Добавляем информацию о target
                row.append(target_row['item_id'])
                row.append(target_row[self.main_column_name])
    
                rows.append(row)

        # Формирование заголовков
        columns = ['uid']
        for i in range(1, depth + 1):
            columns.extend([f'item_id_{i}', f'{self.main_column_name}_{i}'])
    
        columns.extend(['target_item_id', f'target_{self.main_column_name}'])
    
        return pd.DataFrame(rows, columns=columns)

        

    def processing_depth(self, depth=3):
        self.main_state: dict[str, set()] = {}
        grouped = self.data.groupby(self.users_id_column)

        for user_id, user_data in grouped:
            user_data = user_data.sort_values(self.time_column)
            target_indices = user_data[user_data['flg_main_state'] == 1].index

            for target_idx in target_indices:
                target_pos = user_data.loc[target_idx, 'row_num'] # Находим позицию целевого события в последовательности пользователя

                target_start = max(0, target_pos - depth)
                if user_id in self.main_state.keys():
                    self.main_state[user_id].add((int(target_start), int(target_pos)))
                else:
                    self.main_state[user_id] = {(int(target_start), int(target_pos))}

        return self.main_state   
 