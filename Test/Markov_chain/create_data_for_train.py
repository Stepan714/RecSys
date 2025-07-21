from get_sequences import Train
import pandas as pd

def create_dataset_for_train(df, mh, DEPTH=4, last='like', pre_last='listen'):
    def get_probability(from_event, from_item, to_event, to_item):
        key = (f"{from_event}_{from_item}", f"{to_event}_{to_item}")
        return prob_dict.get(key, 0.0)  # Возвращаем 0.0 вместо "0%"
    
    trainer = Train(
        data=df,
        main_column_name='event_type',
        main_column_value=last,
        time_column='timestamp',
        users_id_column='uid'
    )
    
    trainer.find_main_state()
    trainer.processing_depth(depth=DEPTH)
    sequences_df = trainer.build_sequences_dataframe(depth=DEPTH)

    # Создаем явную копию, чтобы избежать SettingWithCopyWarning
    filtered_df = sequences_df[sequences_df[f'event_type_{DEPTH}']==pre_last].copy()

    # Подготовка данных вероятностей
    Map = pd.DataFrame.from_dict(mh.map_name_nodes, orient='index', columns=['node_name'])
    Map.index.name = 'node_id'
    Map = Map.reset_index()
    Map["node_id"] = Map["node_id"].astype("string")
    Map["node_id"] = "id_node_" + Map["node_id"]

    mh.P = mh.P.rename(columns={'Из': 'from_node_id', 'В': 'to_node_id'})
    
    result = (mh.P
              .merge(Map, left_on='from_node_id', right_on='node_id', how='left')
              .rename(columns={'node_name': 'from_node_name'})
              .merge(Map, left_on='to_node_id', right_on='node_id', how='left')
              .rename(columns={'node_name': 'to_node_name'}))
    
    result = result[['from_node_name', 'to_node_name', 'Переходы', 'Вероятность']]
    
    result['Из'] = result['from_node_name'].astype('string')
    result['В'] = result['to_node_name'].astype('string')
    result['Переходы'] = result['Переходы'].astype('Int64')
    result['Вероятность'] = result['Вероятность'].astype('string')

    # Разделение на компоненты
    result[['from_event_type', 'from_item_id']] = result['Из'].str.split('_', n=1, expand=True)
    result[['to_event_type', 'to_item_id']] = result['В'].str.split('_', n=1, expand=True)
    
    # Обработка START
    result.loc[result['from_event_type'] == 'START', 'from_item_id'] = ''

    # Создание словаря вероятностей (более эффективный способ)
    prob_dict = {
        (f"{row['from_event_type']}_{row['from_item_id']}", 
         f"{row['to_event_type']}_{row['to_item_id']}"): float(row['Вероятность'].rstrip('%'))/100
        for _, row in result.iterrows()
    }

    # Добавление вероятностей с использованием .loc
    for d in range(DEPTH-1):
        filtered_df.loc[:, f'P{d+1}{d+2}'] = filtered_df.apply(lambda x: get_probability(
            x[f'event_type_{d+1}'], str(x[f'item_id_{d+1}']),
            x[f'event_type_{d+2}'], str(x[f'item_id_{d+2}'])
        ), axis=1)

    filtered_df.loc[:, f'P{DEPTH}T'] = filtered_df.apply(
        lambda x: get_probability(
            x[f'event_type_{DEPTH}'], 
            str(x[f'item_id_{DEPTH}']),
            x['target_event_type'], 
            str(x['target_item_id'])
        ), 
        axis=1
    )

    return filtered_df