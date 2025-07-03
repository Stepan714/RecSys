import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import display
from itertools import product

class MarkovChain:
    def __init__(self, name=None, data=None, struct=None, time_column=None, users_id_column=None):
        # struct, for example (действие, id_song) = (название поля, название поля айдишника)
        self.NAME = name # String
        self.data = data # pandas DataFrame
        self.struct = struct # tuple
        self.time_column = time_column # string
        self.users_id_column = users_id_column # string
        self.count_nodes = 0
        self.map_name_nodes: dict[int, str] = {}

        self.markov_chain: dict[str, dict[str, int]] = {}
        # {
        #     'id_node_1' : {2, 4, 6},
        #     'id_node_2': {1, 5, 70},
        #     ...
        #
        # }

    def preprocessing_data(self):
        if self.data is None:
            raise ValueError("Подан пустой датасет")

        self.data = self.data.sort_values(by=[self.users_id_column, self.time_column], ascending=True)

        if not self.struct:
            self.count_nodes = 0
            self.map_name_nodes = {}
            return

        unique_values = [self.data[col].unique() for col in self.struct]
        unique_counts = [len(values) for values in unique_values]
        self.count_nodes = np.prod(unique_counts)

        self.map_name_nodes = {}
        node_id = 0

        for combination in product(*unique_values):
            key = "_".join(str(val) for val in combination)
            self.map_name_nodes[node_id] = key
            node_id += 1

        return

    def build_markov_chain(self):
        if self.data is None or not self.struct:
            return

        START_NODE = "START"  # Добавляем специальную стартовую ноду
        self.map_name_nodes[-1] = START_NODE  # Используем -1 как ID для стартовой ноды
        self.count_nodes += 1

        self.markov_chain = {  # Инициализация цепи с учетом стартовой ноды
            f'id_node_{i}': {}
            for i in range(-1, self.count_nodes - 1)  # Включаем -1 (старт) и обычные ноды
        }

        self.node_name_to_id = {v: k for k, v in self.map_name_nodes.items()}  # Создаем обратный маппинг для быстрого поиска

        grouped = self.data.groupby(self.users_id_column)

        for user_id, user_data in grouped:
            prev_node = f'id_node_{-1}' # Всегда начинаем со стартовой ноды
            first_transition = True

            for _, row in user_data.iterrows():
                current_values = [str(row[col]) for col in self.struct]
                current_node_key = "_".join(current_values)
                current_node_id = self.node_name_to_id[current_node_key]

                if current_node_id in self.markov_chain[prev_node]:  # Добавляем переход
                    self.markov_chain[prev_node][f'id_node_{current_node_id}'] += 1
                else:
                    self.markov_chain[prev_node][f'id_node_{current_node_id}'] = 1

                prev_node = f'id_node_{current_node_id}'
                first_transition = False

        return

    def show_markov_chain(self, visualize=True, top_n=10):
        # Красивый табличный вывод
        print(f"\nМарковская цепь '{self.NAME}' (первые {top_n} переходов):")
        print("=" * 60)

        # Создаем DataFrame для красивого отображения
        transitions = []
        for source, targets in self.markov_chain.items():
            for target, count in targets.items():
                transitions.append({
                    'Из': source,
                    'В': target,
                    'Переходы': count,
                    'Вероятность': f"{count / sum(targets.values()):.2%}"
                })

        df = pd.DataFrame(transitions).sort_values('Переходы', ascending=False)
        display(df.head(top_n))

        # Визуализация графа
        if visualize:
            self._visualize_markov_chain()

    def _visualize_markov_chain(self):
        plt.figure(figsize=(14, 10))
        G = nx.DiGraph()

        # Добавляем узлы и ребра
        for source, targets in self.markov_chain.items():
            G.add_node(source)
            for target, weight in targets.items():
                G.add_edge(source, target, weight=weight)

        # Позиционирование узлов
        pos = nx.spring_layout(G, k=0.5, iterations=50)

        # Рисуем узлы
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue', alpha=0.9)

        # Рисуем ребра с толщиной по количеству переходов
        edge_width = [d['weight'] * 0.1 for u, v, d in G.edges(data=True)]
        nx.draw_networkx_edges(G, pos, width=edge_width, edge_color='gray', arrowsize=20)

        # Подписи узлов (только id без префикса)
        labels = {node: node.replace('id_node_', '') for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_weight='bold')

        # Подписи ребер с количеством переходов
        edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

        plt.title(f'Визуализация марковской цепи "{self.NAME}"', size=15)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

