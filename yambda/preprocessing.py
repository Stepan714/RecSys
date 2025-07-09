from datasets import Dataset

def filter_user_history_by_time(events: Dataset, start_time: int, end_time: int) -> Dataset:
    def keep_and_truncate(example):
        mask = [(start_time <= ts < end_time) for ts in example["timestamp"]]

        if not any(mask):
            return None

        for key in example.keys():
            if isinstance(example[key], list) and len(example[key]) == len(mask):
                example[key] = [v for v, m in zip(example[key], mask) if m]

        return example

    filtered = events.map(keep_and_truncate, remove_columns=[], desc="Filtering by time")
    filtered = filtered.filter(lambda ex: ex is not None)

    return filtered