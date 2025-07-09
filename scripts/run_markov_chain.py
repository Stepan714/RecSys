from yambda.dataset import YambdaDataset
from yambda.preprocessing import filter_user_history_by_time
from markov_chain.markov_chain import build_markov_chain_grouped
from markov_chain.json_markov_chain import save_markov_chain

def main():
    print("[1] Загружаем датасет...")
    dataset = YambdaDataset("sequential", "50m")
    events = dataset.interaction("multi_event")

    print("[2] Фильтруем по времени...")
    start_time = 10_000_000
    end_time = 10_500_000
    filtered_events = filter_user_history_by_time(events, start_time, end_time)
    print(f"[2.1] После фильтрации осталось пользователей: {len(filtered_events)}")

    print("[3] Строим марковскую цепь...")
    markov_chain = build_markov_chain_grouped(filtered_events)
    print(f"[3.1] Количество состояний: {len(markov_chain)}")

    print("[4] Сохраняем в JSON...")
    save_markov_chain(markov_chain, "saved_chains/markov_chain_test.json")
    print("[5] Готово ✅")



if __name__ == "__main__":
    main()