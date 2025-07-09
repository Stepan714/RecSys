from collections import defaultdict

def build_markov_chain_grouped(dataset):
    # Собираем пользователей, совершивших переход (s1 → s2)
    transition_users = defaultdict(set)       # (s1, s2) → set(user_ids)
    from_state_users = defaultdict(set)       # s1 → set(user_ids)

    for idx, user in enumerate(dataset):
        item_ids = user["item_id"]
        event_types = user["event_type"]
        user_id = idx  # можно заменить на user["user_id"], если есть
        #user_id = user["uid"]

        states = list(zip(item_ids, event_types))

        for s1, s2 in zip(states[:-1], states[1:]):
            transition_users[(s1, s2)].add(user_id)
            from_state_users[s1].add(user_id)

    # Структура: state_from → {state_to: prob}
    transition_probs = defaultdict(dict)

    for (s1, s2), users in transition_users.items():
        total_from = len(from_state_users[s1])
        prob = len(users) / total_from if total_from > 0 else 0.0
        transition_probs[s1][s2] = prob

    return transition_probs
