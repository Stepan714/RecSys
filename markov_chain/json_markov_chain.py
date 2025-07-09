import json
from collections import defaultdict


def save_markov_chain(markov_chain: defaultdict, filepath: str) -> None:
    serializable_chain = {
        str(from_state): {str(to_state): prob for to_state, prob in to_states.items()}
        for from_state, to_states in markov_chain.items()
    }

    with open(filepath, "w") as f:
        json.dump(serializable_chain, f, indent=2)


def load_markov_chain(filepath: str) -> defaultdict:
    with open(filepath, "r") as f:
        data = json.load(f)

    chain = defaultdict(dict)
    for from_state_str, to_states in data.items():
        from_state = eval(from_state_str)
        chain[from_state] = {eval(to_state): prob for to_state, prob in to_states.items()}

    return chain
