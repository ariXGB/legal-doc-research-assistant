MAX_HISTORY_TURNS = 5

def add_turn(history, question, answer):
    history.append({"question": question, "answer": answer})
    if len(history) > MAX_HISTORY_TURNS:
        history.pop(0)
    return history


def clear_history():
    return []