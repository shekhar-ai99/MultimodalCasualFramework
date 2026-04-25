def generate_explanation(state, action, model) -> str:
    return f"If action {action} is taken, predicted outcome improves."


def gated_explanation(q_values, conformal_set, state, model):
    if len(conformal_set) == 1:
        action = conformal_set[0]
        return action, generate_explanation(state, action, model)
    return None, "High uncertainty - abstain"
