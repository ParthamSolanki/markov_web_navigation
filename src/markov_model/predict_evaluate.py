import math
import random
from collections import Counter


# Calculating & returning the benchmarking metrics after using Laplace smoothing.
def predict_and_evaluate(model, test_sequences, k, alpha=1.0, num_states=17):
    total_log_likelihood = 0.0
    correct_predictions = 0
    total_predictions = 0

    for seq in test_sequences:
        if len(seq) <= k:
            continue

        for i in range(len(seq) - k):
            history = tuple(seq[i : i + k])
            actual_next = seq[i + k]

            counts = model.get(history, Counter())
            total_observations = sum(counts.values())
            denominator = total_observations + (alpha * num_states)

            # Top 1 prediction, with a fallback if the data doesn't exists, so it doesn't throw error.
            if total_observations == 0:
                best_state = random.randint(1, num_states)
            else:
                best_state = max(
                    range(1, num_states + 1), key=lambda s: counts.get(s, 0)
                )

            if best_state == actual_next:
                correct_predictions += 1

            # log-likelihood for the observed state
            actual_prob = (counts.get(actual_next, 0) + alpha) / denominator
            total_log_likelihood += math.log(actual_prob)
            total_predictions += 1

    if total_predictions == 0:
        return 0.0, 0.0

    avg_log_likelihood = total_log_likelihood / total_predictions
    accuracy = correct_predictions / total_predictions

    return avg_log_likelihood, accuracy
