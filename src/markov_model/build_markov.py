from collections import Counter, defaultdict


# This function builds the k-order Markov Model transition tables.
# Also prunes history contexts observed fewer than tau times optionally if value passed.
def build_markov_model(sequences, k, tau=0):
    transitions = defaultdict(Counter)

    for seq in sequences:
        if len(seq) <= k:
            continue
        for i in range(len(seq) - k):
            history = tuple(seq[i : (i + k)])
            next_state = seq[i + k]
            transitions[history][next_state] += 1

    if tau > 0:
        return {
            h: counts
            for h, counts in transitions.items()
            if sum(counts.values()) >= tau
        }

    return dict(transitions)


# Converting to dict before returning as due to defualt dict if you access a non existing key it creates one on the fly which is not ideal.
# Like a["Hello"] even if doesn't exist in the source seq file it would be created on the fly with an empty counter causing issues.
