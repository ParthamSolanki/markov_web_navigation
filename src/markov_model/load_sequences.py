from .config import SEQ_FILE


def load_msnbc_sequences(file_path=SEQ_FILE):
    sequences = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            seq = [int(x) for x in line.split() if x.isdigit()]
            if seq:
                sequences.append(seq)
    return sequences
