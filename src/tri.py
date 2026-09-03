import json
import numpy as np
from nltk.tokenize import word_tokenize

metadata_file = "/home/chatziko/PycharmProjects/PythonProject/IDMVAE/archive/UCA Image Dataset/UCA_Frame_data/train/metadata.jsonl"

lengths = []

with open(metadata_file, "r") as f:
    for line in f:
        sample = json.loads(line)

        tokens = word_tokenize(sample["caption"].lower())
        lengths.append(len(tokens))

lengths = np.array(lengths)

print("Number of captions:", len(lengths))
print("Mean:", lengths.mean())
print("Median:", np.median(lengths))
print("90th percentile:", np.percentile(lengths, 90))
print("95th percentile:", np.percentile(lengths, 95))
print("99th percentile:", np.percentile(lengths, 99))
print("Maximum:", lengths.max())

for max_sent_len in [32, 48, 64]:
    # -1 because one position is reserved for <eos>
    max_caption_tokens = max_sent_len - 1

    truncated = (lengths > max_caption_tokens).sum()

    print(
        f"maxSentLen={max_sent_len}: "
        f"{truncated}/{len(lengths)} "
        f"({100 * truncated / len(lengths):.2f}%) truncated"
    )