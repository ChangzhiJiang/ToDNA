"""Compare the recovered helper with an independent all-pairs Hamming oracle."""
import argparse
import random
import subprocess
import tempfile
from pathlib import Path


def rc(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def test(binary, length):
    rng = random.Random(918)
    threshold = length // 10
    sequences = ["".join(rng.choices("ACGT", k=length)) for _ in range(60)]
    sequences += [sequences[0], rc(sequences[1])]
    for anchor, edits in enumerate((1, threshold-1, threshold, threshold+1, 2*threshold), 2):
        mutated = list(sequences[anchor])
        for i in rng.sample(range(length), edits):
            mutated[i] = next(c for c in "ACGT" if c != mutated[i])
        sequences += ["".join(mutated), rc("".join(mutated))]
    # A--B--C is connected even though A and C exceed the pair threshold.
    a = sequences[10]
    b = "".join(next(c for c in "ACGT" if c != base) if i < threshold else base
                for i, base in enumerate(a))
    c = "".join(next(x for x in "ACGT" if x != base) if i < 2*threshold else base
                for i, base in enumerate(a))
    sequences += [b, c]
    parent = list(range(len(sequences)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    for i, a in enumerate(sequences):
        for j, b in enumerate(sequences[:i]):
            distance = min(sum(x != y for x, y in zip(a, b)),
                           sum(x != y for x, y in zip(a, rc(b))))
            if distance <= threshold:
                parent[find(i)] = find(j)
    with tempfile.TemporaryDirectory() as directory:
        inp, out = Path(directory)/"sequences.txt", Path(directory)/"components.txt"
        inp.write_text("\n".join(sequences) + "\n")
        subprocess.run([str(binary), str(inp), str(out)], check=True, capture_output=True)
        labels = list(map(int, out.read_text().split()))
        assert len(labels) == len(sequences)
        for i in range(len(sequences)):
            for j in range(len(sequences)):
                assert (labels[i] == labels[j]) == (find(i) == find(j)), (length, i, j)
    print(f"PASS: {length} bp, <= {threshold} edits, duplicates, RC, boundary and transitivity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    for length in (200, 250):
        test(args.binary.resolve(), length)
