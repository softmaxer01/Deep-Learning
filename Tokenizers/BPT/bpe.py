from collections import Counter

class BPE:
    def __init__(self,text,vocab_size):
        self.itr = vocab_size - 256
        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        tokens = text.encode("utf-8")
        self.tokens = list(map(int,tokens))
        self.build_merge_table(tokens,self.itr)
        self.build_vocab()

    def get_stats(self, ids):
        return Counter(zip(ids[:-1], ids[1:]))

    def merge(self, ids, pair, idx):
        new_ids = []
        i = 0
        pair_0, pair_1 = pair
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair_0 and ids[i+1] == pair_1:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def build_merge_table(self, tokens, iterations):
        tokens = list(tokens)
        for i in range(iterations):
            stats = self.get_stats(tokens)
            if not stats:
                break
            pair, freq = stats.most_common(1)[0]
            idx = 256 + i
            tokens = self.merge(tokens, pair, idx)
            self.merges[pair] = idx
        return tokens, self.merges

    def build_vocab(self):
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        return self.vocab

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while True:
            stats = self.get_stats(tokens)
            if not stats:
                break
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8", errors="replace")