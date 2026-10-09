#!/usr/bin/env python3
"""An independent BERT tokenizer (ASCII), to check lib/wordpiece against.

Usage: transformers-wordpiece-check.py VOCAB.txt MAX_LENGTH < sentences

One sentence per input line; prints the padded input ids, space-separated, one
line each. Written from BERT's documented BasicTokenizer and WordpieceTokenizer
(clean, lower-case, split on punctuation, greedy longest match with a "##"
continuation, [UNK] for an unmatched word or one over 100 characters), sharing
no code with the OCaml implementation. It is a second implementation of the
specification, not the reference library: the reference tokenizer is not
available where this was written.
"""
import sys


def basic(text):
    cleaned = []
    for ch in text:
        o = ord(ch)
        if o == 0 or o == 0xFFFD or (o < 32 and ch not in "\t\n\r") or o == 127:
            continue
        cleaned.append(" " if ch in " \t\n\r" else ch)
    out = []
    for token in "".join(cleaned).strip().split():
        token = token.lower()
        piece = ""
        for ch in token:
            o = ord(ch)
            if 33 <= o <= 47 or 58 <= o <= 64 or 91 <= o <= 96 or 123 <= o <= 126:
                if piece:
                    out.append(piece)
                    piece = ""
                out.append(ch)
            else:
                piece += ch
        if piece:
            out.append(piece)
    return out


def wordpiece(vocab, word):
    if len(word) > 100:
        return ["[UNK]"]
    pieces, start = [], 0
    while start < len(word):
        end, found = len(word), None
        while start < end:
            cand = word[start:end]
            if start > 0:
                cand = "##" + cand
            if cand in vocab:
                found = cand
                break
            end -= 1
        if found is None:
            return ["[UNK]"]
        pieces.append(found)
        start = end
    return pieces


def main(vocab_path, max_length):
    max_length = int(max_length)
    vocab = {}
    for i, line in enumerate(open(vocab_path, encoding="utf8").read().split("\n")):
        if line == "" and i == len(vocab):
            continue
        vocab.setdefault(line, i)
    for line in sys.stdin:
        text = line.rstrip("\n")
        toks = [p for w in basic(text) for p in wordpiece(vocab, w)]
        toks = toks[: max_length - 2]
        ids = [vocab["[CLS]"]] + [vocab.get(t, vocab["[UNK]"]) for t in toks] + [vocab["[SEP]"]]
        ids += [vocab["[PAD]"]] * (max_length - len(ids))
        print(" ".join(map(str, ids)))


if __name__ == "__main__":
    main(*sys.argv[1:3])
