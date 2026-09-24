import numpy as np

from scripts.probe_match_model import SHORTEST, match_stream


def test_match_stream_is_causal_and_exact():
    rng = np.random.default_rng(0)
    words = [bytes(rng.integers(97, 101, size=rng.integers(2, 7))) for _ in range(30)]
    doc = np.frombuffer(b' '.join(words[i] for i in rng.integers(0, 30, size=400)), dtype=np.uint8)
    b = bytes(doc)
    length, proposal, source = match_stream(doc)
    for t in range(len(b)):
        seen = b[:t]
        if length[t]:
            L, s = int(length[t]), int(source[t])
            assert s < t and proposal[t] == b[s] and L >= SHORTEST
            assert b[s - L:s] == b[t - L:t]
        # A proposal exists whenever the last SHORTEST bytes occurred before.
        if t >= SHORTEST and b[t - SHORTEST:t] in seen[:-1]:
            assert length[t] > 0
        # Changing the future never changes the proposal.
        cut = np.concatenate([doc[:t], np.full(len(b) - t, 0, np.uint8)])
        l2, p2, s2 = match_stream(cut)
        assert (l2[t], p2[t], s2[t]) == (length[t], proposal[t], source[t])


def test_match_stream_reads_bytes_not_machine_words():
    doc = np.frombuffer(b'abcdefgh abcdefgh', dtype=np.uint8)
    wide = match_stream(doc.astype(np.int64))
    narrow = match_stream(doc)
    assert all((w == n).all() for w, n in zip(wide, narrow))
    assert narrow[0][9 + SHORTEST] == SHORTEST and narrow[1][9 + SHORTEST] == ord('f')
