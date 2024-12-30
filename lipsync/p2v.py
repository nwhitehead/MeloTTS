
# ARPABET symbols used by MeloTTS
# Includes numeric stress level

arpa = {
    'AH0',
    'S',
    'AH1',
    'EY2',
    'AE2',
    'EH0',
    'OW2',
    'UH0',
    'NG',
    'B',
    'G',
    'AY0',
    'M',
    'AA0',
    'F',
    'AO0',
    'ER2',
    'UH1',
    'IY1',
    'AH2',
    'DH',
    'IY0',
    'EY1',
    'IH0',
    'K',
    'N',
    'W',
    'IY2',
    'T',
    'AA1',
    'ER1',
    'EH2',
    'OY0',
    'UH2',
    'UW1',
    'Z',
    'AW2',
    'AW1',
    'V',
    'UW2',
    'AA2',
    'ER',
    'AW0',
    'UW0',
    'R',
    'OW1',
    'EH1',
    'ZH',
    'AE0',
    'IH2',
    'IH',
    'Y',
    'JH',
    'P',
    'AY1',
    'EY0',
    'OY2',
    'TH',
    'HH',
    'D',
    'ER0',
    'CH',
    'AO1',
    'AE1',
    'AO2',
    'OY1',
    'AY2',
    'IH1',
    'OW0',
    'L',
    'SH',
}

phonemes = set([p.rstrip('012') for p in arpa])

# Viseme set is one proposed set, there are many possible sets.

visemes = { 'sil', 'PP', 'FF', 'TH', 'DD', 'kk', 'CH', 'SS', 'nn', 'RR', 'aa', 'E', 'I', 'O', 'U' }

# Map goes from phoneme to list of visemes.

# Decision about what to put on right side was made by looking at pictures of phonemes and pictures of visemes and picking
# things that looked reasonably close.

p2v_map = {
    '_': ['sil'],
    'AA': ['aa'],
    'AE': ['aa'],
    'AH': ['aa'],
    'AO': ['aa'],
    'AW': ['aa', 'U'],
    'AY': ['aa', 'I'],
    'B': ['PP'],
    'CH': ['CH'],
    'D': ['DD'],
    'DH': ['DD'],
    'EH': ['E'],
    'ER': ['E'],
    'EY': ['E', 'I'],
    'F': ['FF'],
    'G': ['kk'],
    'HH': ['aa'],
    'IH': ['I'],
    'IY': ['I'],
    'JH': ['CH'],
    'K': ['kk'],
    'L': ['nn'],
    'M': ['PP'],
    'N': ['nn'],
    'NG': ['kk'],
    'OW': ['O', 'U'],
    'OY': ['O', 'I'],
    'P': ['PP'],
    'R': ['RR'],
    'S': ['SS'],
    'SH': ['CH'],
    'T': ['DD'],
    'TH': ['TH'],
    'UH': ['U'],
    'UW': ['U'],
    'V': ['FF'],
    'W': ['U'],
    'Y': ['kk'],
    'Z': ['kk'],
    'ZH': ['CH'],
}

def to_visemes(phoneme_list, timestamps):
    """Convert list of phonemes and timestamps to list of visemes and timestamps (output length may change)"""
    # phoneme_list should include silence at end, "_" phoneme, so we have timestamp of end of last voiced phoneme
    # OR the timestamps is one longer to include final time of last phoneme end
    assert ((len(phoneme_list) == len(timestamps)) and (len(phoneme_list) > 0) and (phoneme_list[-1] == ',')) or ((len(phoneme_list) + 1 == len(timestamps)))
    if len(phoneme_list) == len(timestamps):
        phonemes = phoneme_list[:-1]
    else:
        phonemes = phoneme_list[:]
    visemes = []
    vtimestamps = []
    for i, (p, t) in list(enumerate(zip(phoneme_list, timestamps))):
        ph = p.rstrip('012').upper()
        if ph in p2v_map:
            v = p2v_map[ph]
        else:
            v = ['sil']
        visemes.extend(v)
        end_time = timestamps[i + 1]
        n = len(v)
        vtimestamps.extend([round(t + i / n * (end_time - t)) for i in range(n)])
    return visemes, vtimestamps

def test_data():
    # Should be 71 entries, 39 unique without stress number
    assert len(arpa) == 71
    assert len(phonemes) == 39

    for p in p2v_map:
        assert (p in phonemes) or (p in  ['_', '-'])
        for v in p2v_map[p]:
            assert v in visemes

    assert to_visemes(['Y', 'UW1'], [0, 0.1, 0.2]) == (['kk', 'U'], [0.0, 0.1])
    assert to_visemes(['HH', 'AH0', 'L', 'OW1'], [0, 0.1, 0.2, 0.3, 0.4]) == (['aa', 'aa', 'nn', 'O', 'U'], [0.0, 0.1, 0.2, 0.3, 0.35])

if __name__ == '__main__':
    test_data()
