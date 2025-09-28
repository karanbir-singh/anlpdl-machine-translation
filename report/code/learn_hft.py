def learn_hft(input_file: str, target_vocab_size=4000, k_fraction=0.05) -> Dict:
    """
    Learn HFT vocabulary from an input file.

    Args:
        input_file (str): the file from which the vocabulary is learned 
        target_vocab_size (int, optional): wanted vocabulary size. Defaults to 4000.
        k_fraction (float, optional): fraction of the target vocabulary size. Defaults to 0.05.

    Returns:
        Dict: learned vocabulary
    """
    
    # Initial vocabulary = all characters
    char_counts = Counter()
    for tokens in load_stream(input_file):
        for token in tokens:
            for ch in token:
                char_counts[ch] += 1
    vocab = dict(char_counts) # {token: freq}

    # Represent each token as a list of current subwords
    segmented_tokens = []
    for tokens in load_stream(input_file):
        segmented_line = []
        for token in tokens:
            segmented_line.append(list(token))
        segmented_tokens.append(segmented_line)

    # Loop until reaching target vocab size
    while len(vocab) < target_vocab_size:
        # Count subword frequencies and candidate pairs
        subword_freq = Counter()
        pair_freq = Counter()
        for line in segmented_tokens:
            for token_subwords in line:
                # Subword frequencies
                for subword in token_subwords:
                    subword_freq[subword] += 1
                    
                # Pairs
                for i in range(len(token_subwords) - 1):
                    pair = (token_subwords[i], token_subwords[i+1])
                    pair_freq[pair] += 1

        # Select top K candidates
        k = max(1, int(k_fraction * (target_vocab_size - len(vocab))))
        top_pairs = pair_freq.most_common(k)
        if not top_pairs:
            # print("No more pairs to merge.")
            break

        # Merge subwords according to top pairs
        merges = {pair: pair[0] + pair[1] for pair, _ in top_pairs}
        
        # Add to vocab
        for pair, freq in top_pairs:
            new_token = merges[pair]
            vocab[new_token] = freq

        # Remove rare subwords
        min_freq = top_pairs[-1][1]
        for token in list(vocab.keys()):
            if len(token) > 1 and vocab[token] < min_freq:
                del vocab[token]

        # Update segmented tokens
        for line in segmented_tokens:
            for i, token_subwords in enumerate(line):
                j = 0
                new_subwords = []
                while j < len(token_subwords):
                    if j < len(token_subwords)-1:
                        pair = (token_subwords[j], token_subwords[j+1])
                        if pair in merges:
                            new_subwords.append(merges[pair])
                            j += 2
                            continue
                    new_subwords.append(token_subwords[j])
                    j += 1
                line[i] = new_subwords
    return vocab