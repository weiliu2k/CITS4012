def lexical_diversity(text):
    return len(set(text)) / len(text)

def tf(text, token):
    count = text.count(token)
    total = len(text)
    return 100 * count / total    