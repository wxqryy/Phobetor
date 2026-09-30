import unicodedata


ALPHABET = '\n' + ''.join(chr(code) for code in range(32, 127))
CHAR_TO_ID = {character: index for index, character in enumerate(ALPHABET)}
VOCAB_SIZE = len(ALPHABET)
REPLACEMENTS = str.maketrans({
    '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
    '\u2013': '-', '\u2014': '-', '\u2212': '-', '\u2026': '...',
    '\u00a0': ' ', '\t': ' ', '\r': '\n', '\u00df': 'ss',
})


def normalize(text):
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = unicodedata.normalize('NFKD', text.translate(REPLACEMENTS))
    text = ''.join(character for character in text if not unicodedata.combining(character))
    return ''.join(character if character in CHAR_TO_ID else ' ' for character in text)


def encode(text):
    return [CHAR_TO_ID[character] for character in normalize(text)]


def decode(ids):
    return ''.join(ALPHABET[index] for index in ids)
