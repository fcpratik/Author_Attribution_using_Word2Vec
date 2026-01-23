import re

# Regex pattern:
# 1) words with optional apostrophes: don't, i'm
# 2) numbers
# 3) punctuation as separate tokens
TOKEN_PATTERN = re.compile(
    r"[a-z]+(?:'[a-z]+)?|\d+|[.,!?;:()\"-]"
)
# The re.compile() method in Python returns a regular expression object,
# specifically a re.Pattern object

def tokenize(text: str):
    text = text.lower()
    tokens = TOKEN_PATTERN.findall(text)
    return tokens
