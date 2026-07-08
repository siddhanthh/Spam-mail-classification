import re

def get_words(text):
    return re.findall(r'[a-zA-Z0-9]+', text.lower())

def extract_features(text):
    features = []
    
    WORDS = [
        "make", "address", "all", "3d", "our", "over", "remove", "internet", 
        "order", "mail", "receive", "will", "people", "report", "addresses", 
        "free", "business", "email", "you", "credit", "your", "font", "000", 
        "money", "hp", "hpl", "george", "650", "lab", "labs", "telnet", "857", 
        "data", "415", "85", "technology", "1999", "parts", "pm", "direct", 
        "cs", "meeting", "original", "project", "re", "edu", "table", "conference"
    ]
    
    CHARS = [";", "(", "[", "!", "$", "#"]
    
    words = get_words(text)
    total_words = len(words) if len(words) > 0 else 1
    total_chars = len(text) if len(text) > 0 else 1
    
    # Word frequencies
    for w in WORDS:
        count = words.count(w)
        freq = 100.0 * count / total_words
        features.append(freq)
        
    # Char frequencies
    for c in CHARS:
        count = text.count(c)
        freq = 100.0 * count / total_chars
        features.append(freq)
        
    # Capital run length features
    capital_runs = re.findall(r'[A-Z]+', text)
    
    if capital_runs:
        run_lengths = [len(r) for r in capital_runs]
        cr_average = sum(run_lengths) / len(run_lengths)
        cr_longest = max(run_lengths)
        cr_total = sum(run_lengths)
    else:
        cr_average = 1.0  # Dataset uses 1 if none
        cr_longest = 1
        cr_total = 0
        
    features.append(cr_average)
    features.append(cr_longest)
    features.append(cr_total)
    
    return features
