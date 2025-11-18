"""
Noise generators for creating different types of noisy queries
"""
import random
import re
from typing import List, Tuple

# Set random seed for reproducibility
random.seed(42)

def add_typos(query: str, typo_rate: float = 0.15) -> str:
    """
    Add random typos to a query.

    Typo types:
    - Character deletion
    - Character insertion
    - Character substitution
    - Character swap (transpose adjacent characters)

    Args:
        query: Original query string
        typo_rate: Probability of introducing a typo per word (default 0.15)

    Returns:
        Query with typos introduced
    """
    words = query.split()
    noisy_words = []

    for word in words:
        # Skip very short words
        if len(word) <= 2 or random.random() > typo_rate:
            noisy_words.append(word)
            continue

        # Choose a random typo type
        typo_type = random.choice(['delete', 'insert', 'substitute', 'swap'])

        # Choose a random position (not first or last character for better readability)
        if len(word) > 3:
            pos = random.randint(1, len(word) - 2)
        else:
            pos = random.randint(0, len(word) - 1)

        if typo_type == 'delete' and len(word) > 3:
            # Delete a character
            word = word[:pos] + word[pos+1:]
        elif typo_type == 'insert':
            # Insert a random character
            char = random.choice('abcdefghijklmnopqrstuvwxyz')
            word = word[:pos] + char + word[pos:]
        elif typo_type == 'substitute':
            # Substitute with a nearby key on QWERTY keyboard
            keyboard_neighbors = {
                'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'sfcxe', 'e': 'wrds',
                'f': 'dgcvrt', 'g': 'fhbvty', 'h': 'gjnbyu', 'i': 'ujko', 'j': 'hknmu',
                'k': 'jlmio', 'l': 'kop', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
                'p': 'ol', 'q': 'wa', 'r': 'etfd', 's': 'wedxza', 't': 'ryfg',
                'u': 'yihj', 'v': 'cfgb', 'w': 'qeas', 'x': 'zsdc', 'y': 'tugh',
                'z': 'asx'
            }
            char = word[pos].lower()
            if char in keyboard_neighbors:
                new_char = random.choice(keyboard_neighbors[char])
                word = word[:pos] + new_char + word[pos+1:]
        elif typo_type == 'swap' and len(word) > 2 and pos < len(word) - 1:
            # Swap adjacent characters
            word = word[:pos] + word[pos+1] + word[pos] + word[pos+2:]

        noisy_words.append(word)

    return ' '.join(noisy_words)


def make_ambiguous(query: str) -> str:
    """
    Make a query ambiguous by shortening it or using abbreviations.

    Strategies:
    - Remove question words and keep only key terms
    - Use abbreviations for common words
    - Remove context that clarifies meaning

    Args:
        query: Original query string

    Returns:
        Ambiguous version of the query
    """
    # Common abbreviations that could be ambiguous
    abbreviations = {
        'information': 'info',
        'organization': 'org',
        'government': 'gov',
        'company': 'co',
        'corporation': 'corp',
        'department': 'dept',
        'management': 'mgmt',
        'development': 'dev',
        'application': 'app',
        'temperature': 'temp',
        'average': 'avg',
        'maximum': 'max',
        'minimum': 'min',
        'doctor': 'dr',
        'professor': 'prof',
        'university': 'uni',
        'business': 'biz',
        'technology': 'tech',
    }

    # Question words to remove
    question_words = ['what', 'where', 'when', 'why', 'how', 'who', 'which',
                      'is', 'are', 'was', 'were', 'does', 'do', 'did', 'can',
                      'could', 'would', 'should', 'the', 'a', 'an']

    words = query.lower().split()

    # Strategy 1: Remove question words and articles (50% chance)
    if random.random() < 0.5 and len(words) > 3:
        words = [w for w in words if w not in question_words]
        # Keep at least 2-3 words
        if len(words) > 3:
            words = words[:random.randint(2, 3)]

    # Strategy 2: Use abbreviations (50% chance)
    if random.random() < 0.5:
        words = [abbreviations.get(w, w) for w in words]

    # Strategy 3: Use acronyms for multi-word terms (30% chance)
    if random.random() < 0.3 and len(words) >= 3:
        # Take first letters of first 2-3 words
        num_words = min(random.randint(2, 3), len(words))
        acronym = ''.join([w[0].upper() for w in words[:num_words] if w])
        return acronym

    return ' '.join(words) if words else query


def make_context_dependent(query: str) -> str:
    """
    Transform query into a context-dependent follow-up question.

    Strategies:
    - Use pronouns like "it", "this", "that"
    - Create follow-up questions like "how does it work?"
    - Remove specific entities and use general references

    Args:
        query: Original query string

    Returns:
        Context-dependent version of the query
    """
    follow_up_templates = [
        "How does it work?",
        "Tell me more",
        "What about this?",
        "Can you explain that?",
        "Why is that?",
        "What does this mean?",
        "How do I use it?",
        "When should I do this?",
        "Is it the same?",
        "What's the difference?",
        "How much is it?",
        "Where can I find it?",
        "Who is responsible for this?",
        "What are the requirements?",
        "How long does it take?",
        "What happens next?",
    ]

    # Strategy 1: Direct follow-up question (60% chance)
    if random.random() < 0.6:
        return random.choice(follow_up_templates)

    # Strategy 2: Replace specific terms with pronouns
    words = query.split()

    # Find potential nouns to replace (words that are capitalized or longer words)
    if len(words) > 3:
        # Replace a noun with a pronoun
        replacements = {
            'person': 'they',
            'place': 'there',
            'thing': 'it',
            'this': 'this',
            'that': 'that'
        }

        # Simple heuristic: replace the last significant word with a pronoun
        for i in range(len(words) - 1, -1, -1):
            if len(words[i]) > 4 and words[i].lower() not in ['what', 'where', 'when', 'which']:
                words[i] = random.choice(['it', 'this', 'that'])
                break

        return ' '.join(words)

    return random.choice(follow_up_templates)


def make_adversarial(query: str) -> str:
    """
    Create adversarial versions by adding constraints.

    Strategies:
    - Add constraints to avoid key terms
    - Request specific output formats
    - Add contradictory requirements
    - Request unusual perspectives

    Args:
        query: Original query string

    Returns:
        Adversarial version with constraints
    """
    words = query.split()

    # Extract potential key terms (longer words, likely to be important)
    key_terms = [w for w in words if len(w) > 4 and w.lower() not in
                 ['what', 'where', 'when', 'which', 'about', 'should', 'would', 'could']]

    adversarial_templates = [
        # Constraint-based
        lambda q, terms: f"{q} without using the word '{random.choice(terms)}'" if terms else q,
        lambda q, terms: f"{q} without technical jargon" if terms else q,
        lambda q, terms: f"{q} (explain like I'm five)" if terms else q,
        lambda q, terms: f"{q} in one sentence" if terms else q,
        lambda q, terms: f"{q} using only common words" if terms else q,

        # Format-based
        lambda q, terms: f"{q} as a bullet list" if terms else q,
        lambda q, terms: f"{q} in simple terms" if terms else q,
        lambda q, terms: f"{q} with examples" if terms else q,

        # Perspective-based
        lambda q, terms: f"{q} from a beginner's perspective" if terms else q,
        lambda q, terms: f"{q} pros and cons only" if terms else q,
        lambda q, terms: f"{q} (assume I know nothing about this)" if terms else q,

        # Contradictory
        lambda q, terms: f"{q} but briefly and comprehensively" if terms else q,
        lambda q, terms: f"{q} in detail but quickly" if terms else q,
    ]

    # Choose a random adversarial transformation
    template = random.choice(adversarial_templates)
    return template(query, key_terms)


def extract_key_term(query: str) -> str:
    """
    Extract a key term from the query for adversarial constraints.

    Args:
        query: Query string

    Returns:
        A key term from the query
    """
    words = query.split()
    # Filter out common words and short words
    stop_words = {'what', 'where', 'when', 'why', 'how', 'who', 'which',
                  'is', 'are', 'was', 'were', 'the', 'a', 'an', 'and', 'or',
                  'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    key_words = [w for w in words if len(w) > 3 and w.lower() not in stop_words]

    if key_words:
        return random.choice(key_words)
    elif words:
        return random.choice(words)
    else:
        return "it"
