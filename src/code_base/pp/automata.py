import ahocorasick
from .labels import known_labels

def create_automata(known_labels):
    A = ahocorasick.Automaton()
    for label in known_labels:
        if label:
            A.add_word(label, label)
    A.make_automaton()
    return A

Automata = create_automata(known_labels=known_labels)