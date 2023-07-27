class Encoder():
    """
    This is far simpler than it looks. 

    We are simply taking characters from your input text, adding them to a list, and associating
    each character with its index in that list. For example:

    [
        0: "a",
        1: "l",
        2: "e",
        3: "x"
    ]

    This index will become its encoded integer value to be used within our language model. We will
    want to be able to encode and decode each value on command. 
    
    It's used throughout every gpt model, so we will keep this separate and reusable in this class.
    """

    def __init__(self, input_text):
        self.chars = sorted(list(set(input_text)))  # sorted unique characters in text
        self.vocab_size = len(self.chars)  # number of unique characters in text

    def character_index(self):
        # create a mapping from characters to integers based on the index of each character
        return {ch: i for i, ch in enumerate(self.chars)}

    def integer_index(self):
        # create a mapping from integers to characters based on the index of each character
        return {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text):
        character_index = self.character_index()
        def encode_text(s): return [character_index[c] for c in s]  # encoder: take a string, output a list of integers
        return encode_text(text)

    def decode(self, text):
        integer_index = self.integer_index()
        def decode_text(l): return ''.join([integer_index[i] for i in l])  # decoder: take a list of integers, output a string
        return decode_text(text)
