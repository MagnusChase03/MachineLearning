import numpy as np

from NeuralNetwork import NeuralNetwork

dictionary = []
for i in range(65, 65 + 26):
    dictionary.append(chr(i))

for i in range(97, 97 + 26):
    dictionary.append(chr(i))

dictionary.append('.')
dictionary.append('\'')
dictionary.append(' ')
dictionary.append('\n')


def encode(string):
    encoding = []
    for char in string:
        if char in dictionary:
            encoding.append(dictionary.index(char))

    return encoding

def decode(encoding):
    string = ""
    for index in encoding:
        string += dictionary[index]

    return string

def main():
    network = NeuralNetwork(8, 6, 1, 0.01);

main()
