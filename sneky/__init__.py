import os
import nltk
import numpy as np
from torch import max as max
from torch import softmax as softmax
from torch import from_numpy as from_numpy
from torch import load as load
from torch import device as devicess
from torch import cuda as cuda
import torch.nn as nn
import wget
import random
import requests
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return stemmer.stem(word.lower())
def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out
def device():
    devicess('cuda' if cuda.is_available() else 'cpu')
    """This thing must be implemented in your sneky so without this you cant use it."""
    #return device
def nltkpunkt():
    nltk.download('punkt')
    """Needed package from nltk called 'punkt' also need in code."""
def brain():
    if os.path.isfile("data.pth"):
        os.remove('data.pth')
        data = "http://randomguybay.github.io/sneky-licence/main/data.pth"
        wget.download(data, "data.pth")
    else:
        data = "http://randomguybay.github.io/sneky-licence/main/data.pth"
        wget.download(data, "data.pth")
    """This module try to download needed data to work fine. In next version of package there will be training ability so you can use your own data.pth."""
def clear():
    os.system('cls')
    """Clear whole terminal"""
def ai_json():
    ai = requests.get("https://randomguybay.github.io/sneky-licence/main/intents.json")
    intents = ai.json()
    """This will get sentences 'database' also in next version you will have ability to set your own sentece 'database'."""
    return intents
def ai_load():
    FILE = "data.pth"
    data = load(FILE)
    """This will load data to work fine."""
    return data
def recieve_text(text):
    sentence = str(text)
    """Recieve your text, but you have to define that text from variable."""
    return sentence
def speak(bot_name, variable_text):
    FILE = "data.pth"
    data = load(FILE)
    input_size = data['input_size']
    hidden_size = data['hidden_size']
    output_size = data['output_size']
    all_words = data['all_words']
    tags = data['tags']
    model_state = data['model_state']
    model = NeuralNet(input_size, hidden_size, output_size).to(device())
    model.load_state_dict(model_state)
    model.eval()
    """This will load sertain data from data."""
    sentence = tokenize(recieve_text(variable_text))
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = from_numpy(x)

    output = model(x)
    _, predicted = max(output, dim=1)
    tag = tags[predicted.item()]

    probs = softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.99:
        for intent in ai_json()['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")

    else:
        print(f"{bot_name}: dunno")
    """This is your chat, dont forget about bot name."""
def help():
    print("This package contains: device(), nltkpunkt(), brain(), clear(), ai_json(), ai_load(), recieve_text(text), speak(bot_name, variable_text).")