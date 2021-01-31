import json
import os
import random

import nltk
import numpy as np
import requests
import torch
import torch.nn as nn
import wget
from nltk.stem.porter import PorterStemmer
from torch import cuda as cuda
from torch import device as devicess
from torch import from_numpy as from_numpy
from torch import load as load
from torch import max as max
from torch import softmax as softmax
from torch.utils.data import Dataset, DataLoader

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
    # return device


def nltkpunkt():
    nltk.download('punkt')
    """Needed package from nltk called 'punkt' also need in code."""


def brain(is_file, data_file="", link=""):
    if is_file is True:
        FILE = data_file
        data = load(FILE)
        """If you have file data.pth then define its path and activate it like sneky.brain(is_file=True, data_file=PATH_TO_FILE). If you done this then you dont need sneky.ai_load()"""
        return data
    if is_file is False:
        if os.path.isfile('data.pth'):
            os.remove('data.pth')
            data = link
            wget.download(data, "data.pth")
        else:
            data = link
            wget.download(data, "data.pth")
    """If you have that file on website then just use sneky.brain(is_file=False, link='link_to_that_website'). that link must have link_to_that_website/data.pth"""


def clear():
    os.system('cls')
    """Clear whole terminal"""


def ai_json(is_database, database_file="", link=""):
    if is_database is True:
        ai = open(database_file)
        intents = json.load(ai)
        """If you have your database then just use sneky.ai_json(is_file=True, database_file=PATH_TO_THAT_FILE). database_file is mane_of_your_database.json"""
        return intents
    if is_database is False:
        ai = requests.get(url=str(link))
        intents = ai.json()
        """If you have your database online then just do it like this. sneky.ai_json(is_file=False, link='link_to_that_website'. that link must have link_to_that_website/name_of_your_database.json"""
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


def speak(variable_text, is_database, database_file="", link=""):
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
        for intent in ai_json(is_database, database_file=database_file, link=link)['intents']:
            if tag == intent["tag"]:
                output_text = random.choice(intent['responses'])
                return output_text

    else:
        output_text = 'I dont know'
        return output_text
    """This is your chat, dont forget about bot name."""


def help():
    print(
        "This package contains: device(), nltkpunkt(), brain(is_file, data_file, link), clear(), ai_json(is_file, database_file, link), ai_load(), recieve_text(text), speak(bot_name, variable_text).")


def training(database_file, how_long=10000):
    with open(database_file, "r") as f:
        intents = json.load(f)
    all_words = []
    tags = []
    xy = []
    for intent in intents['intents']:
        tag = intent['tag']
        tags.append(tag)
        for pattern in intent['patterns']:
            w = tokenize(pattern)
            all_words.extend(w)
            xy.append((w, tag))

    ignore_words = ["?", ".", ",", "!", "'"]
    all_words = [stem(w) for w in all_words if w not in ignore_words]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))

    x_train = []
    y_train = []
    for (pattern_sentence, tag) in xy:
        bag = bag_of_words(pattern_sentence, all_words)
        x_train.append(bag)

        label = tags.index(tag)
        y_train.append(label)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    class ChatDataset(Dataset):
        def __init__(self):
            self.n_samples = len(x_train)
            self.x_data = x_train
            self.y_data = y_train

        def __getitem__(self, index):
            return self.x_data[index], self.y_data[index]

        def __len__(self):
            return self.n_samples

    batch_size = 8
    hidden_size = 8
    output_size = len(tags)
    input_size = len(x_train[0])
    learning_rate = 0.001
    num_epochs = int(how_long)

    dataset = ChatDataset()
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    device = devicess('cuda' if cuda.is_available() else 'cpu')
    clear()
    model = NeuralNet(input_size, hidden_size, output_size).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for (words, labels) in train_loader:
            words = words.to(device)
            labels = labels.to(dtype=torch.long).to(device)

            # Forward pass
            outputs = model(words)
            # if y would be one-hot, we must apply
            # labels = torch.max(labels, 1)[1]
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}%')

    print(f'final loss: {loss.item():.4f}%')

    data = {
        "model_state": model.state_dict(),
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size,
        "all_words": all_words,
        "tags": tags
    }

    FILE = "data.pth"
    torch.save(data, FILE)

    print(f'training complete. file saved to {FILE}')
    """Make your assistant learn new stuff from database"""
