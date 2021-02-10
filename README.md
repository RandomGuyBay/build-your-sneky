Build your own assistant/voice assistant with this basic AI chat.

Newly added brain algorithm

# Description
    
It consists of nine main modules:

- `device`: This thing must be implemented in your sneky so without this you cant use it
- `nltkpunkt`: Needed package from nltk called 'punkt' also need in code
- `brain`: This module try to download needed data to work fine. In next version of package there will be training ability so you can use your own data.pth
- `clear`: Clear whole terminal
- `ai_json`: This will get sentences 'database' also in next version you will have ability to set your own sentece 'database'
- `ai_load`: This will load data to work fine
- `recieve_text`: Recieve your text, but you have to define that text from variable
- `speak`: This is your chat, dont forget about bot name
- `training`: This module trains your sneky from database. outputs data.pth that is needed

## New thing called "Brain algorith"

- brain aglorithm adds thinking before answer
    So if you ask "What do you love"
    I wont anymore answer "I like"
    
    you dont need to change anything

# Installation
 
## Normal installation

```bash
pip install build-your-sneky
```

## Source installation

```bash
python setup.py install
```

# Usage

## Usage with files

```py
import sneky
sneky.device()
sneky.nltkpunkt()
sneky.brain(is_file=True, data_file='PATH_TO_FILE.pth')
sneky.clear()
sneky.ai_json(is_database=True, database_file='PATH_TO_FILE.json')
while True:
    text = input("You: ")
    print(sneky.speak(text, is_database=True, database_file='PATH_TO_FILE.json')) - or if you want you can store that answer as variable
```

## usage with links

```py
import sneky
sneky.device()
sneky.nltkpunkt()
sneky.brain(is_file=False, link='link_to_that_website/data.pth')
sneky.clear()
sneky.ai_json(is_database=False, link='link_to_that_website/name_of_your_database.json')
while True:
    text = input("You: ")
    print(sneky.speak(text, is_database=False, link='link_to_that_website/name_of_your_database.json')) - or if you want you can store that answer as variable
```

# training usage

```py
import sneky
sneky.training(database_file=PATH_TO_FILE.json, how_long=NUMBER) if you dont use number default is 10000
```

# database template
```json
{
  "intents": [
    {
      "tag": "greeting",
      "patterns": [
        "Hi",
        "Hey"
      ],
      "responses": [
        "Hey",
        "Hello"
      ]
    },
    {
      "tag": "favorite",
      "patterns": [
        "what do you like?",
        "what is your favorite thing?",
        "what do you love?"
      ],
      "responses": [
        "I like snakes",
        "I like green",
        "I like music",
        "I love electric cars"
      ]
    }
  ]
}
```