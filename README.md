Build your own assistant/voice assistant with this basic AI chat. This package does not contain AI training but training will be implemented in next version.

# Description
    
It consists of eight main modules:

- `device`: This thing must be implemented in your sneky so without this you cant use it
- `nltkpunkt`: Needed package from nltk called 'punkt' also need in code
- `brain`: This module try to download needed data to work fine. In next version of package there will be training ability so you can use your own data.pth
- `clear`: Clear whole terminal
- `ai_json`: This will get sentences 'database' also in next version you will have ability to set your own sentece 'database'
- `ai_load`: This will load data to work fine
- `recieve_text`: Recieve your text, but you have to define that text from variable
- `speak`: This is your chat, dont forget about bot name

This package works only with python 3.7

# Installation
 
## Normal installation

```bash
pip install build-your-sneky
```

## Usage
```py
import sneky
sneky.nltkpunkt()
sneky.device()
sneky.brain()
sneky.ai_json()
sneky.ai_load()
sneky.clear()
while True:
    text = input("You: ")
    sneky.speak("AI", text)
```