from setuptools import find_packages, setup
import os

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

os.system("pip install -r requirements.txt")
os.system("pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html")

setup(
    name='build-your-sneky',
    packages=find_packages(),
    version='1.0.0',
    description='Build your own assistant/voice assistant with this basic AI chat. This package does not contain AI training but training will be implemented in next version.',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    keywords=['Chat', 'Ai', 'Home_Automation'],
    author='RandomGuyBay',
    url='https://github.com/RandomGuyBay/build-your-sneky',
    download_url='https://pypi.org/project/build-your-sneky/',
    license='MIT',
)
