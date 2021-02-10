from setuptools import find_packages, setup

with open('README.md') as readme_file:
    README = readme_file.read()

with open('HISTORY.md') as history_file:
    HISTORY = history_file.read()

with open("requirements.txt", "r") as f:
    requirements = list(map(str.strip, f.read().split("\n")))[:-1]


setup(
    install_requires=requirements,
    name='build-your-sneky',
    packages=find_packages(),
    version='1.2.0',
    description='Build your own assistant/voice assistant with this basic AI chat. Newly added "brain algorithm"',
    long_description_content_type="text/markdown",
    long_description=README + '\n\n' + HISTORY,
    keywords=['Chat', 'Ai', 'Home_Automation'],
    author='RandomGuyBay',
    url='https://github.com/RandomGuyBay/build-your-sneky',
    download_url='https://pypi.org/project/build-your-sneky/',
    license='MIT',
    classifiers=["Programming Language :: Python :: 3.7", "Programming Language :: Python :: 3.8", "Programming Language :: Python :: 3.9"],
)

