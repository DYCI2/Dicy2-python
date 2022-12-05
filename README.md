# Dicy2 Python Library

Dicy2 is a collection of agents and tools for the interactive generation of structured musical sequences. It can be integrated into musical setups ranging from the production of structured material within a compositional process to the design of autonomous agents for improvised interaction. Dicy2 is available as a [library for Max](https://forum.ircam.fr/projects/detail/dicy2/) and a [plugin for Ableton Live](https://forum.ircam.fr/projects/detail/dicy2-for-live/). 

In this repository we provide the generative core of these applications as a Python library. The Dicy2 Python library defines the core models and tools for the interactive generation of symbolic musical sequences and implements several models, generative heuristics, time management strategies, and architectures of interactive agents.

------

## References

Dicy2 is a library of the Ircam Musical Representations team, designed and developed by [Jérôme Nika](https://jeromenika.com/) and Joakim Borg (Python generative engine / Gig RepMus API) in the framework of the ANR-DYCI2, ANR-MERCI, ERC-REACH projects directed by Gérard Assayag, and the Ircam "UPI CompAI" project. 

If using the library, please quote: __Nika, J., Déguernel, K., Chemla, A., Vincent, E., & Assayag, G. (2017, October). Dyci2 agents: merging the" free"," reactive", and" scenario-based" music generation paradigms. In International Computer Music Conference. [(Article)](https://hal.archives-ouvertes.fr/hal-01583089/document).__


* __[Video presentation about Dicy2 in French](https://youtu.be/Co_9xZTFPEs)__

* __[Video presentation about Dicy2 in English](https://youtu.be/RXKJbpJb8w4?t=1530)__

* __Some videos of collaborations with musicians using Dicy2 or its previous versions: https://www.youtube.com/channel/UCAKZIW0mMWCrX80yS96ZxAw__.


------

## Getting Started

### Requirements
* MacOS (the Python library is likely to work on any OS but has only been tested on MacOS)
* Python >= 3.9

### Installation
1. Clone the repository and initialize submodules: `git clone --recurse-submodules git@github.com:DYCI2/Dicy2-python.git` 
2. Install dependencies `pip3 install -r requirements.txt`

### Running the Dicy2 Python Library
In the root of this repository, there are two examples illustrating two different usages of the Dicy2 Python Library:
* `generator_tutorial.py`: A minimal example on how to quickly generate content using the `Generator` class. Edit the file to run the different queries or to write your own queries
* `dicy2_server.py`: The server that is used by the [Dicy2 Max Library](https://github.com/DYCI2/Dicy2/releases). This illustrates communication over OSC, see the header of the file for more information. Run with `./dicy2_server.py`


### Going Further
The examples outlined under [Running the Dicy2 Python Library](#Running-the-Dicy2-Python-Library) are excellent starting points, but there are a couple of other possible classes to start from if you want to integrate the DYCI2 Python Library in your own project:
* The `Generator` class (`dicy2/generator.py`) is a good starting point for generating content in an offline setting without managing time
* The `GenerationScheduler` class (`dicy2/generation_scheduler.py`) is a good starting point for generating content with time management in real-time 
* The `Agent` (`dicy2/agent.py`) and `DICY2Server` (`dicy2.py`) are good starting points for managing one or several `GenerationSchedulers` over the OSC protocol

For a specification on the OSC protocol used in the Dicy2 Python Library, see `docs/osc_protocol.md`. Its usage is also thoroughly documentented in the header of the `dyci2_server.py` file.

## Building the Dicy2 Server Application
While this use case is unlikely to be relevant for most users, it is possible to build a standalone application of the `dicy2_server.py` from the Python code using [PyInstaller](https://pyinstaller.org/en/stable/). A pre-built application is already distributed in the release of the [Dicy2 Max Library](https://github.com/DYCI2/Dicy2/releases). 

See the head of the `Makefile` for information on building 


## Troubleshooting
Please write to `jerome.nika@ircam.fr` for any question, or to share with us your projects using Dicy2!

## License
GPL v3

