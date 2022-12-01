# DYCI2 Python Library

The DYCI2 libraries contain collections of generative agents and tools for smart composition and human-machine co-improvisation. 
They integrate the results of the [DYCI2 project](http://repmus.ircam.fr/dyci2/home) and shapes a next-generation software with regard to [OMax](https://github.com/DYCI2/OMax4). 

This repository contains the __Python DYCI2 library__, defining the core models and tools for creative generation of sequences (and in particular musical sequences) from models of sequences. It implements several models, generative heuristics, time management strategies, and architectures of interactive agents. 

Advanced users and developers can use the Python library autonomously.

---

## Applications Based on the DYCI2 Python Library

This Python library is the core of two distinct applications dedicated to different music production workflows:

* __[The Max DYCI2 library](https://github.com/DYCI2/Dyci2Lib/releases)__: a library of generative agents and tools for human-machine live interaction in the MaxMSP environment. These agents combine machine learning models and generative processes with reactive listening modules. This library offers a collection of “agents/instruments” embedding free, planned and reactive approaches to corpus-based generation, as well as models of short-term dynamic scenarios ("meta-Djing"). Github repository: https://github.com/DYCI2/Dyci2Lib.
* __[The OM/OM\# DYCI2 library](https://github.com/DYCI2/om-dyci2/releases)__: The declination of the library for the OpenMusic environment, OM-DYCI2 specializes in the large-scale offline generation of "meta-composed" material. It imprements libdyci2, a C wrapper to the DYCI2 Python library allowing to compile libdyci2 as a dynamic library, and om-dyci2: a library for using DYCI2/libdyci2 in the OM/o7 computer-aided composition environment. Github repository: https://github.com/DYCI2/om-dyci2.


------

## More About the DYCI2 Libraries

__A conference paper about the DYCI2 libraries can be downloaded [here](https://hal.archives-ouvertes.fr/hal-01583089/document).__

__A recent presentation about the DYCI2 libraries (20 min in english): https://youtu.be/RXKJbpJb8w4?t=1530__

__Some videos of collaborations with musicians using DYCI2 or its previous versions: https://www.youtube.com/channel/UCAKZIW0mMWCrX80yS96ZxAw__.

__Author and contributors:__ Jérôme Nika (IRCAM STMS LAB), Joakim Borg (IRCAM STMS LAB), Ken Déguernel (IRCAM STMS LAB / Inria Nancy - Grand Est), Axel Chemla--Romeu-Santos (IRCAM STMS LAB), Georges Bloch (UniStra), Diemo Schwarz (IRCAM STMS LAB), Gérard Assayag (IRCAM STMS LAB); 
__DYCI2 research project :__ Gérard Assayag (Ircam, PI), Emmanuel Vincent (Inria Nancy, WP leader), Jérôme Nika (Ircam, WP coordinator), Marc Chemillier (EHESS, associate researcher).

------

## Getting Started

### Requirements
* MacOS (the Python library is likely to work on any OS but has only been tested on MacOS)
* Python >= 3.9

### Installation
1. Clone the repository and initialize submodules: `git clone --recurse-submodules git@github.com:DYCI2/Dyci2Lib.git` 
2. Install dependencies `pip3 install -r requirements.txt`

### Running the DYCI2 Python Library
In the root of this repository, there are two examples illustrating two different usages of the DYCI2 Python Library:
* `dyci2_server.py`: The server that is used by the [DYCI2 Max Library](https://github.com/DYCI2/Dyci2Lib/releases). This illustrates communication over OSC, see the header of the file for more information. Run with `./dyci2_server.py`
* `generator_tutorial.py`: A minimal example on how to quickly generate content using the `Generator` class. Edit the file to run the different queries or to write your own queries

### Going Further
The examples outlined under [Running the DYCI2 Python Library](#Running-the-DYCI2-Python-Library) are excellent starting points, but there are a couple of other possible classes to start from if you want to integrate the DYCI2 Python Library in your own project:
* The `Generator` class (`dyci2/generator.py`) is a good starting point for generating content in an offline setting without managing time
* The `GenerationScheduler` class (`dyci2/generation_scheduler.py`) is a good starting point for generating content with time management in real-time 
* The `Agent` (`dyci2/agent.py`) and `DYCI2Server` (`dyci2_server.py`) are good starting points for managing one or several `GenerationSchedulers` over the OSC protocol

For a specification on the OSC protocol used in the DYCI2 Python Library, see `docs/osc_protocol.md`. Its usage is also thoroughly documentented in the header of the `dyci2_server.py` file.

## Building the DYCI2 Server Application
While this use case is unlikely to be relevant for most users, it is possible to build a standalone application of the `dyci2_server.py` from the Python code using [PyInstaller](https://pyinstaller.org/en/stable/). A pre-built application is already distributed in the release of the [DYCI2 Max Library](https://github.com/DYCI2/Dyci2Lib/releases). 

See the head of the `Makefile` for information on building 


## Troubleshooting
Please write to `jerome.nika@ircam.fr` for any question, or to share with us your projects using DYCI2!

## License
GPL v3

