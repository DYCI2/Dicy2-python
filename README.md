# DYCI2 Python Library

The DYCI2 libraries contain collections of generative agents and tools for smart composition and human-machine co-improvisation. 
They integrate the results of the [DYCI2 project](http://repmus.ircam.fr/dyci2/home) and shapes a next-generation software with regard to [OMax](https://github.com/DYCI2/OMax4). 

This repository contains the __Python DYCI2 library__, defining the core models and tools for creative generation of sequences (and in particular musical sequences) from models of sequences. It implements several models, generative heuristics, time management strategies, and architectures of interactive agents. 

Advanced users and developpers can use the Python library autonomously. See the tutorials corresponding to the different modules.
Html doc: http://repmus.ircam.fr/downloads/docs/DYCI2_library/. PDF doc: in DYCI2lib/Python_library.

------
# Applications based on the DYCI2 Python Library

This Python library is the core of two distinct applications dedicated to different music production workflows :

* __[The Max DYCI2 library](https://github.com/DYCI2/Dyci2Lib/releases)__: a library of generative agents and tools for human-machine live interaction. These agents combine machine learning models and generative processes with reactive listening modules. This library offers a collection of “agents/instruments” embedding free, planned and reactive approaches to corpus-based generation, as well as models of short-term dynamic scenarios ("meta-Djing"). Github repository: https://github.com/DYCI2/Dyci2Lib.
* __[The OM/OM\# DYCI2 library](https://github.com/DYCI2/om-dyci2/releases)__: The declination of the library for the OpenMusic environment, OM-DYCI2 specializes in the large-scale offline generation of "meta-composed" material. It imprements libdyci2, a C wrapper to the DYCI2 Python library allowing to compile libdyci2 as a dynamic library, and om-dyci2: a library for using DYCI2/libdyci2 in the OM/o7 computer-aided composition environment. Github repository: https://github.com/DYCI2/om-dyci2.


------

## More about the DYCI2 libraries

__A conference paper about the DYCI2 libraries can be downloaded [here](https://hal.archives-ouvertes.fr/hal-01583089/document).__

__A recent presentation about the DYCI2 libraries (20 min in english): https://youtu.be/RXKJbpJb8w4?t=1530__

__Some videos of collaborations with musicians using DYCI2 or its previous versions: https://www.youtube.com/channel/UCAKZIW0mMWCrX80yS96ZxAw__.

__Author and contributors:__ Jérôme Nika (IRCAM STMS LAB), Joakim Borg (IRCAM STMS LAB), Ken Déguernel (IRCAM STMS LAB / Inria Nancy - Grand Est), Axel Chemla--Romeu-Santos (IRCAM STMS LAB), Georges Bloch (UniStra), Diemo Schwarz (IRCAM STMS LAB), Gérard Assayag (IRCAM STMS LAB); 
__DYCI2 research project :__ Gérard Assayag (Ircam, PI), Emmanuel Vincent (Inria Nancy, WP leader), Jérôme Nika (Ircam, WP coordinator), Marc Chemillier (EHESS, associate researcher).

------

## Requirements
* Mac OS
* Python 3.9 and >

## Python configuration
* Download and install the **last** version of Python **3** (https://www.python.org/downloads).
* Open Terminal to install the dependencies: `cd [DRAG_AND_DROP_THE_DYCI2-Python-Library_DIRECTORY]`, enter, `pip3 install -r requirements.txt --user`, enter.

## Troubleshooting
Please write to `jerome.nika@ircam.fr` for any question, or to share with us your projects using DYCI2 !

## License
GPL v3

