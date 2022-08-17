# -*-coding:Utf-8 -*

####################################################################################
# navigator.py
# Parameters and methods to navigate through a symbolic sequence.
# Jérôme Nika, Ken Déguernel - IRCAM STMS Lab
# copyleft 2016 - 2018
####################################################################################

# TODO
# ICI ET DANS LES MODEL NAVIGATORS : TESTER SI LE REQUIRED LABEL EST BIEN DU MEME TYPE QUE LES LABELS DANS LE MODEL

"""
Navigator
===================

This module defines parameters and methods to navigate through a symbolic sequence.
The classes defined in this module are used in association with models (cf. :mod:`Model`) when creating
**model navigator** classes (cf. :mod:`ModelNavigator`).

"""
from abc import ABC, abstractmethod
from typing import List, Optional, Generic, TypeVar

from dyci2.equiv import Equiv
from dyci2.label import Dyci2Label
from dyci2.parameter import Parametric
from merge.main.candidate import Candidate

T = TypeVar('T')


class Navigator(Parametric, Generic[T], ABC):
    """ TODO: Docstring """

    @abstractmethod
    def learn_sequence(self,
                       sequence: List[Optional[T]],
                       labels: List[Optional[Dyci2Label]],
                       equiv: Optional[Equiv] = None) -> None:
        """ TODO: Docstring (can be copied from Model / FactorOracle) """

    @abstractmethod
    def learn_event(self,
                    event: Optional[T],
                    label: Optional[Dyci2Label],
                    equiv: Optional[Equiv] = None) -> None:
        """ TODO: Docstring (can be copied from Model / FactorOracle) """

    @abstractmethod
    def set_time(self, time: int) -> None:
        """ TODO: Docstring (difference from rewind is that it may go forward or backward in time) """

    @abstractmethod
    def rewind_generation(self, time_index: int) -> None:
        """ TODO: Docstring """

    @abstractmethod
    def feedback(self, output_event: Optional[Candidate]) -> None:
        """ TODO: Docstring """

    @abstractmethod
    def reset_position_in_sequence(self, randomize: bool = False):
        """ TODO: Docstring """

    @abstractmethod
    def clear(self) -> None:
        """ TODO: Docstring """
