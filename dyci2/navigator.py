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
from typing import List, Optional, Callable, Generic, TypeVar

from dyci2.dyci2_label import Dyci2Label
from dyci2.parameter import Parametric
from merge.main.candidate import Candidate

T = TypeVar('T')


class Navigator(Parametric, Generic[T], ABC):
    """ TODO: Docstring """

    @abstractmethod
    def learn_sequence(self,
                       sequence: List[Optional[T]],
                       labels: List[Optional[Dyci2Label]],
                       equiv: Optional[Callable] = None) -> None:
        """ TODO: Docstring (can be copied from Model / FactorOracle) """

    @abstractmethod
    def learn_event(self,
                    event: Optional[T],
                    label: Optional[Dyci2Label],
                    equiv: Optional[Callable] = None) -> None:
        """ TODO: Docstring (can be copied from Model / FactorOracle) """

    @abstractmethod
    def feedback(self, output_event: Optional[Candidate]) -> None:
        """ TODO: Docstring """

    @abstractmethod
    def rewind_generation(self, time_index: int) -> None:
        """ TODO: Docstring """

    @abstractmethod
    def clear(self) -> None:
        """ TODO: Docstring """
