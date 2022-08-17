from abc import ABC, abstractmethod

from dyci2.label import Dyci2Label


class Equiv(ABC):

    @staticmethod
    @abstractmethod
    def eq(a: Dyci2Label, b: Dyci2Label) -> bool:
        """ TODO: Docstring """


class BasicEquiv(Equiv):
    @staticmethod
    def eq(a: Dyci2Label, b: Dyci2Label) -> bool:
        return a == b
