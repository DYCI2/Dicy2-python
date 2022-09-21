from abc import ABC, abstractmethod

from dyci2.label import Dyci2Label

""" 
EQUIV
===================

Method of overriding the default equivalence function as defined in :class:`~ListLabel`. 
This can be used for creating custom rules of equivalence to use in the :class:`~PrefixIndexing` algorithm

"""

class Equiv(ABC):

    @staticmethod
    @abstractmethod
    def eq(a: Dyci2Label, b: Dyci2Label) -> bool:
        """ TODO: Docstring """


class BasicEquiv(Equiv):
    @staticmethod
    def eq(a: Dyci2Label, b: Dyci2Label) -> bool:
        return a == b
