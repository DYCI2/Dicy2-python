# -*-coding:Utf-8 -*

#############################################################################
# model.py
# Models of symbolic sequences.
# Jérôme Nika, IRCAM STMS LAB / Ken Deguernel, INRIA Nancy - IRCAM STMS Lab
# copyleft 2016 - 2017
#############################################################################

# TODO : RAJOUTER DANS DOC LES ARGUMENTS LIES A "LABEL_TYPE" ET "CONTENT_TYPE"

"""
Model
===================

This module defines different models of symbolic sequences.
The classes defined in this module are minimal and only implement the construction algorithms and basic methods.
Navigation and creative aspects are handled by other classes in the library (cf. :mod:`Navigator` and
:mod:`ModelNavigator`).
Main classes: :class:`~Model.Model`, :class:`~Model.FactorOracle`.
Tutorial for the class :class:`~Model.FactorOracle` in :file:`_Tutorials_/FactorOracleAutomaton_tutorial.py`.

"""

import random
from typing import List, Union, Callable

from dyci2.label import *
from collections import Counter


class Model:
    # FIXME[MergeState]: A[x], B[], C[], D[], E[]
    """The class :class:`~Model.Model` is an **abstract class**.
    Any new model of sequence must inherit from this class.

    :param sequence: sequence learnt in the model.
    :type sequence: list or str
    :param labels: sequence of labels chosen to describe the sequence.
    :type labels: list or str
    :param equiv: compararison function given as a lambda function, default if no parameter is given: self.equiv.
    :type equiv: function

    :!: **equiv** has to be consistent with the type of the elements in labels.
    """

    def __init__(self, sequence=[], labels=[], equiv=(lambda x, y: x == y), label_type=None, content_type=None):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        self.sequence: List[Union[int, str]] = []
        self.content_type = content_type
        self.labels = []    # TODO[A]: is type signature really str/int or should it be List[Label]?
        self.label_type = label_type
        self.equiv: Callable = equiv

        self.build(sequence, labels)

    def index_last_state(self):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        """ Index of the last state in the model."""
        return len(self.labels) - 1

    def init_model(self):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        """ Initialization method called before learning the sequence in the model."""
        pass

    def learn_sequence(self, sequence, labels, equiv=None):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        """
        Learns (appends) a new sequence in the model.

        :param sequence: sequence learnt in the Factor Oracle automaton
        :type sequence: list or str
        :param labels: sequence of labels chosen to describe the sequence
        :type labels: list or str
        :param equiv: Compararison function given as a lambda function, default if no parameter is given: self.equiv.
        :type equiv: function

        :!: **equiv** has to be consistent with the type of the elements in labels.

        """
        if equiv is None:
            equiv = self.equiv
        try:
            assert len(labels) == len(sequence)
        except AssertionError as exception:
            print("Sequence and sequence of labels have different lengths.", exception)
            return None
        else:
            # TODO POUR CONTENTS QUAND LA CLASSE EXISTERA
            labels_to_learn = from_list_to_labels(labels, self.label_type)
            sequence_to_learn = from_list_to_contents(sequence, self.content_type)
            # print("LABELS TO LEARN = {}".format(labels_to_learn))
            print(self.content_type)
            # print("CONTENTS TO LEARN = {}".format(sequence_to_learn))
            for i in range(len(labels_to_learn)):
                self.learn_event(sequence_to_learn[i], labels_to_learn[i], equiv)

    def learn_event(self, state, label, equiv=None):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        """
        Learns (appends) a new state in the model.

        :param state:
        :param label:
        :param equiv: Compararison function given as a lambda function, default if no parameter is given: self.equiv.
        :type equiv: function

        :!: **equiv** has to be consistent with the type of label.

        """
        if equiv is None:
            equiv = self.equiv

        # print("\n\nlabel init to learn= {}".format(label))
        # print("content init to learn= {}".format(state))
        # print("sequence before {}".format(self.sequence))
        self.sequence.append(state)
        # print("sequence after {}".format(self.sequence))
        # self.labels.append(label)
        # print("label to append = {}".format(from_list_to_labels([label], self.label_type)[0]))
        # print("labels before {}".format(self.labels))
        self.labels.append(from_list_to_labels([label], self.label_type)[0])

    # print("labels after {}".format(self.labels))

    def build(self, sequence, labels):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        """
        Builds the model.

        :param sequence: sequence learnt in the model.
        :type sequence: list or str
        :param labels: sequence of labels chosen to describe the sequence
        :type labels: list or str
        :param equiv: Compararison function given as a lambda function, default if no parameter is given: self.equiv.
        :type equiv: function

        :!: **equiv** has to be consistent with the type of the elements in labels.

        """
        if not self.label_type is None:
            try:
                assert issubclass(self.label_type, Label)
            except AssertionError as exception:
                print("label_type must inherit from the class Label.", exception)
                return None
            else:
                self.equiv = self.label_type.__eq__
        self.init_model()
        self.learn_sequence(sequence, labels, self.equiv)

    def print_model(self):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        for i in range(self.index_last_state()):
            print("({}) {}:{}".format(i, self.labels[i], self.sequence[i]))

