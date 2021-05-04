# -*-coding:Utf-8 -*

####################################################################################
# model_navigator.py
# Definition of "model navigators" using the metaclass MetaModelNavigator
# Jérôme Nika, IRCAM STMS LAB
# copyleft 2016 - 2017
####################################################################################

""" 
Model Navigator
======================
Definition of "model navigators" using the metaclass :class:`~MetaModelNavigator.MetaModelNavigator`.
Main classes: :class:`~ModelNavigator.FactorOracleNavigator`. 
Tutorial for the class :class:`~ModelNavigator.FactorOracleNavigator` in :file:`_Tutorials_/FactorOracleNavigator_tutorial.py`.

Tutorial in :file:`_Tutorials_/FactorOracleNavigator_tutorial.py`

"""
from typing import Optional

from dyci2.navigator import *
from factor_oracle_model import FactorOracle


# TODO : surchager set use_taboo pour que tous les -1 passent à 0 si on passe à FALSE
# TODO : mode 0 : répétitions authorisées, mode 1 = on prend le min, mode 2, interdire les déjà passés
# TODO : SURCHARGER POUR INTERDIRE LES AUTRES

class FactorOracleGenerator:
    """
        **Factor Oracle Navigator class**.
        This class implements heuristics of navigation through a Factor Oracle automaton for creative applications:
        different ways to find paths in the labels of the automaton to collect the associated contents and generate new
        sequences using concatenative synthesis.
        Original navigation heuristics, see **Assayag, Bloch, "Navigating the Oracle: a heuristic approach", in
        Proceedings of the International Computer Music Conference 2007** (https://hal.archives-ouvertes.fr/hal-01161388).

        :see also: **Tutorial in** :file:`_Tutorials_/FactorOracleNavigator_tutorial.py`.
        :see also: This "model navigator" class is created with the metaclass :class:`~MetaModelNavigator.MetaModelNavigator`.

        :Example:

        >>> sequence = ['A1','B1','B2','C1','A2','B3','C2','D1','A3','B4','C3']
        >>> labels = [s[0] for s in sequence]
        >>> FON = FactorOracleGenerator(sequence, labels)
    """

    def __init__(self, sequence=[], labels=[], max_continuity=20,
                 control_parameters=[], history_parameters=[], equiv=(lambda x, y: x == y),
                 label_type=None, content_type=None):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        """
        Constructor for the class FactorOracleNavigator.
        :see also: The class FactorOracle in FactorOracleAutomaton.py

        :Example:

        >>> sequence = ['A1','B1','B2','C1','A2','B3','C2','D1','A3','B4','C3']
        >>> labels = [s[0] for s in sequence]
        >>> FON = FactorOracleGenerator(sequence, labels)

        """

        self.navigator: Navigator = Navigator(sequence, labels, max_continuity, control_parameters,
                                              history_parameters, equiv)
        print(self.navigator.labels)

        self.model: FactorOracle = FactorOracle(sequence, labels, equiv, label_type, content_type)
        print(self.model.labels)
        self.reinit_navigation_param()  # TODO[A] shouldn't be here

    def learn_event(self, state, label, equiv=None):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        self.model.learn_event(state, label, equiv)
        self.navigator.learn_event(state, label, equiv)

    def learn_sequence(self, sequence, labels, equiv=None):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        self.model.learn_sequence(sequence, labels, equiv)
        self.navigator.learn_sequence(sequence, labels, equiv)

    def free_generation(self, length, new_max_continuity=None, forward_context_length_min=0, init=False, equiv=None,
                        print_info=False):
        # TODO[A] Move to Generator
        """ Free navigation through the sequence.
        Naive version of the method handling the free navigation in a sequence (random).
        This method has to be overloaded by a model-dependant version when creating a **model navigator** class
        (cf. :mod:`ModelNavigator`).

        :param length: length of the generated sequence
        :type length: int
        :param new_max_continuity: new value for self.max_continuity (not changed id no parameter is given)
        :type new_max_continuity: int
        :param forward_context_length_min: minimum length of the forward common context
        :type forward_context_length_min: int
        :param init: reinitialise the navigation parameters ?, default : False. (True when starting a new generation)
        :type init: bool
        :param equiv: Compararison function given as a lambda function, default: self.equiv.
        :type equiv: function
        :param print_info: print the details of the navigation steps
        :type print_info: bool
        :return: generated sequence
        :rtype: list
        :see also: Example of overloaded method: :meth:`FactorOracleNavigator.free_navigation`.

        :!: **equiv** has to be consistent with the type of the elements in labels.
        :!: The result **strongly depends** on the tuning of the parameters self.max_continuity,
            self.avoid_repetitions_mode, self.no_empty_event.
        """

        if equiv is None:
            equiv = self.equiv

        if not new_max_continuity is None:
            self.max_continuity = new_max_continuity

        if init:
            self.reinit_navigation_param()

        sequence = []
        generated_indexes = self.free_navigation(length, new_max_continuity, forward_context_length_min, init, equiv,
                                                 print_info)
        for generated_index in generated_indexes:
            if generated_index is None:
                sequence.append(None)
            else:
                sequence.append(self.sequence[generated_index])
        return sequence

    def simply_guided_generation(self, required_labels, new_max_continuity=None, forward_context_length_min=0,
                                 init=False, equiv=None, print_info=False, shift_index=0, break_when_none=False):
        # TODO[A] Move to Generator
        """ Navigation in the sequence, simply guided step by step by an input sequence of label.
        Naive version of the method handling the navigation in a sequence when it is guided by target labels.
        This method has to be overloaded by a model-dependant version when creating a **model navigator** class
        (cf. :mod:`ModelNavigator`).


        :param required_labels: guiding sequence of labels
        :type required_labels: list
        :param new_max_continuity: new value for self.max_continuity (not changed id no parameter is given)
        :type new_max_continuity: int
        :param forward_context_length_min: minimum length of the forward common context
        :type forward_context_length_min: int
        :param init: reinitialise the navigation parameters ?, default : False. (True when starting a new generation)
        :type init: bool
        :param equiv: Compararison function given as a lambda function, default: self.equiv.
        :type equiv: function
        :param print_info: print the details of the navigation steps
        :type print_info: bool
        :return: generated sequence
        :rtype: list
        :see also: Example of overloaded method: :meth:`FactorOracleNavigator.free_navigation`.

        :!: **equiv** has to be consistent with the type of the elements in labels.
        :!: The result **strongly depends** on the tuning of the parameters self.max_continuity,
        self.avoid_repetitions_mode, self.no_empty_event.
        """

        if equiv is None:
            equiv = self.equiv

        if not new_max_continuity is None:
            self.max_continuity = new_max_continuity

        if init:
            self.reinit_navigation_param()

        sequence = []
        generated_indexes = self.simply_guided_navigation(required_labels, new_max_continuity,
                                                          forward_context_length_min, init, equiv, print_info,
                                                          shift_index, break_when_none)
        for generated_index in generated_indexes:
            if generated_index is None:
                sequence.append(None)
            else:
                sequence.append(self.sequence[generated_index])
        return sequence


    def simply_guided_navigation(self, required_labels, new_max_continuity=None,
                                 forward_context_length_min=0, init=False, equiv=None, print_info=False,
                                 shift_index=0,
                                 break_when_none=False):
        # FIXME[MergeState]: A[], B[], C[], D[], E[]
        """ Navigation through the automaton, simply guided step by step by an input sequence of label.
            Returns a novel sequence being consistent with the internal logic of the sequence on which the automaton is
            built, and matching the labels in required_labels.
            (Returns a **path**, i.e., a list of indexes. Generated sequence: cf. :meth:`Navigator.simply_guided_generation`.)

            :param required_labels: guiding sequence of labels
            :type required_labels: list
            :param new_max_continuity: new value for self.max_continuity (not changed id no parameter is given)
            :type new_max_continuity: int
            :param forward_context_length_min: minimum length of the forward common context
            :type forward_context_length_min: int
            :param init: reinitialise the navigation parameters ?, default : False. (True when starting a new generation)
            :type init: bool
            :param equiv: Compararison function given as a lambda function, default: self.equiv.
            :type equiv: function
            :param print_info: print the details of the navigation steps
            :type print_info: bool
            :return: list of indexes of the generated path.
            :rtype: list (int)
            :see also: :meth:`Navigator.simply_guided_generation`
            :see also: Tutorial in FactorOracleNavigator_tutorial.py.

            :!: **equiv** has to be consistent with the type of the elements in labels.
            :!: The result **strongly depends** on the tuning of the parameters self.max_continuity,
            self.avoid_repetitions_mode, self.no_empty_event.


            :Example:

            >>> sequence = ['A1','B1','B2','C1','A2','B3','C2','D1','A3','B4','C3']
            >>> labels = [s[0] for s in sequence]
            >>> FON = FactorOracleGenerator(sequence, labels)
            >>>
            >>> FON.current_position_in_sequence = random.randint(0, FON.index_last_state())
            >>> FON.avoid_repetitions_mode = 1
            >>> FON.max_continuity = 2
            >>> FON.no_empty_event = True
            >>> forward_context_length_min = 0
            >>>
            >>> guide = ['C','A','B','B','C', 'C', 'D']
            >>> generated_sequence = FON.simply_guided_generation(guide,
            >>>                                                   forward_context_length_min = forward_context_length_min,
            >>>                                                   init = True, print_info = True)


            """

        if equiv is None:
            equiv = self.equiv

        if not new_max_continuity is None:
            self.max_continuity = new_max_continuity

        if init:
            self.reinit_navigation_param()
            init_states = [i for i in range(1, self.index_last_state()) if
                           self.direct_transitions.get(i) and equiv(
                               self.direct_transitions.get(i)[0], required_labels[0])]
            self.current_position_in_sequence = init_states[random.randint(0, len(init_states) - 1)]

        generated_sequence_of_indexes = []
        s = None
        for i in range(0, len(required_labels)):
            s = self.simply_guided_navigation_one_step(required_labels[i], new_max_continuity,
                                                       forward_context_length_min, equiv, print_info,
                                                       shift_index=i + shift_index)

            if break_when_none and s is None:
                return generated_sequence_of_indexes
            else:
                generated_sequence_of_indexes.append(s)
            # print("\n\n-->SIMPLY NAVIGATION SETS POSITION: {}<--".format(s))
            # factor_oracle_navigator.current_position_in_sequence = s
        return generated_sequence_of_indexes

    # TODO : ATTENTION, SI UTILISE AILLEURS, BIEN PENSER AU MECANISME EQUIVALENT A INIT POUR ..._navigation_TOUT-COURT
    def simply_guided_navigation_one_step(self, required_label, new_max_continuity=None,
                                          forward_context_length_min=0, equiv=None, print_info=False,
                                          shift_index=0):
        # FIXME[MergeState]: A[], B[], C[], D[], E[]
        str_print_info = "{} (cont. = {}/{}): {}".format(shift_index, self.current_continuity,
                                                         self.max_continuity,
                                                         self.current_position_in_sequence)

        s = None
        init_continuations_matching_label, filtered_continuations_matching_label = self.filtered_continuations_with_label(
            self.current_position_in_sequence, required_label, forward_context_length_min, equiv)

        s = self.follow_continuation_using_transition(filtered_continuations_matching_label)
        if not s is None:
            str_print_info += " -{}-> {}".format(self.labels[s], s)
        else:
            s = self.follow_continuation_with_jump(filtered_continuations_matching_label)
            if not s is None:
                str_print_info += " ...> {} -{}-> {}".format(s - 1,
                                                             self.direct_transitions.get(s - 1)[0],
                                                             self.direct_transitions.get(s - 1)[1])
            else:
                # s = factor_oracle_navigator.find_matching_label_without_continuation(required_label,
                #                             init_continuations_matching_label, equiv)
                s = self.find_matching_label_without_continuation(
                    required_label,
                    self.navigator.filter_using_history_and_taboos(list(range(1, self.model.index_last_state()))),
                    equiv)
                if not s is None:
                    str_print_info += " xxnothingxx - random matching label: {}".format(s)
                else:
                    str_print_info += " xxnothingxx"

        if not s is None:
            # print("\n\n-->SIMPLY NAVIGATION SETS POSITION: {}<--".format(s))
            self.current_position_in_sequence = s
        # factor_oracle_navigator.current_position_in_sequence = s
        # factor_oracle_navigator.history_and_taboos[s] += 1

        if print_info:
            print(str_print_info)

        return s

    ################################################################################################################
    #   LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY   #
    ################################################################################################################

    def filtered_continuations(self, index_state, forward_context_length_min=0, equiv=None):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        """ Continuations from the state at index index_state in the automaton (see method continuations),
        and filtered continuations satisfying the constraints of taboos and repetitions
        (cf. FactorOracleNavigator.history_and_taboos and FactorOracleNavigator.avoid_repetitions_mode).

        :param index_state: start index
        :type index_state: int
        :param required_label: label to read (optional)
        :param forward_context_length_min: minimum length of the forward common context
        :type forward_context_length_min: int
        :param equiv: Compararison function given as a lambda function, default: factor_oracle_navigator.equiv.
        :type equiv: function
        :return: Indexes in the automaton of the possible continuations from the state at index index_state in the
        automaton.
        :rtype: tuple(list (int), list (int))
        :see also: FactorOracleNavigator.continuations(...)

        :!: **equiv** has to be consistent with the type of the elements in labels.

        """

        init_continuations = self.model.get_candidates(index_state=index_state, required_label=None,
                                                       forward_context_length_min=forward_context_length_min,
                                                       equiv=equiv, authorize_direct_transition=True)
        # print("\n\nInitial continuations from index {}: {}".format(index_state, init_continuations))
        # filtered_continuations = [c for c in init_continuations
        #                           if (not (factor_oracle_navigator.history_and_taboos[c] is None)
        #                               and (factor_oracle_navigator.avoid_repetitions_mode < 2
        #                                    or factor_oracle_navigator.avoid_repetitions_mode >= 2
        #                                    and factor_oracle_navigator.history_and_taboos[c] == 0))]
        filtered_continuations = self.navigator.filter_using_history_and_taboos(init_continuations)
        # print("Continuations from index {} after filtering: {}".format(index_state, filtered_continuations))
        return init_continuations, filtered_continuations

    def filtered_continuations_with_label(self, index_state, required_label,
                                          forward_context_length_min=0, equiv=None):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        """ Continuations labeled by required_label from the state at index index_state in the automaton (see method
        continuations), and filtered continuations satisfying the constraints of taboos and repetitions
        (cf. FactorOracleNavigator.history_and_taboos and FactorOracleNavigator.avoid_repetitions_mode).

        :param index_state: start index
        :type index_state: int
        :param required_label: label to read
        :param forward_context_length_min: minimum length of the forward common context
        :type forward_context_length_min: int
        :param equiv: Compararison function given as a lambda function, default: factor_oracle_navigator.equiv.
        :type equiv: function
        :return: Indexes in the automaton of the possible continuations from the state at index index_state in the automaton.
        :rtype: tuple(list (int), list (int))
        :see also: FactorOracleNavigator.continuations_with_label(...)

        :!: **equiv** has to be consistent with the type of the elements in labels.

        """
        init_continuations = self.model.get_candidates(index_state, required_label,
                                                       forward_context_length_min, equiv,
                                                       authorize_direct_transition=True)
        filtered_continuations = self.navigator.filter_using_history_and_taboos(init_continuations)
        return init_continuations, filtered_continuations
