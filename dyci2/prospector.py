# -*-coding:Utf-8 -*

####################################################################################
# prospector.py
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
import random
from abc import ABC
from typing import Callable, Tuple, Optional, List, Type

from dyci2.navigator import Navigator
from factor_oracle_model import FactorOracle
# TODO : surchager set use_taboo pour que tous les -1 passent à 0 si on passe à FALSE
# TODO : mode 0 : répétitions authorisées, mode 1 = on prend le min, mode 2, interdire les déjà passés
# TODO : SURCHARGER POUR INTERDIRE LES AUTRES
from label import Label
from memory import MemoryEvent, Memory
from model import Model
from utils import DontKnow


class Prospector:
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

        >>> #sequence = ['A1','B1','B2','C1','A2','B3','C2','D1','A3','B4','C3']
        >>> #labels = [s[0] for s in sequence]
        >>> #FON = FactorOracleGenerator(sequence, labels)
    """

    def __init__(self, model_class: Type[Model], navigator_class: Type[Navigator], memory: Memory, max_continuity=20,
                 control_parameters=(), history_parameters=(), equiv: Callable = (lambda x, y: x == y),
                 label_type=None, content_type=None):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        """
        Constructor for the class FactorOracleNavigator.
        :see also: The class FactorOracle in FactorOracleAutomaton.py

        :Example:

        >>> #sequence = ['A1','B1','B2','C1','A2','B3','C2','D1','A3','B4','C3']
        >>> #labels = [s[0] for s in sequence]
        >>> #FON = FactorOracleGenerator(sequence, labels)

        """
        # TODO: Assert compatbility between model_class and navigator_class

        self.model: Model = model_class(memory, equiv)
        self.navigator: Navigator = navigator_class(memory, equiv, max_continuity, control_parameters,
                                                    history_parameters)
        self.content_type: Type[MemoryEvent] = memory.content_type
        self.label_type: Type[Label] = memory.label_type

        self.navigator.reinit_navigation_param_old_modelnavigator()

    def learn_event(self, event: MemoryEvent, equiv: Optional[Callable] = None):
        """ raises: TypeError if event is incompatible with current memory """
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        if isinstance(event, self.content_type) and isinstance(event.label(), self.label_type):
            self.model.learn_event(event, equiv)
            self.navigator.learn_event(event, equiv)
        else:
            raise TypeError(f"Invalid content/label type for event {str(event)}")

    def learn_sequence(self, sequence: List[MemoryEvent], equiv: Optional[Callable] = None):
        """ raises: TypeError if sequence is incompatible with current memory """
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]

        # TODO[D]: This is never called! In original code it is called from Model.__init__, which obviously isn't
        #  possible. Need to call this from outside when simplifying calls

        # TODO[C]: Ensure that the sequence always is validated at top level so that the list of MemoryEvents always
        #  has (1) a single LabelType and (2) a single ContentType.
        if len(sequence) > 0 and isinstance(sequence[0], self.content_type) and isinstance(sequence[0].label(), self.label_type):
            self.model.learn_sequence(sequence, equiv)
            self.navigator.learn_sequence(sequence, equiv)
        else:
            raise TypeError(f"Invalid content/label type for sequence")

    def l_pre_free_navigation(self, equiv: Optional[Callable], new_max_continuity: int,
                              init: bool) -> Tuple[bool, Callable]:
        if equiv is None:
            equiv = self.model.equiv

        if new_max_continuity is not None:
            self.navigator.max_continuity = new_max_continuity

        if init:
            self.navigator.reinit_navigation_param_old_modelnavigator()

        print("FREE GENERATION")
        print_info: bool = True

        if self.navigator.current_position_in_sequence < 0:
            self.navigator.set_current_position_in_sequence_with_sideeffects(
                random.randint(1, self.model.index_last_state()))
        return print_info, equiv

    def l_pre_guided_navigation(self, required_labels: List[DontKnow], equiv: Optional[Callable],
                                new_max_continuity: Optional[int], init: bool) -> Callable:
        # ###########################################
        # #### From old simply_guided_generation ####
        # ###########################################
        if equiv is None:
            equiv = self.model.equiv

        if not new_max_continuity is None:
            self.navigator.max_continuity = new_max_continuity

        if init:
            self.navigator.reinit_navigation_param_old_modelnavigator()
        # ###########################################
        # #### From old simply_guided_navigation ####
        # ###########################################
        if equiv is None:
            equiv = self.model.equiv

        if not new_max_continuity is None:
            self.navigator.max_continuity = new_max_continuity

        if init:
            self.navigator.reinit_navigation_param_old_modelnavigator()
            init_states = [i for i in range(1, self.model.index_last_state()) if
                           self.model.direct_transitions.get(i) and equiv(
                               self.model.direct_transitions.get(i)[0], required_labels[0])]
            self.navigator.set_current_position_in_sequence_with_sideeffects(
                init_states[random.randint(0, len(init_states) - 1)])

        return equiv

    def r_free_navigation_one_step(self, iteration_index: int, forward_context_length_min: int = 0,
                                   equiv: Optional[Callable] = None,
                                   print_info: bool = False) -> Optional[int]:
        str_print_info = "{} (cont. = {}/{}): {}".format(iteration_index, self.navigator.current_continuity,
                                                         self.navigator.max_continuity,
                                                         self.navigator.current_position_in_sequence)

        init_continuations, filtered_continuations = self.filtered_continuations(
            self.navigator.current_position_in_sequence,
            forward_context_length_min, equiv)
        # print("FREE GENERATION 3.{}.1".format(i))

        s = self.navigator._follow_continuation_using_transition(filtered_continuations, self.model.direct_transitions)
        if not s is None:
            # print("FREE GENERATION 3.{}.2".format(i))
            str_print_info += " -{}-> {}".format(self.model.labels[s], s)
        # factor_oracle_navigator.current_position_in_sequence = s
        else:
            # print("FREE GENERATION 3.{}.3".format(i))
            s = self.navigator._follow_continuation_with_jump(filtered_continuations, self.model.direct_transitions)
            if not s is None:
                # print("FREE GENERATION 3.{}.4".format(i))
                str_print_info += " ...> {} -{}-> {}".format(s - 1,
                                                             self.model.direct_transitions.get(s - 1)[
                                                                 0],
                                                             self.model.direct_transitions.get(s - 1)[
                                                                 1])
            # factor_oracle_navigator.current_position_in_sequence = s
            else:
                # print("FREE GENERATION 3.{}.5".format(i))
                # s = factor_oracle_navigator.navigate_without_continuation(factor_oracle_navigator
                #     .filter_using_history_and_taboos(init_continuations))
                # LAST 15/10
                s = self.navigator._follow_continuation_with_jump(list(range(self.model.index_last_state())),
                                                                  self.model.direct_transitions)
                if not s is None:
                    str_print_info += " xxnothingxx - random: {}".format(s)
                # factor_oracle_navigator.current_position_in_sequence = s
                else:
                    str_print_info += " xxnothingxx"
                # factor_oracle_navigator.current_position_in_sequence = s

        if print_info:
            print(str_print_info)

        # print("FREE GENERATION 3.{}.6".format(i))

        if not s is None:
            self.navigator.set_current_position_in_sequence_with_sideeffects(s)
        # print("\n\n--> FREE NAVIGATION SETS POSITION IN SEQUENCE: {}<--".format(s))
        # factor_oracle_navigator.history_and_taboos[s] += 1
        return s

    # TODO : ATTENTION, SI UTILISE AILLEURS, BIEN PENSER AU MECANISME EQUIVALENT A INIT POUR ..._navigation_TOUT-COURT
    def simply_guided_navigation_one_step(self, required_label, new_max_continuity=None,
                                          forward_context_length_min=0, equiv=None, print_info=False,
                                          shift_index=0):
        # FIXME[MergeState]: A[], B[], C[], D[], E[]
        str_print_info = "{} (cont. = {}/{}): {}".format(shift_index, self.navigator.current_continuity,
                                                         self.navigator.max_continuity,
                                                         self.navigator.current_position_in_sequence)

        s = None
        init_continuations_matching_label, filtered_continuations_matching_label = \
            self.filtered_continuations_with_label(self.navigator.current_position_in_sequence, required_label,
                                                   forward_context_length_min, equiv)

        s = self.navigator._follow_continuation_using_transition(filtered_continuations_matching_label,
                                                                 self.model.direct_transitions)
        if not s is None:
            str_print_info += " -{}-> {}".format(self.model.labels[s], s)
        else:
            s = self.navigator._follow_continuation_with_jump(filtered_continuations_matching_label,
                                                              self.model.direct_transitions)
            if not s is None:
                str_print_info += " ...> {} -{}-> {}".format(s - 1,
                                                             self.model.direct_transitions.get(s - 1)[0],
                                                             self.model.direct_transitions.get(s - 1)[1])
            else:
                # s = factor_oracle_navigator.find_matching_label_without_continuation(required_label,
                #                             init_continuations_matching_label, equiv)
                s = self.navigator.find_matching_label_without_continuation(
                    required_label,
                    self.navigator.filter_using_history_and_taboos(list(range(1, self.model.index_last_state()))),
                    equiv)
                if not s is None:
                    str_print_info += " xxnothingxx - random matching label: {}".format(s)
                else:
                    str_print_info += " xxnothingxx"

        if not s is None:
            # print("\n\n-->SIMPLY NAVIGATION SETS POSITION: {}<--".format(s))
            self.navigator.set_current_position_in_sequence_with_sideeffects(s)
        # factor_oracle_navigator.current_position_in_sequence = s
        # factor_oracle_navigator.history_and_taboos[s] += 1

        if print_info:
            print(str_print_info)

        return s

    def scenario_based_generation(self, use_intervals: bool, labels: List[Label], continuity_with_future: List[float],
                                  authorized_indexes: List[int], authorized_transformations: DontKnow,
                                  sequence_to_interval_fun: Optional[Callable], equiv_interval: Optional[Callable],
                                  equiv: Optional[Callable]) -> Tuple[int, int, int]:
        # FIXME[MergeState]: A[], B[], C[], D[], E[]
        return self.navigator.find_prefix_matching_with_labels(use_intervals, self.model.labels, labels,
                                                               continuity_with_future, authorized_indexes,
                                                               authorized_transformations, sequence_to_interval_fun,
                                                               equiv_interval, equiv)

    def rewind_generation(self, index_in_navigation: int) -> None:
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        self.navigator.rewind_generation(index_in_navigation)

    ################################################################################################################
    #   TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP   #
    ################################################################################################################

    def l_get_no_empty_event(self) -> bool:
        return self.navigator.no_empty_event

    def l_set_no_empty_event(self, v: bool) -> None:
        self.navigator.no_empty_event = v

    def l_get_index_last_state(self) -> int:
        return self.model.index_last_state()

    def l_get_sequence_nonmutable(self) -> List[DontKnow]:
        return self.model.sequence

    def l_get_sequence_maybemutable(self) -> List[DontKnow]:
        return self.model.sequence

    def l_set_sequence(self, sequence: List[DontKnow]):
        self.model.sequence = sequence

    def l_get_labels_nonmutable(self) -> List[DontKnow]:
        return self.model.labels

    def l_get_labels_maybemutable(self) -> List[DontKnow]:
        return self.model.labels

    def l_set_labels(self, labels: List[DontKnow]):
        self.model.labels = labels

    def l_get_position_in_sequence(self) -> int:
        return self.navigator.current_position_in_sequence

    def l_set_position_in_sequence(self, index: int):
        self.navigator.set_current_position_in_sequence_with_sideeffects(index)

    ################################################################################################################
    #   LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY   #
    ################################################################################################################

    def filtered_continuations(self, index_state, forward_context_length_min=0, equiv: Optional[Callable] = None):
        # FIXME[MergeState]: A[x], B[], C[], D[], E[]
        """ Continuations from the state at index index_state in the automaton (see method continuations),
        and filtered continuations satisfying the constraints of taboos and repetitions
        (cf. FactorOracleNavigator.history_and_taboos and FactorOracleNavigator.avoid_repetitions_mode).

        :param index_state: start index
        :type index_state: int
        #:param required_label: label to read (optional)
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


# TODO[C] Get rid of.
implemented_model_navigator_classes = {"FactorOracleNavigator": Prospector}
