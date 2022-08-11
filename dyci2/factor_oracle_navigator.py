import copy
import logging
import random
import warnings
from typing import List, Optional, Callable, Dict, Tuple, TypeVar

from dyci2 import intervals
from dyci2.dyci2_label import Dyci2Label
from dyci2.navigator import Navigator
from dyci2.parameter import Parameter, OrdinalRange
from dyci2.prefix_indexing import PrefixIndexing
from dyci2.utils import noneIsInfinite
from merge.main.candidate import Candidate

T = TypeVar('T')


class FactorOracleNavigator(Navigator[T]):
    """
    TODO: Update docstring
    The class :class:`~Navigator.Navigator` implements **parameters and methods that are used to navigate through a
    model of sequence**.
    These parameters and methods are **model-independent**.
    This class defines in particular the naive versions of the methods :meth:`Navigator.simply_guided_navigation`
    and :meth:`Navigator.free_navigation` handling the navigation through a sequence when it is respectively guided by
    target labels and free.
    These methods are overloaded by model-dependant versions (and other model-dependent parameters or methods can be
    added) when creating a **model navigator** class (cf. :mod:`ModelNavigator`).
    This class is not supposed to be used alone, only in association with a model within a model navigator. Therefore
    its attributes are only "flags" that can be used when defining a model navigator.

    # # :param sequence: sequence learnt in the model.
    # :type sequence: list or str
    # :param labels: sequence of labels chosen to describe the sequence.
    # :type labels: list or str
    # :param equiv: comparison function given as a lambda function, default if no parameter is given: self.equiv.
    # :type equiv: function

    # :param current_navigation_index: current length of the navigation
    # :type current_navigation_index: int

    # :param current_position_in_sequence: current position of the readhead in the model. ** When this attribute receives
    # a new value, :meth:`Navigator.record_execution_trace` is called to update :attr:`self.execution_trace`, and
    # :meth:`Navigator.update_history_and_taboos` is called to update :attr:`self.history_and_taboos`.**
    # :type current_position_in_sequence: int
    # :param current_continuity: current number of consecutive elements retrieved in the sequence at the current step of
    # generation
    # :type current_continuity: int
    # :param max_continuity: limitation of the length of the sub-sequences that can be retrieved from the sequence.
    # :type max_continuity: int
    # :param no_empty_event: authorize or not to output empty events.
    # :type no_empty_event: bool
    # :param avoid_repetitions_mode: 0: authorize repetitions; 1: favor less previously retrieved events;
    # 2: forbid repetitions.
    # :type avoid_repetitions_mode: int
    # :param control_parameters: list of the slots of the class that are considered as "control parameters" i.e. that can
    # be used by a user to author / monitor the generation processes.
    # :type control_parameters: list(str)
    # :param execution_trace_parameters: list of the slots of the class that are stored in the execution trace used in
    # :meth:`Generator.go_to_anterior_state_using_execution_trace`.
    # :type control_parameters: list(str)
    # #:param execution_trace: History of the previous runs of the generation model. The list of the parameters of the
    # model whose values are stored in the execution trace is defined in :attr:`self.execution_trace_parameters`.
    # :type execution_trace: dict
    """

    def __init__(self,
                 equiv: Optional[Callable] = (lambda x, y: x == y),
                 max_continuity: int = 20,
                 control_parameters=(),
                 execution_trace_parameters=(),
                 continuity_with_future: Tuple[float, float] = (0.0, 1.0)):
        self.logger = logging.getLogger(__name__)
        self.sequence: List[Optional[T]] = [None]  # initial state
        self.labels: List[Optional[Dyci2Label]] = [None]  # initial state
        self.equiv: Callable = equiv
        self.max_continuity: Parameter[int] = Parameter(max_continuity, OrdinalRange(0, None))
        self.avoid_repetitions_mode: Parameter[int] = Parameter(0)
        self.continuity_with_future: Parameter[Tuple[float, float]] = Parameter(continuity_with_future)
        self.execution_trace = {}

        self.history_and_taboos: List[Optional[int]] = [0] * (len(self.sequence))
        self.current_continuity: int = -1
        self.current_position_in_sequence: int = -1
        self.current_navigation_index: int = -1
        self.clear()

        self.control_parameters = ["avoid_repetitions_mode", "max_continuity"]
        if type(control_parameters) != type(None):
            self.logger.debug("argument control_parameters = {}".format(control_parameters))
            for param in control_parameters:
                # TODO : PLUTOT FAIRE AVEC UN TRY ASSERT... POUR ETRE PLUS PROPRE
                if param in self.__dict__.keys():
                    self.control_parameters.append(param)
        else:
            self.logger.debug("argument control_parameters = None")

        self.execution_trace_parameters = ["current_position_in_sequence", "history_and_taboos", "current_continuity"]
        for param in execution_trace_parameters:
            # TODO : TRY ASSERT... POUR ETRE PLUS PROPRE
            if param in self.__dict__.keys():
                self.execution_trace_parameters.append(param)

    def __setattr__(self, name_attr, val_attr):
        object.__setattr__(self, name_attr, val_attr)
        # TODO : SUPPRIMER TRACE AVANT TEMPS PERFORMANCE

    ################################################################################################################
    # PUBLIC: INHERITED METHODS
    ################################################################################################################

    def learn_sequence(self,
                       sequence: List[Optional[T]],
                       labels: List[Optional[Dyci2Label]],
                       equiv: Optional[Callable] = None) -> None:
        """
        TODO: Update docstring
        Learns (appends) a new sequence in the model.

        # :param sequence: sequence learnt in the Factor Oracle automaton
        # :type sequence: list or str
        # :param labels: sequence of labels chosen to describe the sequence
        # :type labels: list or str
        # :param equiv: Compararison function given as a lambda function, default if no parameter is given: self.equiv.
        # :type equiv: function

        :!: **equiv** has to be consistent with the type of the elements in labels.

        """
        if equiv is None:
            equiv = self.equiv

        for event, label in zip(sequence, labels):
            self.learn_event(event, label, equiv)

    def learn_event(self,
                    event: Optional[T],
                    label: Optional[Dyci2Label],
                    equiv: Optional[Callable] = None) -> None:
        self.sequence.append(event)
        self.labels.append(label)
        current_last_idx = len(self.history_and_taboos) - 1
        self._authorize_indexes([current_last_idx])
        self.history_and_taboos.append(None)

    def rewind_generation(self, time_index: int) -> None:
        self._go_to_anterior_state_using_execution_trace(index_in_navigation=time_index)

    def clear(self):
        """ (Re)initializes the navigation parameters (current navigation index, history of retrieved indexes,
        current continuity,...). """
        self.history_and_taboos = [None] + [0] * (len(self.sequence) - 1)
        self.current_continuity = 0
        self.current_position_in_sequence = -1
        self.current_navigation_index = - 1

    def feedback(self, output_event: Optional[Candidate]) -> None:
        if output_event is not None:
            self.set_position_in_sequence(output_event.event.index + 1)  # To account for Model's initial None

    ################################################################################################################
    # PUBLIC: CLASS-SPECIFIC RUNTIME CONTROL
    ################################################################################################################

    def set_position_in_sequence(self, val_attr: Optional[int]) -> None:
        self.current_position_in_sequence = val_attr
        if val_attr is not None and val_attr > -1:
            self.current_navigation_index += 1
            self.logger.debug("\nNEW POSITION IN SEQUENCE: {}".format(val_attr))
            self.logger.debug("NEW NAVIGATION INDEX: {}".format(self.current_navigation_index))
            self.logger.debug("OLD LEN EXECUTION TRACE: {}".format(len(self.execution_trace)))

            if self.current_navigation_index > 0 and val_attr == \
                    self.execution_trace[self.current_navigation_index - 1]["current_position_in_sequence"] + 1:
                self.current_continuity += 1
                self.logger.debug("Continuity + 1 = {}".format(self.current_continuity))
            else:
                self.current_continuity = 0
                self.logger.debug("Continuity set to 0")

            self._update_history_and_taboos(val_attr)
            self._record_execution_trace(self.current_navigation_index)
            self.logger.debug("NEW LEN EXECUTION TRACE: {}".format(len(self.execution_trace)))
        # print("NEW EXECUTION TRACE: {}".format(self.execution_trace))

    ################################################################################################################
    #   PUBLIC: NAVIGATION STRATEGIES
    ################################################################################################################

    def find_prefix_matching_with_labels(self,
                                         use_intervals: bool,
                                         memory_labels: List[Optional[Dyci2Label]],
                                         labels_to_match: List[Dyci2Label],
                                         authorized_indices: List[int],
                                         authorized_transformations: List[int],
                                         sequence_to_interval_fun: Optional[Callable],
                                         equiv_interval: Optional[Callable]) -> Dict[int, List[List[int]]]:

        index_delta_prefixes: Dict[int, List[List[int]]]
        if use_intervals:
            index_delta_prefixes, _ = intervals.filtered_prefix_indexing_intervals(
                sequence=memory_labels,
                pattern=labels_to_match,
                length_interval=self.continuity_with_future,
                authorized_indexes=authorized_indices,
                authorized_intervals=authorized_transformations,
                sequence_to_interval_fun=sequence_to_interval_fun,
                equiv=equiv_interval,
                print_info=False
            )

        else:
            index_delta_prefixes, _ = PrefixIndexing.filtered_prefix_indexing(
                sequence=labels_to_match,
                pattern=memory_labels,
                length_interval=self.continuity_with_future,
                authorized_indexes=authorized_indices,
                equiv=self.equiv,
                print_info=False
            )

        return index_delta_prefixes

    def follow_continuation_using_transition(self,
                                             authorized_indices: List[int],
                                             direct_transitions: Dict[int, Tuple[Dyci2Label, int]]) -> List[int]:
        """
        Continuation using direct transition from self.current_position_in_sequence.

        In the method free_generation, this method is called with authorized_indexes = possible continuations
        filtered to satisfy the constraints of taboos and repetitions.
        In the method simply_guided_generation, this method is called with authorized_indexes = possible continuations
        **matching the required label** filtered to satisfy the constraints of taboos and repetitions.

        # :param authorized_indexes: list of authorized indexes to filter taboos, repetitions, and label when needed.
        # :type authorized_indexes: list(int)
        :return: index of the state
        :rtype: int

        """

        direct_transition: Optional[Tuple[Dyci2Label, int]] = direct_transitions.get(self.current_position_in_sequence)

        if direct_transition is not None and self.current_continuity < self.max_continuity.get():
            # TODO: Assign a value to a match instead of returning it directly
            return [i for i in authorized_indices if i == direct_transition[1]]
        return []

    def continuations_with_jump(self,
                                authorized_indices: List[int],
                                direct_transitions: Dict[int, Tuple[Dyci2Label, int]]) -> List[int]:

        """
        List of continuations with jumps to indexes with similar contexts direct transition from
        self.current_position_in_sequence.

        In the method free_generation, this method is called with authorized_indexes = possible continuations filtered
        to satisfy the constraints of taboos and repetitions.
        In the method simply_guided_generation, this method is called with authorized_indexes = possible continuations
        **matching the required label** filtered to satisfy the constraints of taboos and repetitions.

        # :param authorized_indexes: list of authorized indexes to filter taboos, repetitions, and label when needed.
        # :type authorized_indexes: list(int)
        :return: indexes of the states
        :rtype: list(int)

        """
        direct_transition: Optional[Tuple[Dyci2Label, int]] = direct_transitions.get(self.current_position_in_sequence)

        if direct_transition:
            authorized_indices = [i for i in authorized_indices if i != direct_transition[1]]

        if len(authorized_indices) > 0:
            if self.avoid_repetitions_mode.get() > 0:
                self.logger.debug(
                    f"\nTrying to avoid repetitions: possible continuations {authorized_indices}...")
                # TODO: This nested list comprehension could be optimized
                minimum_history_taboo_value: int = min([self.history_and_taboos[i] for i in authorized_indices],
                                                       key=noneIsInfinite)
                authorized_indices = [i for i in authorized_indices
                                      if self.history_and_taboos[i] == minimum_history_taboo_value]
                self.logger.debug(f"... reduced to {authorized_indices}.")

        return authorized_indices

    def follow_continuation_with_jump(self,
                                      authorized_indices: List[int],
                                      direct_transitions: Dict[int, Tuple[Dyci2Label, int]]) -> List[int]:

        """
        Random selection of a continuation with jump to indexes with similar contexts direct transition from
        self.current_position_in_sequence.

        In the method free_generation, this method is called with authorized_indexes = possible continuations filtered
        to satisfy the constraints of taboos and repetitions.
        In the method simply_guided_generation, this method is called with authorized_indexes = possible continuations
        **matching the required label** filtered to satisfy the constraints of taboos and repetitions.

        # :param authorized_indexes: list of authorized indexes to filter taboos, repetitions, and label when needed.
        # :type authorized_indexes: list(int)
        :return: index of the state
        :rtype: int

        """
        authorized_indices = self.continuations_with_jump(authorized_indices, direct_transitions)
        if len(authorized_indices) > 0:
            # TODO: Migrate this random choice to top level
            random_choice: int = random.randint(0, len(authorized_indices) - 1)
            return [authorized_indices[random_choice]]
        return []

    def find_matching_label_without_continuation(self,
                                                 required_label: Dyci2Label,
                                                 authorized_indices: List[int],
                                                 equiv: Optional[Callable] = None) -> List[int]:

        """
        Random state in the sequence matching required_label if self.no_empty_event is True (else None).

        # :param required_label: label to read
        # :param authorized_indexes: list of authorized indexes to filter taboos, repetitions, and label when needed.
        # :type authorized_indexes: list(int)
        # :param equiv: Compararison function given as a lambda function, default: self.equiv.
        # :type equiv: function
        # :return: index of the state
        # :rtype: int

        :!: **equiv** has to be consistent with the type of the elements in labels.

        """
        if equiv is None:
            equiv = self.equiv

        authorized_indices = [i for i in authorized_indices if equiv(self.labels[i], required_label)]
        if len(authorized_indices) > 0:
            random_choice: int = random.randint(0, len(authorized_indices) - 1)
            return [authorized_indices[random_choice]]

        return []

    def filter_using_history_and_taboos(self, indices: List[int]) -> List[int]:
        filtered_list = [i for i in indices
                         if (not (self.history_and_taboos[i] is None)
                             and (self.avoid_repetitions_mode.get() < 2
                                  or self.avoid_repetitions_mode.get() >= 2
                                  and self.history_and_taboos[i] == 0))]
        # print("Possible next indexes = {}, filtered list = {}".format(list_of_indexes, filtered_list))
        return filtered_list

    ################################################################################################################
    #   PRIVATE
    ################################################################################################################

    def _go_to_anterior_state_using_execution_trace(self, index_in_navigation):

        """
        This method is called when the run of a new query rewrites previously generated anticipations.
        It uses :attr:`self.execution_trace` to go back at the state where the navigator was at the "tiling time".

        :param index_in_navigation: "tiling index" in the generated sequence
        :type index_in_navigation: int

        :see also: The list of the parameters of the model whose values are stored in the execution trace is defined in
        :attr:`self.execution_trace_parameters`.

        """

        self.logger.debug(
            "GO TO ANTERIOR STATE USING EXECUTION TRACE\nGoing back to state when {} was generated:\n{}".format(
                index_in_navigation, self.execution_trace[index_in_navigation]))
        history_after_generating_prev = self.execution_trace[index_in_navigation]
        for name_slot, value_slot in history_after_generating_prev.items():
            self.__dict__[name_slot] = value_slot

    def _record_execution_trace(self, index_in_navigation):
        """
        Stores in :attr:`self.execution_trace` the values of different parameters of the model when generating thefu
        event in the sequence at the index given in argument.

        :param index_in_navigation:
        :type index_in_navigation: int

        :see also: The list of the parameters of the model whose values are stored in the execution trace is defined
        in :attr:`self.execution_trace_parameters`.

        """
        trace_index = {}
        for name_slot in self.execution_trace_parameters:
            # trace_index[name_slot] = copy.deepcopy(self.__dict__[name_slot])
            trace_index[name_slot] = copy.deepcopy(self.__dict__[name_slot])

        self.execution_trace[index_in_navigation] = trace_index

    def _update_history_and_taboos(self, index_in_sequence):

        """
        Increases the value associated to the index given in argument in :attr:`self.history_and_taboos`.
        Handles the taboos linked to :attr:`self.max_continuity`.

        :param index_in_sequence:
        :type index_in_sequence: int


        """
        # print("Record history_and_taboos for index {} in sequence.\nPREVIOUSLY:\n{}".format(index_in_sequence,
        #                                                                                     self.history_and_taboos))
        if not self.history_and_taboos[index_in_sequence] is None:
            self.history_and_taboos[index_in_sequence] += 1
        # print("Increases history of selected index.\nNew self.history_and_taboos = {}".format(self.history_and_taboos))
        previous_continuity = None
        previous_position_in_sequence = None
        previous_previous_continuity_in_sequence = None

        if self.current_navigation_index > 0:
            previous_continuity = self.execution_trace[self.current_navigation_index - 1]["current_continuity"]
            # print("Current continuity = {}, previous continuity = {}".format(self.current_continuity, previous_continuity))
            previous_position_in_sequence = self.execution_trace[self.current_navigation_index - 1][
                "current_position_in_sequence"]
            # print("Previous position in sequence = {}".format(previous_position_in_sequence))
            if self.current_navigation_index > 1:
                previous_previous_continuity_in_sequence = self.execution_trace[self.current_navigation_index - 2][
                    "current_continuity"]

        if self.current_continuity == self.max_continuity.get() - 1 and self.current_position_in_sequence + 1 < self._index_last_state():
            self._forbid_indexes([self.current_position_in_sequence + 1])
        # print("Continuity reaches (self.max_continuity - 1). \n"
        #       "New self.history_and_taboos = {}".format(self.history_and_taboos))

        elif not previous_previous_continuity_in_sequence is None \
                and self.current_continuity == 0 \
                and not previous_position_in_sequence is None \
                and previous_position_in_sequence < self._index_last_state() \
                and not previous_continuity is None:  # and previous_continuity > 0:
            self.history_and_taboos[previous_position_in_sequence + 1] = previous_previous_continuity_in_sequence
        # print("Delete taboo set because of max_continuity at last step. \n"
        #       "New self.history_and_taboos = {}".format(self.history_and_taboos))

    def _forbid_indexes(self, indexes):

        """
        Introduces "taboos" (events that cannot be retrieved) in the navigation mechanisms.

        :param indexes: indexes of forbidden indexes (/!\ depending on the model the first event can be at index 0 or 1).
        :type indexes: list(int)

        """
        for i in indexes:
            self.history_and_taboos[i] = None

    def _authorize_indexes(self, indexes):

        """
        Delete the "taboos" (events that cannot be retrieved) in the navigation mechanisms for the states listed in
        the parameter indexes.

        :param indexes: indexes of authorized indexes (/!\ depending on the model the first event can be at index 0 or 1).
        :type indexes: list(int)

        """
        for i in indexes:
            self.history_and_taboos[i] = 0

    def _is_taboo(self, index):

        return self.history_and_taboos[index] is None

    def _delete_taboos(self):

        """
        Delete all the "taboos" (events that cannot be retrieved) in the navigation mechanisms.
        """
        s = []
        for i in range(len(self.sequence)):
            if self._is_taboo(i):
                s.append(i)
        self._authorize_indexes(s)

    def _index_last_state(self) -> int:
        """ Index of the last state in the model."""
        return len(self.labels) - 1
