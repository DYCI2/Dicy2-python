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
Tutorial for the class :class:`~ModelNavigator.FactorOracleNavigator`
in :file:`_Tutorials_/FactorOracleNavigator_tutorial.py`.

Tutorial in :file:`_Tutorials_/FactorOracleNavigator_tutorial.py`

"""
import logging
import random
from abc import abstractmethod, ABC
from typing import Callable, Tuple, Optional, List, Type, TypeVar, Generic

from candidate import Candidate
from candidates import Candidates
from dyci2.navigator import Navigator
from factor_oracle_model import FactorOracle
from factor_oracle_navigator import FactorOracleNavigator
from label import Label
from memory import MemoryEvent, Memory
from model import Model
from parameter import Parametric
from transforms import Transform

M = TypeVar('M', bound=Model)
N = TypeVar('N', bound=Navigator)


class Prospector(Parametric, Generic[M, N], ABC):
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

    def __init__(self, model: M, navigator: N, memory: Memory):
        """
        Constructor for the class FactorOracleNavigator.
        :see also: The class FactorOracle in FactorOracleAutomaton.py

        :Example:

        >>> #sequence = ['A1','B1','B2','C1','A2','B3','C2','D1','A3','B4','C3']
        >>> #labels = [s[0] for s in sequence]
        >>> #FON = FactorOracleGenerator(sequence, labels)

        """
        self.logger = logging.getLogger(__name__)
        self.model: M = model
        self.navigator: N = navigator

        self.content_type: Type[MemoryEvent] = memory.content_type
        self.label_type: Type[Label] = memory.label_type

        self.navigator.clear()

    @abstractmethod
    def navigation_single_step(self, required_label: Optional[Label], **kwargs) -> Candidates:
        """ TODO: Docstring """

    @abstractmethod
    def scenario_single_step(self, labels: List[Label], index_in_generation: int, previous_steps: List[Candidate],
                             authorized_transformations: List[int], **kwargs) -> Candidates:
        """ TODO: Docstring """

    def learn_event(self, event: MemoryEvent, equiv: Optional[Callable] = None):
        """ raises: TypeError if event is incompatible with current memory """
        if isinstance(event, self.content_type) and isinstance(event.label(), self.label_type):
            self.model.learn_event(event, equiv)
            self.navigator.learn_event(event, equiv)
        else:
            raise TypeError(f"Invalid content/label type for event {str(event)}")

    def learn_sequence(self, sequence: List[MemoryEvent], equiv: Optional[Callable] = None):
        """ raises: TypeError if sequence is incompatible with current memory """
        # TODO: Ensure that the sequence always is validated at top level so that the list of MemoryEvents always
        #  has (1) a single LabelType and (2) a single ContentType. OR simply parse with all([isinstance(e) for e in a])
        if len(sequence) > 0 and isinstance(sequence[0], self.content_type) and isinstance(sequence[0].label(),
                                                                                           self.label_type):
            self.model.learn_sequence(sequence, equiv)
            self.navigator.learn_sequence(sequence, equiv)
        else:
            raise TypeError(f"Invalid content/label type for sequence")

    def rewind_generation(self, index_in_navigation: int) -> None:
        self.navigator.rewind_generation(index_in_navigation)

    def feedback(self, output_event: Optional[Candidate]) -> None:
        self.model.feedback(output_event)
        self.navigator.feedback(output_event)

    def encode_with_transform(self, transform: Transform):
        self.model.encode_with_transform(transform)

    def decode_with_transform(self, transform: Transform):
        self.model.decode_with_transform(transform)

    def get_memory(self) -> Memory:
        return self.model.memory

    def set_equiv_function(self, equiv: Callable[[Label, Label], bool]):
        self.model.equiv = equiv
        self.navigator.equiv = equiv


# TODO: this could rather take some abstract FactorOracleLikeModel, FactorOracleLikeNavigator and take these as args.
class Dyci2Prospector(Prospector[FactorOracle, FactorOracleNavigator]):
    def __init__(self, model: Type[FactorOracle], navigator: Type[FactorOracleNavigator],
                 memory: Memory, max_continuity=20, control_parameters=(), history_parameters=(),
                 equiv: Callable = (lambda x, y: x == y), continuity_with_future: Tuple[float, float] = (0.0, 1.0), ):
        super().__init__(memory=memory,
                         model=model(memory=memory, equiv=equiv),
                         navigator=navigator(memory=memory, equiv=equiv,
                                             max_continuity=max_continuity,
                                             control_parameters=control_parameters,
                                             execution_trace_parameters=history_parameters,
                                             continuity_with_future=continuity_with_future))

    # TODO[Jerome]: This one needs some more attention - inconsistencies between randoms ([1..length] vs [0..len-1])
    def prepare_navigation(self, required_labels: List[Label], init: bool = False) -> None:
        if init:
            self.navigator.clear()

        if self.navigator.current_position_in_sequence < 0:
            if len(required_labels) > 0:
                init_states: List[int] = [i for i in range(1, self.model.index_last_state()) if
                                          self.model.direct_transitions.get(i)
                                          and equiv(self.model.direct_transitions.get(i)[0], required_labels[0])]
                # TODO: Handle case where init_states is empty?
                new_position: int = random.randint(0, len(init_states) - 1)
                self.navigator.set_position_in_sequence(new_position)
            else:
                new_position: int = random.randint(1, self.model.index_last_state())
                self.navigator.set_position_in_sequence(new_position)

    def navigation_single_step(self, required_label: Optional[Label], forward_context_length_min: int = 0,
                               print_info: bool = False, shift_index: int = 0,
                               no_empty_event: bool = True) -> Candidates:
        candidates: Candidates = self.model.select_events(index_state=self.navigator.current_position_in_sequence,
                                                          label=required_label,
                                                          forward_context_length_min=forward_context_length_min,
                                                          authorize_direct_transition=True)

        # TODO[Jerome]: I think easiest solution would be to generate a generic binary `index_map` to handle all
        #               index-based filtering and just apply this wherever it is needed
        candidates.data = self.navigator.filter_using_history_and_taboos(candidates.data)

        candidates = self.navigator.weight_candidates(candidates=candidates,
                                                      required_label=required_label,
                                                      model_direct_transitions=self.model.direct_transitions,
                                                      shift_index=shift_index,
                                                      print_info=print_info, no_empty_event=no_empty_event)

        return candidates

    def scenario_single_step(self, labels: List[Label], index_in_generation: int, previous_steps: List[Candidate],
                             authorized_transformations: Optional[List[int]] = None, no_empty_event: bool = True,
                             **kwargs) -> Candidates:
        """ raises: IndexError if `labels` is empty """
        if len(previous_steps) == 0:
            return self._scenario_initial_candidate(labels=labels,
                                                    authorized_transformations=authorized_transformations)
        else:
            return self.navigation_single_step(labels[0], shift_index=index_in_generation,
                                               no_empty_event=no_empty_event)

    def _scenario_initial_candidate(self, labels: List[Label], authorized_transformations: List[int]) -> Candidates:
        authorized_indices: List[int] = [c.index for c in
                                         self.navigator.filter_using_history_and_taboos(
                                             self.model.memory_as_candidates(exclude_last=False,
                                                                             exclude_first=False).data)]

        use_intervals: bool = self._use_intervals(authorized_transformations)
        if use_intervals:
            func_intervals_to_labels: Optional[Callable]
            func_intervals_to_labels = self.label_type.make_sequence_of_intervals_from_sequence_of_labels
            equiv_mod_interval: Optional[Callable] = self.label_type.equiv_mod_interval
        else:
            func_intervals_to_labels = None
            equiv_mod_interval = None

        full_memory: Candidates = self.model.memory_as_candidates(exclude_last=False, exclude_first=False)

        candidates: Candidates = self.navigator.find_prefix_matching_with_labels(
            use_intervals=use_intervals,
            candidates=full_memory,
            labels=labels,
            authorized_indices=authorized_indices,
            authorized_transformations=authorized_transformations,
            sequence_to_interval_fun=func_intervals_to_labels,
            equiv_interval=equiv_mod_interval)

        return candidates

    def _use_intervals(self, authorized_transformations: List[int]):
        return self.label_type is not None and self.label_type.use_intervals \
               and len(authorized_transformations) > 0 and authorized_transformations != [0]
