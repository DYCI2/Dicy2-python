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
import random
from typing import Callable, Tuple, Optional, List, Type

from candidate import Candidate
from candidates import Candidates
from dyci2.navigator import Navigator
# TODO : surchager set use_taboo pour que tous les -1 passent à 0 si on passe à FALSE
# TODO : mode 0 : répétitions authorisées, mode 1 = on prend le min, mode 2, interdire les déjà passés
# TODO : SURCHARGER POUR INTERDIRE LES AUTRES
from label import Label
from memory import MemoryEvent, Memory
from model import Model
from transforms import Transform
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
        """
        Constructor for the class FactorOracleNavigator.
        :see also: The class FactorOracle in FactorOracleAutomaton.py

        :Example:

        >>> #sequence = ['A1','B1','B2','C1','A2','B3','C2','D1','A3','B4','C3']
        >>> #labels = [s[0] for s in sequence]
        >>> #FON = FactorOracleGenerator(sequence, labels)

        """
        # TODO: Assert compatibility between model_class and navigator_class

        # TODO: Model and Navigator should not be initialized here - to handle kwargs properly @ init,
        #       it's better to initialize at parse, alt. pass explicit `model_kwargs` and `navigator_kwargs`
        self.model: Model = model_class(memory, equiv)

        self.navigator: Navigator = navigator_class(memory, equiv, max_continuity, control_parameters,
                                                    history_parameters)
        self.content_type: Type[MemoryEvent] = memory.content_type
        self.label_type: Type[Label] = memory.label_type

        self.navigator.clear()

    def learn_event(self, event: MemoryEvent, equiv: Optional[Callable] = None):
        """ raises: TypeError if event is incompatible with current memory """
        if isinstance(event, self.content_type) and isinstance(event.label(), self.label_type):
            self.model.learn_event(event, equiv)
            self.navigator.learn_event(event, equiv)
        else:
            raise TypeError(f"Invalid content/label type for event {str(event)}")

    def learn_sequence(self, sequence: List[MemoryEvent], equiv: Optional[Callable] = None):
        """ raises: TypeError if sequence is incompatible with current memory """
        # TODO[D]: This is never called! In original code it is called from Model.__init__, which obviously isn't
        #  possible. Need to call this from outside when simplifying calls

        # TODO[C]: Ensure that the sequence always is validated at top level so that the list of MemoryEvents always
        #  has (1) a single LabelType and (2) a single ContentType. OR simply parse with all([isinstance(e) for e in a])
        if len(sequence) > 0 and isinstance(sequence[0], self.content_type) and isinstance(sequence[0].label(),
                                                                                           self.label_type):
            self.model.learn_sequence(sequence, equiv)
            self.navigator.learn_sequence(sequence, equiv)
        else:
            raise TypeError(f"Invalid content/label type for sequence")

    # TODO: This function should ideally not exist once setting of parameter and initialization is handled correctly
    #       Or rather - this function should probably exist but be a `clear` function, where relevant aspects are
    #       migrated to their corresponding parts
    def l_prepare_navigation(self, required_labels: List[Label], equiv: Optional[Callable] = None,
                             new_max_continuity: Optional[int] = None, init: bool = False) -> Callable:
        # TODO: Don't pass this here, use `set_param`
        if equiv is None:
            equiv = self.model.equiv

        # TODO: Don't pass this here, use `set_param`
        if new_max_continuity is not None:
            self.navigator.max_continuity = new_max_continuity

        if init:
            self.navigator.clear()

        # TODO: This is not a good solution, this should be part of an explicit `init/clear` function or smth.
        if self.navigator.current_position_in_sequence < 0:
            if len(required_labels) > 0:
                init_states: List[int] = [i for i in range(1, self.model.memory_length()) if
                                          self.model.direct_transitions.get(i)
                                          and equiv(self.model.direct_transitions.get(i)[0], required_labels[0])]
                # TODO: Handle case where init_states is empty?
                new_position: int = random.randint(0, len(init_states) - 1)
                self.navigator.set_current_position_in_sequence_with_sideeffects(new_position)
            else:
                new_position: int = random.randint(1, self.model.memory_length())
                self.navigator.set_current_position_in_sequence_with_sideeffects(new_position)
        return equiv

    def navigation_single_step(self, required_label: Optional[Label], forward_context_length_min: int = 0,
                               equiv: Optional[Callable] = None, print_info: bool = False,
                               shift_index: int = 0, no_empty_event: bool = True) -> Candidates:
        candidates: Candidates
        # TODO:
        #   current_position_in_sequence: previously output event as defined by feedback function.
        #                                 Generalize this getter as interface function
        #   forward_context_length_min:   Don't pass this here, use `set_param`
        #                                 _
        #   equiv:                        Don't pass this here, use `set_param`
        #                                 _
        #   authorize_direct_transition:  Don't pass this here, use `set_param`. Alt. pass a `**model_kwargs` dict
        #  ---
        #  ALSO: return some sort of `navigator_kwargs` from this function
        candidates = self.model.get_candidates(index_state=self.navigator.current_position_in_sequence,
                                               label=required_label,
                                               forward_context_length_min=forward_context_length_min,
                                               equiv=equiv, authorize_direct_transition=True)

        # TODO[B2]: Move filtering to Prospector (after discussion with Jérôme)
        # TODO: I think easiest solution would be to generate a generic binary `index_map` to handle all index-based
        #  filtering and just apply this wherever it is needed
        candidates.data = self.navigator.filter_using_history_and_taboos(candidates.data)

        # TODO:
        #  model_direct_transition: Part of `navigator_kwargs` passed from Model. Or remove: really don't like this dict
        #                           _
        #  shift_index:             Part of `navigator_kwargs* passed from outside (not from Model - need strategy)
        #                           _
        #  all_memory:              Pass Memory directly instead and convert to whatever format needed
        #                           _
        #  required_label:          Should be part of the interface!!!
        #                           _
        #  print_info:              Handle with logging solution instead
        #                           _
        #  equiv:                   Don't pass this here, use `set_param`
        candidates = self.navigator.weight_candidates(candidates=candidates,
                                                      model_direct_transitions=self.model.direct_transitions,
                                                      shift_index=shift_index,
                                                      all_memory=self.model.l_memory_as_candidates(exclude_last=True),
                                                      required_label=required_label,
                                                      print_info=print_info, equiv=equiv,
                                                      no_empty_event=no_empty_event)
        # TODO[B2]: This should be migrated to feedback function instead
        if candidates.length() > 0:
            self.navigator.set_current_position_in_sequence_with_sideeffects(candidates.at(0).index)

        return candidates

    # TODO: use_intervals, continuity_with_future and equiv should be parameters and/or handled elsewhere rather than
    #       passed here as part of the interface. Same for authorized transformations (maybe, at least isn't relevant
    #       for navigation_single_step so need to handle as optional arg if arg. Actually, this is really the definition
    #       of an invariant: E(index_in_scenario=0, authorized_transformations!=None). Creating function calls like this
    #       will really mess up the idea of invariants if we need to pass these arguments...
    def scenario_single_step(self, labels: List[Label], index_in_generation: int, previous_steps: List[Candidate],
                             use_intervals: bool, continuity_with_future: Tuple[float, float],
                             authorized_transformations: DontKnow, equiv: Optional[Callable],
                             no_empty_event: bool = True) -> Candidates:
        """ raises: IndexError if `labels` is empty """
        if len(previous_steps) == 0:
            return self._scenario_initial_candidate(labels=labels, use_intervals=use_intervals,
                                                    continuity_with_future=continuity_with_future,
                                                    authorized_transformations=authorized_transformations, equiv=equiv)
        else:
            return self.navigation_single_step(labels[0], equiv=equiv, shift_index=index_in_generation,
                                               no_empty_event=no_empty_event)

    def _scenario_initial_candidate(self, labels: List[Label], use_intervals: bool,
                                    continuity_with_future: Tuple[float, float], authorized_transformations: DontKnow,
                                    equiv: Optional[Callable]) -> Candidates:
        full_memory: Candidates = self.model.l_memory_as_candidates(exclude_last=False,
                                                                                   exclude_first=False)
        full_memory.data = self.navigator.filter_using_history_and_taboos(full_memory.data)

        if use_intervals:  # TODO: This should be controlled by prospector (or navigator even?), not passed as argument
            func_intervals_to_labels: Optional[Callable]
            func_intervals_to_labels = self.label_type.make_sequence_of_intervals_from_sequence_of_labels
            equiv_mod_interval: Optional[Callable] = self.label_type.equiv_mod_interval
        else:
            func_intervals_to_labels = None
            equiv_mod_interval = None

        candidates: List[Candidate] = self.navigator.find_prefix_matching_with_labels(
            use_intervals=use_intervals,
            candidates=full_memory.data,
            labels=labels,
            full_memory=full_memory,
            continuity_with_future=continuity_with_future,
            authorized_transformations=authorized_transformations,
            sequence_to_interval_fun=func_intervals_to_labels,
            equiv_interval=equiv_mod_interval,
            equiv=equiv)

        return Candidates(candidates, full_memory.memory)

    def rewind_generation(self, index_in_navigation: int) -> None:
        self.navigator.rewind_generation(index_in_navigation)

    def feedback(self, time: int, output_event: Optional[Candidate]) -> None:
        self.model.feedback(time, output_event)
        self.navigator.feedback(time, output_event)

    # TODO: Should be part of interface but perhaps renamed. Could also be just one function with flag `apply_inverse`
    def l_encode_with_transform(self, transform: Transform):
        # self.model.l_set_sequence([None] + transform.encode_sequence(self.model.sequence[1::]))
        # TODO: Just call `model.encode_with_transform` and `navigator.encode_with_transform`
        self.model.l_set_labels([None] + transform.encode_sequence(self.model.labels[1::]))

    # TODO: Should be part of the interface but perhaps renamed
    def l_decode_with_transform(self, transform: Transform):
        # self.model.l_set_sequence([None] + transform.decode_sequence(self.model.sequence[1::]))
        # TODO: Just call `model.decode_with_transform` and `navigator.decode_with_transform`
        self.model.l_set_labels([None] + transform.decode_sequence(self.model.labels[1::]))

    def get_memory(self) -> Memory:
        return self.model.memory

    ################################################################################################################
    #   TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP TEMP   #
    ################################################################################################################

    # def l_get_no_empty_event(self) -> bool:
    #     return self.navigator.no_empty_event
    #
    # def l_set_no_empty_event(self, v: bool) -> None:
    #     self.navigator.no_empty_event = v
    #
    #     self.navigator.no_empty_event = v
    #
    # def l_get_index_last_state(self) -> int:
    #     return self.model.index_last_state()
    #
    # def l_get_sequence_nonmutable(self) -> List[DontKnow]:
    #     return self.model.sequence
    #
    # def l_get_sequence_maybemutable(self) -> List[DontKnow]:
    #     return self.model.sequence
    #
    # def l_set_sequence(self, sequence: List[DontKnow]):
    #     self.model.sequence = sequence
    #
    # def l_get_labels_nonmutable(self) -> List[DontKnow]:
    #     return self.model.labels
    #
    # def l_get_labels_maybemutable(self) -> List[DontKnow]:
    #     return self.model.labels
    #
    # def l_set_labels(self, labels: List[DontKnow]):
    #     self.model.labels = labels
    #
    # def l_get_position_in_sequence(self) -> int:
    #     return self.navigator.current_position_in_sequence
    #
    # def l_set_position_in_sequence(self, index: int):
    #     self.navigator.set_current_position_in_sequence_with_sideeffects(index)
