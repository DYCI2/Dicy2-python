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
import warnings
from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, List, Type, Dict, Union, TypeVar, Generic

from dyci2.dyci2_label import Dyci2Label
from dyci2.factor_oracle_model import FactorOracle
from dyci2.factor_oracle_navigator import FactorOracleNavigator
from dyci2.model import Model
from dyci2.navigator import Navigator
from dyci2.parameter import Parametric
from dyci2.transforms import Transform, TransposeTransform
from merge.main.candidate import Candidate
from merge.main.candidates import Candidates, ListCandidates
from merge.main.corpus import Corpus
from merge.main.corpus_event import CorpusEvent
from merge.main.exceptions import QueryError, StateError
from merge.main.influence import Influence, LabelInfluence, NoInfluence
from merge.main.prospector import Prospector

M = TypeVar('M', bound=Model)
N = TypeVar('N', bound=Navigator)


class Dyci2Prospector(Prospector, Parametric, Generic[M, N], ABC):
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

    def __init__(self,
                 model: M,
                 navigator: N,
                 corpus: Optional[Corpus],
                 label_type: Type[Dyci2Label],
                 *args, **kwargs):
        """
        Constructor for the class FactorOracleNavigator.
        :see also: The class FactorOracle in FactorOracleAutomaton.py

        :Example:

        >>> #sequence = ['A1','B1','B2','C1','A2','B3','C2','D1','A3','B4','C3']
        >>> #labels = [s[0] for s in sequence]
        >>> #FON = FactorOracleGenerator(sequence, labels)

        """
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.model: M = model
        self.navigator: N = navigator
        self.label_type: Type[Dyci2Label] = label_type

        self.next_output: Optional[Candidates] = None

        self.corpus: Optional[Corpus] = None
        if corpus is not None:
            self.read_memory(corpus)

        self.navigator.clear()

    @abstractmethod
    def prepare_navigation(self, influences: List[Influence], init: bool = False) -> None:
        """ TODO: Docstring (very important!) """

    @abstractmethod
    def free_navigation(self, ...) -> List[Tuple[int, int]]:
        """ TODO: Docstring """

    @abstractmethod
    def simply_guided_navigation(self, ...) -> List[Tuple[int, int]]:
        """ TODO: Docstring"""

    @abstractmethod
    def _scenario_initial_candidate(self, labels: List[Dyci2Label],
                                    authorized_transformations: List[int]) -> Candidates:
        """ TODO: Docstring (very important!) """

    @abstractmethod
    def _clear(self) -> None:
        """ TODO: docstring """

    def read_memory(self, corpus: Corpus, **kwargs) -> None:
        if self.corpus is not None:
            raise StateError(f"Loading a new corpus into an existing {self.__class__.__name__} is not supported")

        self.corpus = corpus
        for event in corpus.events:
            self.learn_event(event)

    def learn_event(self, event: CorpusEvent, **kwargs):
        """
            raises: TypeError if event is incompatible with current memory
                    StateError if no `Corpus` has been loaded
        """
        # TODO Need a strategy for initializing an empty corpus, since this is what normally (always?) happens in DYCI2
        if self.corpus is None:
            raise StateError("No corpus has been loaded")

        label: Optional[Dyci2Label] = event.get_label(self.label_type)
        if isinstance(event, self.corpus.get_content_type()) and label is not None:
            self.corpus.append(event)
            self.model.learn_event(event, label)
            self.navigator.learn_event(event, label)
        else:
            raise TypeError(f"Invalid content/label type for event {str(event)}")

    def process(self,
                influence: Influence,
                forward_context_length_min: int = 0,
                print_info: bool = False,
                shift_index: int = 0,
                no_empty_event: bool = True,
                **kwargs) -> None:
        if self.corpus is None:
            raise StateError("No corpus has been loaded")

        indices_and_scores: List[Tuple[int, int]]
        if isinstance(influence, LabelInfluence) and isinstance(influence.value, Dyci2Label):
            indices_and_scores = self.simply_guided_navigation()
        elif isinstance(influence, NoInfluence):
            indices_and_scores = self.free_navigation()
        else:
            raise QueryError(f"class {self.__class__.__name__} cannot handle "
                             f"influences of type {influence.__class__.__name__}")

        if self.next_output is not None:
            warnings.warn(f"Existing state in {self.__class__.__name__} overwritten without output")

        event_and_scores: List[Tuple[CorpusEvent, int]]
        events_and_scores = [(self.model.internal_event_at(i), s) for i, s in indices_and_scores]
        self.next_output = ListCandidates([Candidate(e, s, None, self.corpus) for e, s in events_and_scores],
                                          self.corpus)

    def initialize_scenario(self, influences: List[Influence], authorized_transformations: List[int]) -> None:
        if self.corpus is None:
            raise StateError("No corpus has been loaded")

        labels: List[Dyci2Label] = []
        for influence in influences:
            if not (isinstance(influence, LabelInfluence) and isinstance(influence.value, Dyci2Label)):
                raise QueryError(f"class {self.__class__.__name__} cannot handle "
                                 f"influences of type {influence.__class__.__name__}")

            labels.append(influence.value)

        if self.next_output is not None:
            warnings.warn(f"Existing state in {self.__class__.__name__} overwritten without output")

        self.next_output = self._scenario_initial_candidate(labels, authorized_transformations)

    def peek_candidates(self) -> Candidates:
        if self.next_output is None:
            raise StateError("No candidates exist in class")

        return self.next_output

    def pop_candidates(self, **kwargs) -> Candidates:
        if self.next_output is None:
            raise StateError("No candidates exist in class")

        output = self.next_output
        self.next_output = None
        return output

    def rewind_generation(self, index_in_navigation: int) -> None:
        self.navigator.rewind_generation(index_in_navigation)

    def feedback(self, output_event: Optional[Candidate], **kwargs) -> None:
        self.model.feedback(output_event)
        self.navigator.feedback(output_event)

    def clear(self) -> None:
        self.next_output = None

    def encode_with_transform(self, transform: Transform):
        self.model.encode_with_transform(transform)

    def decode_with_transform(self, transform: Transform):
        self.model.decode_with_transform(transform)

    def get_corpus(self) -> Corpus:
        return self.corpus

    def set_equiv_function(self, equiv: Callable[[Dyci2Label, Dyci2Label], bool]):
        self.model.equiv = equiv
        self.navigator.equiv = equiv

    def _use_intervals(self, authorized_transformations: List[int]) -> bool:
        return self.label_type is not None and self.label_type.use_intervals \
               and len(authorized_transformations) > 0 and authorized_transformations != [0]


class FactorOracleProspector(Dyci2Prospector[FactorOracle, FactorOracleNavigator]):
    def __init__(self,
                 corpus: Optional[Corpus],
                 label_type: Type[Dyci2Label],
                 max_continuity=20,
                 control_parameters=(),
                 history_parameters=(),
                 equiv: Callable = (lambda x, y: x == y),
                 continuity_with_future: Tuple[float, float] = (0.0, 1.0)):
        super().__init__(model=FactorOracle(equiv=equiv),
                         navigator=FactorOracleNavigator(
                             equiv=equiv,
                             max_continuity=max_continuity,
                             control_parameters=control_parameters,
                             execution_trace_parameters=history_parameters,
                             continuity_with_future=continuity_with_future),
                         corpus=corpus,
                         label_type=label_type)

    # TODO[Jerome]: This one needs some more attention - inconsistencies between randoms ([1..len] vs [0..len-1])
    def prepare_navigation(self, influences: List[Influence], init: bool = False) -> None:
        if not all([isinstance(influence.value, Dyci2Label) for influence in influences]):
            raise QueryError(f"Invalid label type encountered in {self.__class__.__name__}")

        if init:
            self.navigator.clear()

        # Navigator has not generated anything
        if self.navigator.current_position_in_sequence < 0:
            if len(influences) > 0:
                init_states: List[int] = [i for i in range(1, self.model.internal_index_last_state()) if
                                          self.model.direct_transitions.get(i)
                                          and self.model.equiv(self.model.direct_transitions.get(i)[0], influences[0])]
                # TODO: Handle case where init_states is empty?
                new_position: int = random.randint(0, len(init_states) - 1)
                self.navigator.set_position_in_sequence(new_position)
            else:
                new_position: int = random.randint(1, self.model.internal_index_last_state())
                self.navigator.set_position_in_sequence(new_position)

    def free_navigation(self, ...) -> List[Optional[Tuple[int, int]]]:
        pass  # TODO

    def simply_guided_navigation(self, ...) -> List[Optional[Tuple[int, int]]]:
        pass  # TODO

    def _clear(self) -> None:
        pass  # No additional actions required

    # TODO: Migrate/remove
    # def process(self,
    #             influence: Influence,
    #             forward_context_length_min: int = 0,
    #             print_info: bool = False,
    #             shift_index: int = 0,
    #             no_empty_event: bool = True,
    #             **kwargs) -> None:
    #     if self.corpus is None:
    #         raise StateError("No corpus has been loaded")
    #
    #     if isinstance(influence, LabelInfluence) and isinstance(influence.value, Dyci2Label):
    #         required_label: Dyci2Label = influence.value
    #     elif isinstance(influence, NoInfluence):
    #         required_label: None = None
    #     else:
    #         raise QueryError(f"class {self.__class__.__name__} cannot handle "
    #                          f"influences of type {influence.__class__.__name__}")
    #
    #     authorized_indices: List[int] = self.model.get_authorized_indices(
    #         index_state=self.navigator.current_position_in_sequence,
    #         label=required_label,
    #         forward_context_length_min=forward_context_length_min,
    #         authorize_direct_transition=True
    #     )
    #
    #     # TODO[Jerome]: I think an easier solution would be to generate a generic binary `index_map` to handle all
    #     #               index-based filtering and just apply this wherever it is needed
    #     authorized_indices = self.navigator.filter_using_history_and_taboos(authorized_indices)
    #
    #     authorized_indices = self.navigator.weight_candidates(authorized_indices=authorized_indices,
    #                                                           required_label=required_label,
    #                                                           model_direct_transitions=self.model.direct_transitions,
    #                                                           shift_index=shift_index,
    #                                                           print_info=print_info, no_empty_event=no_empty_event)
    #
    #     if self.next_output is not None: # TODO: THIS SHOULD BE HANDLED BY DYCI2PROSPECTOR NOT FOPROSPECTOR. NO INTERNAL ACCESS TO SELF.NEXT_OUTPUT
    #         warnings.warn(f"Existing state in {self.__class__.__name__} overwritten without output")
    #
    #     events: List[CorpusEvent] = [self.model.internal_event_at(i) for i in authorized_indices]
    #     self.next_output = ListCandidates([Candidate(e, 1.0, None, self.corpus) for e in events], self.corpus)

    def _scenario_initial_candidate(self, labels: List[Dyci2Label],
                                    authorized_transformations: List[int]) -> Candidates:
        # use model's internal corpus model to handle the initial None object
        valid_indices: List[int] = list(range(self.model.internal_sequence_length()))
        authorized_indices: List[int] = self.navigator.filter_using_history_and_taboos(valid_indices)

        use_intervals: bool = self._use_intervals(authorized_transformations)
        if use_intervals:
            func_intervals_to_labels: Optional[Callable]
            func_intervals_to_labels = self.label_type.make_sequence_of_intervals_from_sequence_of_labels
            equiv_mod_interval: Optional[Callable] = self.label_type.equiv_mod_interval
        else:
            func_intervals_to_labels = None
            equiv_mod_interval = None

        # FactorOracleModel's representation of the memory is slightly different from the Corpus
        modelled_events: List[Optional[CorpusEvent]]
        modelled_labels: List[Optional[Dyci2Label]]
        modelled_events, modelled_labels = self.model.get_internal_corpus_model()

        index_delta_prefixes: Dict[int, List[List[int]]] = self.navigator.find_prefix_matching_with_labels(
            use_intervals=use_intervals,
            memory_labels=modelled_labels,
            labels_to_match=labels,
            authorized_indices=authorized_indices,
            authorized_transformations=authorized_transformations,
            sequence_to_interval_fun=func_intervals_to_labels,
            equiv_interval=equiv_mod_interval)

        # TODO: This should be modularized to Generator rather than just using best candidate, i.e.
        #  bypass self._choose_prefix_from_list entirely (and delete the function)
        # Select the best candidate.
        s: Optional[int]  # index
        t: int  # transform
        length_selected_prefix: Optional[int]
        s, t, length_selected_prefix = self._choose_prefix_from_list(index_delta_prefixes)

        if s is not None:
            candidate: Candidate = Candidate(modelled_events[s], length_selected_prefix,
                                             TransposeTransform(t), associated_corpus=self.corpus)
            return ListCandidates(candidates=[candidate], associated_corpus=self.corpus)

        else:
            return ListCandidates.new_empty(self.corpus)

    # TODO: This function should be removed entirely and behaviour should be moved to Jury/OutputSelection
    def _choose_prefix_from_list(
            self,
            index_delta_prefixes: Dict[int, List[List[int]]]) -> Tuple[Optional[int], int, Optional[int]]:
        s: Optional[int] = None
        t: int = 0
        length_selected_prefix: Optional[int] = None

        if len(index_delta_prefixes.keys()) > 0:
            length_selected_prefix = max(index_delta_prefixes.keys())
            random_choice: int = random.randint(0, len(index_delta_prefixes[length_selected_prefix]) - 1)
            s: Union[List[int], int] = index_delta_prefixes[length_selected_prefix][random_choice]
            if type(s) == list:
                t = s[1]
                s = s[0]
        else:
            self.logger.debug("No prefix found")

        return s, t, length_selected_prefix
