import logging
import warnings
from typing import Optional, Type, List

from dyci2.candidate_selector import TempCandidateSelector, DefaultFallbackSelector
from dyci2.parameter import Parametric, Parameter, NominalRange
from dyci2.prospector import Dyci2Prospector
from dyci2.transforms import Transform, NoTransform
from merge.main.candidate import Candidate
from merge.main.candidates import Candidates, ListCandidates
from merge.main.corpus import Corpus
from merge.main.corpus_event import CorpusEvent
from merge.main.generator import Generator
from merge.main.influence import NoInfluence, Influence
from merge.main.jury import Jury
from merge.main.query import Query
from merge.main.query import TriggerQuery, InfluenceQuery


class Dyci2Generator(Generator, Parametric):
    def __init__(self,
                 prospector: Dyci2Prospector,
                 jury_type: Type[Jury] = TempCandidateSelector,
                 authorized_transforms: List[int] = (0,),
                 no_empty_event: bool = True):
        self.logger = logging.getLogger(__name__)
        # self._post_filters: List[PostFilter] = post_filters  # TODO: Implement

        self.prospector: Dyci2Prospector = prospector

        self._initialized: bool = False

        self._authorized_transforms: List[int] = list(authorized_transforms)
        self.active_transform: Transform = NoTransform()

        self.no_empty_event: Parameter = Parameter(no_empty_event, NominalRange([False, True]))

        self._jury: Jury = jury_type()
        self._fallback_jury: Jury = DefaultFallbackSelector()

    ################################################################################################################
    # PUBLIC: INHERITED METHODS
    ################################################################################################################

    def process_query(self, query: Query, print_info: bool = False, **kwargs) -> List[Optional[Candidate]]:
        """ raises: RuntimeError if the query is invalid
                    StateError if the internal state of the system is invalid for querying (no corpus loaded, ...)
        """
        self.logger.debug(f"****************************\nPROCESS QUERY: QUERY = \n**************************\n{query}")
        self.logger.debug(f"****************************\nGENERATION MATCHING QUERY: QUERY = \n**************\n{query}")

        output: List[Optional[Candidate]]
        if isinstance(query, TriggerQuery):
            self.logger.debug("GENERATION MATCHING QUERY FREE ...")
            output = self._free_generation(num_events=query.content, init=not self._initialized,
                                           print_info=print_info)
            self.logger.debug("... GENERATION MATCHING QUERY FREE OK")

        elif isinstance(query, InfluenceQuery) and len(query) == 1:
            self.logger.debug("GENERATION MATCHING QUERY LABEL ...")
            output = self._simply_guided_generation(required_labels=query.content,
                                                    init=not self._initialized,
                                                    print_info=print_info)
            self.logger.debug("... GENERATION MATCHING QUERY LABEL OK")

        elif isinstance(query, InfluenceQuery) and len(query) > 1:
            self.logger.debug("GENERATION MATCHING QUERY SCENARIO ...")
            output = self._scenario_based_generation(list_of_labels=query.content)
            self.logger.debug("... GENERATION MATCHING QUERY SCENARIO OK")

        else:
            raise RuntimeError(f"Invalid query type: {query.__class__.__name__}")

        if len(output) > 0:
            self._initialized = True
        return output

    def read_memory(self, corpus: Corpus, **kwargs) -> None:
        # TODO: Handle multicorpus case: learning a corpus in only a particular Prospector => PathSpec argument
        self.prospector.read_memory(corpus, **kwargs)

    def learn_event(self, event: CorpusEvent, **kwargs) -> None:
        """
            raises: TypeError if event is incompatible with current memory
            StateError if no `Corpus` has been loaded
            LabelError if attempting to learn an event with a label type that doesn't exist in the Corpus
        """
        # TODO: Handle multicorpus case: learning a corpus in only a particular Prospector => PathSpec argument
        self.prospector.learn_event(event, **kwargs)

    def clear(self) -> None:
        self._jury.clear()
        self._fallback_jury.clear()
        self.prospector.clear()

    def feedback(self, event: Optional[Candidate], **kwargs) -> None:
        self._jury.feedback(event, **kwargs)
        self._fallback_jury.feedback(event, **kwargs)
        self.prospector.feedback(event, **kwargs)

    ################################################################################################################
    # PUBLIC: CLASS-SPECIFIC METHODS
    ################################################################################################################

    def encode_memory_with_current_transform(self):
        transform: Transform = self.active_transform
        self.prospector.encode_with_transform(transform)

    def decode_memory_with_current_transform(self):
        transform: Transform = self.active_transform
        self.prospector.decode_with_transform(transform)

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def authorized_transforms(self) -> List[int]:
        return self._authorized_transforms

    @authorized_transforms.setter
    def authorized_transforms(self, value: List[int]):
        self._authorized_transforms = value

    ################################################################################################################
    # PRIVATE
    ################################################################################################################

    def _free_generation(self, num_events: int, forward_context_length_min: int = 0, init: bool = False,
                         print_info: bool = False) -> List[Optional[Candidate]]:
        """ Free navigation through the sequence.
        Naive version of the method handling the free navigation in a sequence (random).
        This method has to be overloaded by a model-dependant version when creating a **model navigator** class
        (cf. :mod:`ModelNavigator`).

        # :param num_events: length of the generated sequence
        # :type num_events: int
        # # :param new_max_continuity: new value for self.max_continuity (not changed id no parameter is given)
        # # :type new_max_continuity: int
        # :param forward_context_length_min: minimum length of the forward common context
        # :type forward_context_length_min: int
        # :param init: reinitialise the navigation parameters ?, default : False. (True when starting a new generation)
        # :type init: bool
        # :param equiv: Compararison function given as a lambda function, default: self.equiv.
        # :type equiv: function
        # :param print_info: print the details of the navigation steps
        # :type print_info: bool
        # :return: generated sequence
        # :rtype: list
        # :see also: Example of overloaded method: :meth:`FactorOracleNavigator.free_navigation`.

        :!: **equiv** has to be consistent with the type of the elements in labels.
        :!: The result **strongly depends** on the tuning of the parameters self.max_continuity,
            self.avoid_repetitions_mode, self.no_empty_event.
        """

        self.prospector.prepare_navigation([], init=init)
        sequence: List[Optional[Candidate]] = []
        for i in range(num_events):
            self.prospector.process(influence=NoInfluence(),
                                    forward_context_length_min=forward_context_length_min,
                                    print_info=print_info,
                                    index_in_generation_cycle=i)

            candidates: Candidates = self.prospector.pop_candidates()
            output: Optional[Candidate] = self._decide(candidates, disable_fallback=not self.no_empty_event.value)
            if output is not None:
                output.transform = self.active_transform
                self.feedback(output)

            # TODO[Jerome]: Unlike corresponding lines in scenario-based, this one may add None. Intentional?
            sequence.append(output)

        return sequence

    def _simply_guided_generation(self,
                                  required_labels: List[Influence],
                                  forward_context_length_min: int = 0,
                                  init: bool = False,
                                  print_info: bool = False,
                                  shift_index: int = 0,
                                  break_when_none: bool = False) -> List[Optional[Candidate]]:
        """ Navigation in the sequence, simply guided step by step by an input sequence of label.
        Naive version of the method handling the navigation in a sequence when it is guided by target labels.
        This method has to be overloaded by a model-dependant version when creating a **model navigator** class
        (cf. :mod:`ModelNavigator`).


        # :param required_labels: guiding sequence of labels
        # :type required_labels: list
        # # :param new_max_continuity: new value for self.max_continuity (not changed id no parameter is given)
        # # :type new_max_continuity: int
        # :param forward_context_length_min: minimum length of the forward common context
        # :type forward_context_length_min: int
        # :param init: reinitialise the navigation parameters ?, default : False. (True when starting a new generation)
        # :type init: bool
        # :param equiv: Compararison function given as a lambda function, default: self.equiv.
        # :type equiv: function
        # :param print_info: print the details of the navigation steps
        # :type print_info: bool
        # :return: generated sequence
        # :rtype: list
        # :see also: Example of overloaded method: :meth:`FactorOracleNavigator.free_navigation`.

        :!: **equiv** has to be consistent with the type of the elements in labels.
        :!: The result **strongly depends** on the tuning of the parameters self.max_continuity,
        self.avoid_repetitions_mode, self.no_empty_event.
        """

        self.logger.debug("HANDLE GENERATION MATCHING LABEL...")

        self.prospector.prepare_navigation(required_labels, init)

        sequence: List[Optional[Candidate]] = []
        for (i, label) in enumerate(required_labels):
            self.prospector.process(influence=label,
                                    forward_context_length_min=forward_context_length_min,
                                    print_info=print_info,
                                    index_in_generation_cycle=i + shift_index,
                                    no_empty_event=self.no_empty_event.value)
            candidates: Candidates = self.prospector.pop_candidates()

            if break_when_none and candidates.size() == 0:
                break
            else:
                output: Optional[Candidate] = self._decide(candidates, disable_fallback=not self.no_empty_event.value)
                if output is not None:
                    output.transform = self.active_transform
                    self.feedback(output)

                sequence.append(output)

        return sequence

    def _scenario_based_generation(self, list_of_labels: List[Influence]) -> List[Optional[Candidate]]:
        """
        Generates a sequence matching a "scenario" (a list of labels). The generation process takes advantage of the
        scenario to introduce anticatory behaviour, that is, continuity with the future of the scenario.
        The generation process is divided in successive "generation phases", cf.
        :meth:`~Generator.Generator.handle_scenario_based_generation_one_phase`.

        # :param list_of_labels: "scenario", required labels
        # :type list_of_labels: list or str
        # :return: generated sequence
        # :rtype: list

        """
        generated_sequence: List[Optional[Candidate]] = []
        self.logger.debug("SCENARIO BASED GENERATION 0")

        while len(generated_sequence) < len(list_of_labels):
            current_index: int = len(generated_sequence)
            seq: List[Optional[Candidate]]
            seq = self._handle_scenario_based_generation_one_phase(list_of_labels=list_of_labels[current_index:],
                                                                   original_query_length=len(list_of_labels),
                                                                   shift_index=current_index)
            generated_sequence.extend(seq)

        return generated_sequence

    def _handle_scenario_based_generation_one_phase(self,
                                                    list_of_labels: List[Influence],
                                                    original_query_length: int,
                                                    shift_index: int = 0) -> List[Optional[Candidate]]:
        """
        # :param list_of_labels: "current scenario", suffix of the scenario given in argument to
        # :meth:`Generator.Generator.handle_scenario_based_generation`.
        # :type list_of_labels: list or str

        A "scenario-based" generation phase:
            1. Anticipation step: looking for an event in the memory sharing a common future with the current scenario.
            2. Navigation step: starting from the starting point found in the first step, navigation in the memory using
            :meth:`~Navigator.Navigator.simply_guided_generation` until launching a new phase is necessary.

        """
        self.logger.debug("SCENARIO ONE PHASE 0")

        # Inverse apply transform to memory to reset to initial state of memory (no transform)
        self.decode_memory_with_current_transform()
        self.active_transform = NoTransform()

        generated_sequence: List[Candidate] = []

        # TODO: CRITICAL: CHANGES STATE ABOVE, IF CRASHING IN INITIALIZE_SCENARIO, WILL BREAK
        warnings.warn("EXCEPTIONS HERE WILL BREAK STATE! FIX")

        # Initial candidate (prefix indexing)
        self.prospector.initialize_scenario(influences=list_of_labels,
                                            authorized_transformations=self._authorized_transforms)
        candidates: Candidates = self.prospector.pop_candidates()

        output: Optional[Candidate] = self._decide(candidates, disable_fallback=True)

        # No initial candidate was found: End phase with either fallback or `None` as output
        if output is None:
            fallback_output: Optional[Candidate] = self._decide(ListCandidates([], self.prospector.get_corpus()),
                                                                disable_fallback=not self.no_empty_event.value)
            if fallback_output is not None:
                fallback_output.transform = self.active_transform
            self.feedback(fallback_output)
            return [fallback_output]

        generated_sequence.append(output)
        self.feedback(output)
        self.active_transform = output.transform

        self.logger.debug(f"SCENARIO BASED ONE PHASE SETS POSITION: {output.event.index}")
        self.logger.debug(f"{shift_index} NEW STARTING POINT {output.event.get_label(self.prospector.label_type)} "
                          f"(orig. --): {output.event.index}\n"
                          f"length future = {output.score} - FROM NOW {self.active_transform}")

        self.encode_memory_with_current_transform()

        # TODO[Jerome]: This is probably redundant here no?
        self.prospector.prepare_navigation(list_of_labels[1:])

        # Consecutive candidates
        shift_index: int = original_query_length - len(list_of_labels) + 1
        for (i, influence) in enumerate(list_of_labels[1:]):  # type: int, Influence
            # Search available candidates only (`no_empty_event=False`)
            self.prospector.process(influence=influence,
                                    index_in_generation_cycle=shift_index + i,
                                    no_empty_event=False)
            candidates: Candidates = self.prospector.pop_candidates()

            output: Optional[Candidate] = self._decide(candidates, disable_fallback=True)

            # Candidate found: append and continue iteration
            if output is not None:
                generated_sequence.append(output)
                output.transform = self.active_transform
                self.feedback(output)

            # No candidates found: End phase without appending None/fallback
            else:
                break

        self.logger.debug(f"---------END handle_scenario_based ->> Return {generated_sequence}")
        return generated_sequence

    def _decide(self, candidates: Candidates, disable_fallback: bool = False) -> Optional[Candidate]:
        output: Optional[Candidate] = self._jury.decide(candidates)
        if output is None and not disable_fallback:
            output = self._fallback_jury.decide(candidates)

        return output
