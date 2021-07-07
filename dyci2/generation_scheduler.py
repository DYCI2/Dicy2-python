# -*-coding:Utf-8 -*

#############################################################################
# generation_scheduler.py
# Classes Generator and GenerationHandler
# Generate new sequences from a model of sequence and a query.
# Jérôme Nika, Dimitri Bouche, Jean Bresson, Ken Déguernel - IRCAM STMS Lab
# copyleft 2016
#############################################################################

""" 
Generator
===================
This module defines agents generating new sequences from a "memory" (model of sequence) and generation queries.
Main classes: :class:`~Generator.Generator` (oriented towards offline generation), :class:`~Generator.GenerationHandler`
(oriented towards interactivity).


"""
import logging
import warnings
from typing import Optional, Callable, Tuple, Type, List

from candidate import Candidate
from candidate_selector import CandidateSelector, TempCandidateSelector, DefaultFallbackSelector
from candidates import Candidates
from dyci2.query import Query, FreeQuery, LabelQuery, TimeMode
from dyci2.transforms import TransposeTransform, NoTransform, Transform
# TODO 2021 : Initially default argument for Generator was (lambda x, y: x == y) --> pb with pickle
# TODO 2021 : (because not serialized ?) --> TODO "Abstract Equiv class" to pass objects and not lambda ?
from factor_oracle_model import FactorOracle
from generation_process import GenerationProcess
from label import Label
from memory import MemoryEvent, Memory
from navigator import FactorOracleNavigator
from output import Output
from parameter import Parametric
from prospector import Dyci2Prospector
from utils import format_list_as_list_of_strings


# TODO : SUPPRIMER DANS LA DOC LES FONCTIONS "EQUIV-MOD..." "SEQUENCE TO INTERVAL..."


def basic_equiv(x, y):
    return x == y


# noinspection PyIncorrectDocstring
class GenerationScheduler(Parametric):
    """ The class **Generator** embeds a **model navigator** as "memory" (cf. metaclass
    :class:`~MetaModelNavigator.MetaModelNavigator`) and processes **queries** (class :class:`~Query.Query`) to
    generate new sequences. This class uses pattern matching techniques (cf. :mod:`PrefixIndexing`) to enrich the
    navigation and generation methods offered by the chosen model with ("ImproteK-like") anticipative behaviour.
    More information on "scenario-based generation": see **Nika, "Guiding Human-Computer Music Improvisation:
    introducing Authoring and Control with Temporal Scenarios", PhD Thesis, UPMC Paris 6, Ircam, 2016**
    (https://tel.archives-ouvertes.fr/tel-01361835)
    and
    **Nika, Chemillier, Assayag, "ImproteK: introducing scenarios into human-computer music improvisation",
    ACM Computers in Entertainment, Special issue on Musical metacreation Part I, 2017**
    (https://hal.archives-ouvertes.fr/hal-01380163).

    The key differences between :class:`~Generator.Generator` and :class:`~Generator.GenerationHandler` are:
        * :meth:`Generator.receive_query` / :meth:`GenerationHandler.receive_query`
        * :meth:`Generator.process_query` / :meth:`GenerationHandler.process_query`

    # :param model_navigator:
    # :type model_navigator: str

    #:param memory: "Model navigator" inheriting from (a subclass of) :class:`~Model.Model` and (a subclass of)
    :class:`~Navigator.Navigator`.
    :type prospector: cf. :mod:`ModelNavigator` and :mod:`MetaModelNavigator`

    #:param initial_query:
    # :type initial_query: bool
    #:param current_generation_query:
    # :type current_generation_query: :class:`~Query.Query`

    #:param current_generation_output:
    # :type current_generation_output: list
    #:param transfo_current_generation_output:
    # :type transfo_current_generation_output: list

    :param continuity_with_future:
    :type continuity_with_future: list

    #:param current_transformation_memory:
    :type active_transform: cf. :mod:`Transforms`
    :param authorized_tranformations:
    :type authorized_tranformations: list(int)
    #:param sequence_to_interval_fun:
    #:type sequence_to_interval_fun: function
    #:param equiv_mod_interval:
    #:type equiv_mod_interval: function


    :see also: :mod:`GeneratorBuilder`, automatic instanciation of Generator objects and GenerationHandler objects from
    annotation files.
    :see also: **Tutorial in** :file:`_Tutorials_/Generator_tutorial.py`.


    :Example:

    >>> #sequence_1 = ['A1','B1','B2','C1','A2','B3','C2','D1','A3','B4','C3']
    >>> #labels_1 = [s[0] for s in sequence_1]
    >>> #generator_1 = GenerationHandlerNew(sequence=sequence_1, labels=labels_1, model_navigator = "FactorOracleNavigator")
    >>> #
    >>> #sequence_2 = make_sequence_of_chord_labels(["d m7(1)", "d m7(2)", "g 7(3)", "g 7(4)", "c maj7(5)","c maj7(6)","c# maj7(7)","c# maj7(8)", "d# m7(9)", "d# m7(10)", "g# 7(11)", "g# 7(12)", "c# maj7(13)", "c# maj7(14)"])
    >>> #labels_2 = make_sequence_of_chord_labels(["d m7", "d m7", "g 7", "g 7", "c maj7","c maj7","c# maj7","c# maj7", "d# m7", "d# m7", "g# 7", "g# 7", "c# maj7", "c# maj7"])
    >>> #authorized_intervals = list(range(-2,6))
    >>> #generator_2 = GenerationHandlerNew(sequence = sequence_2, labels = labels_2, model_navigator = "FactorOracleNavigator", authorized_tranformations = authorized_intervals, sequence_to_interval_fun = chord_labels_sequence_to_interval)

    """

    def __init__(self, memory: Memory, model_class: Type[FactorOracle] = FactorOracle,
                 navigator_class: Type[FactorOracleNavigator] = FactorOracleNavigator,
                 equiv: Optional[Callable] = (lambda x, y: x == y), authorized_tranformations=(0,),
                 continuity_with_future: Tuple[float, float] = (0.0, 1.0)):
        self.logger = logging.getLogger(__name__)
        if equiv is None:
            equiv = basic_equiv
        self.prospector: Dyci2Prospector = Dyci2Prospector(model=model_class, navigator=navigator_class, memory=memory,
                                                           equiv=equiv, continuity_with_future=continuity_with_future)

        self.initialized: bool = True

        self.authorized_transformations: List[int] = list(authorized_tranformations)
        self.active_transform: Transform = NoTransform()

        self._performance_time: int = 0

        self.generation_process: GenerationProcess = GenerationProcess()
        self.candidate_selector: CandidateSelector = TempCandidateSelector()
        self.fallback_selector: CandidateSelector = DefaultFallbackSelector()

    def learn_event(self, event: MemoryEvent) -> None:
        """ Learn a new event in the memory (model navigator).
            raises: TypeError if event is incompatible with current memory """
        self.prospector.learn_event(event=event)

    def learn_sequence(self, sequence: List[MemoryEvent]) -> None:
        """ Learn a new sequence in the memory (model navigator).
            raises: TypeError if sequence is incompatible with current memory """
        self.prospector.learn_sequence(sequence=sequence)

    def process_query(self, query: Query) -> int:
        """ raises: RuntimeError if receiving a relative query as the first query. """
        self.logger.debug("\n--------------------")
        self.logger.debug(f"current_performance_time: {self._performance_time}")
        self.logger.debug(f"current_generation_time: {self.generation_process.generation_time}")

        if self.initialized and self._performance_time < 0 and query.time_mode == TimeMode.RELATIVE:
            # TODO[Jerome]: Is this really a good strategy? Or should it just assume this as ABSOLUTE(NOW)?
            raise RuntimeError("Cannot handle a relative query as the first query")

        self.logger.debug("PROCESS QUERY\n", query)
        if query.time_mode == TimeMode.RELATIVE:
            query.to_absolute(self._performance_time)
            self.logger.debug("QUERY ABSOLUTE\n", query)

        if not self.initialized:
            self._performance_time = 0

        generation_index: int = query.start_date
        self.logger.debug(f"generation_index: {generation_index}")
        if 0 < generation_index < self.generation_process.generation_time:
            self.logger.debug(f"USING EXECUTION TRACE : generation_index = {generation_index} : "
                              f"generation_time = {self.generation_process.generation_time}")
            self.prospector.rewind_generation(generation_index - 1)

        # TODO[Jerome] UNSOLVED! Strategy for execution trace is still not solved
        self.prospector.navigator.current_navigation_index = generation_index - 1

        output: List[Optional[Candidate]] = self._process_query(query)

        self.logger.debug(f"self.current_generation_output {output}")
        self.generation_process.add_output(generation_index, output)

        return generation_index

    def _process_query(self, query: Query) -> List[Optional[Candidate]]:
        self.logger.debug("****************************\nPROCESS QUERY: QUERY = \n****************************", query)
        self.logger.debug("****************************\nGENERATION MATCHING QUERY: QUERY = \n****************", query)

        output: List[Optional[Candidate]]
        if isinstance(query, FreeQuery):
            self.logger.debug("GENERATION MATCHING QUERY FREE ...")
            output = self.free_generation(num_events=query.num_events, init=not self.initialized,
                                          print_info=query.print_info)
            self.logger.debug("... GENERATION MATCHING QUERY FREE OK")

        elif isinstance(query, LabelQuery) and len(query.labels) == 1:
            self.logger.debug("GENERATION MATCHING QUERY LABEL ...")
            output = self.simply_guided_generation(required_labels=query.labels, init=not self.initialized,
                                                   print_info=query.print_info)
            self.logger.debug("... GENERATION MATCHING QUERY LABEL OK")

        elif isinstance(query, LabelQuery) and len(query.labels) > 1:
            self.logger.debug("GENERATION MATCHING QUERY SCENARIO ...")
            output = self.scenario_based_generation(list_of_labels=query.labels, print_info=query.print_info)
            self.logger.debug("... GENERATION MATCHING QUERY SCENARIO OK")

        else:
            raise RuntimeError(f"Invalid query type: {query.__class__.__name__}")

        if len(output) > 0:
            self.initialized = True
        return output

    def free_generation(self, num_events: int, forward_context_length_min: int = 0, init: bool = False,
                        print_info: bool = False) -> List[Optional[Candidate]]:
        """ Free navigation through the sequence.
        Naive version of the method handling the free navigation in a sequence (random).
        This method has to be overloaded by a model-dependant version when creating a **model navigator** class
        (cf. :mod:`ModelNavigator`).

        :param num_events: length of the generated sequence
        :type num_events: int
        # :param new_max_continuity: new value for self.max_continuity (not changed id no parameter is given)
        # :type new_max_continuity: int
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

        self.prospector.prepare_navigation([], init=init)
        sequence: List[Optional[Candidate]] = []
        for i in range(num_events):
            candidates: Candidates
            candidates = self.prospector.navigation_single_step(required_label=None,
                                                                forward_context_length_min=forward_context_length_min,
                                                                print_info=print_info, shift_index=i)

            output: Optional[Candidate] = self.decide(candidates, disable_fallback=False)
            if output is not None:
                output.transform = self.active_transform
                self.feedback(output)

            # TODO[Jerome]: Unlike corresponding lines in scenario-based, this one may add None. Intentional?
            sequence.append(output)

        return sequence

    def simply_guided_generation(self, required_labels: List[Label],
                                 forward_context_length_min: int = 0, init: bool = False,
                                 print_info: bool = False, shift_index: int = 0,
                                 break_when_none: bool = False) -> List[Optional[Candidate]]:
        """ Navigation in the sequence, simply guided step by step by an input sequence of label.
        Naive version of the method handling the navigation in a sequence when it is guided by target labels.
        This method has to be overloaded by a model-dependant version when creating a **model navigator** class
        (cf. :mod:`ModelNavigator`).


        :param required_labels: guiding sequence of labels
        :type required_labels: list
        # :param new_max_continuity: new value for self.max_continuity (not changed id no parameter is given)
        # :type new_max_continuity: int
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

        self.logger.debug("HANDLE GENERATION MATCHING LABEL...")

        self.prospector.prepare_navigation(required_labels, init)

        sequence: List[Optional[Output]] = []
        for (i, label) in enumerate(required_labels):
            candidates: Candidates
            candidates = self.prospector.navigation_single_step(required_label=label,
                                                                forward_context_length_min=forward_context_length_min,
                                                                print_info=print_info,
                                                                shift_index=i + shift_index)

            if break_when_none and candidates.length() == 0:
                break
            else:
                output: Optional[Candidate] = self.decide(candidates, disable_fallback=False)
                if output is not None:
                    output.transform = self.active_transform
                    self.feedback(output)

                sequence.append(output)

        return sequence

    def scenario_based_generation(self, list_of_labels: List[Label],
                                  print_info: bool = False) -> List[Optional[Candidate]]:
        """
        Generates a sequence matching a "scenario" (a list of labels). The generation process takes advantage of the
        scenario to introduce anticatory behaviour, that is, continuity with the future of the scenario.
        The generation process is divided in successive "generation phases", cf.
        :meth:`~Generator.Generator.handle_scenario_based_generation_one_phase`.

        :param list_of_labels: "scenario", required labels
        :type list_of_labels: list or str
        :return: generated sequence
        :rtype: list

        """
        generated_sequence: List[Optional[Candidate]] = []
        self.logger.debug("SCENARIO BASED GENERATION 0")
        while len(generated_sequence) < len(list_of_labels):
            current_index: int = len(generated_sequence)
            seq: List[Candidate]
            seq = self.handle_scenario_based_generation_one_phase(list_of_labels=list_of_labels[current_index:],
                                                                  original_query_length=len(list_of_labels),
                                                                  print_info=print_info,
                                                                  shift_index=current_index)

            if len(seq) > 0:
                generated_sequence.extend(seq)
            else:
                # TODO: Clunky to create Candidates instance for this. Also - doesn't have access to taboo/mask.
                fallback_output: Optional[Candidate] = self.decide(Candidates([], self.prospector.get_memory()))
                if fallback_output is not None:
                    fallback_output.transform = self.active_transform
                    # New code:
                    self.feedback(fallback_output)
                    # Old code for compatibility with line below:
                    #   self.prospector.navigator.set_position_in_sequence(fallback_output.index)
                else:
                    # TODO[Jerome]: Is this really a good idea? Shouldn't it be random state?
                    self.prospector.navigator.set_position_in_sequence(0)
                generated_sequence.append(fallback_output)

        return generated_sequence

    def handle_scenario_based_generation_one_phase(self, list_of_labels: List[Label], original_query_length: int,
                                                   print_info: bool = False, shift_index: int = 0) -> List[Candidate]:
        """

        :param list_of_labels: "current scenario", suffix of the scenario given in argument to
        :meth:`Generator.Generator.handle_scenario_based_generation`.
        :type list_of_labels: list or str

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

        # Initial candidate (prefix indexing)
        candidates: Candidates = self.prospector.scenario_single_step(
            labels=list_of_labels,
            index_in_generation=shift_index,
            authorized_transformations=self.authorized_transformations,
            previous_steps=generated_sequence)

        output: Optional[Candidate] = self.decide(candidates, disable_fallback=True)
        if output is None:
            return []

        generated_sequence.append(output)
        self.active_transform = output.transform
        self.feedback(output)

        self.logger.debug(f"SCENARIO BASED ONE PHASE SETS POSITION: {output.index}")
        self.logger.debug(f"current_position_in_sequence: {output.index}")
        self.logger.debug(f"{shift_index} NEW STARTING POINT {output.event.label()} (orig. --): {output.index}\n"
                          f"length future = {output.score} - FROM NOW {self.active_transform}")

        self.encode_memory_with_current_transform()

        # TODO[Jerome]: This is probably redundant here no?
        self.prospector.prepare_navigation(list_of_labels[1:])

        # Consecutive candidates
        shift_index: int = original_query_length - len(list_of_labels) + 1
        for (i, label) in enumerate(list_of_labels[1:]):  # type: int, Label
            candidates: Candidates = self.prospector.scenario_single_step(labels=[label],
                                                                          index_in_generation=shift_index + i,
                                                                          previous_steps=generated_sequence,
                                                                          no_empty_event=False)

            output: Optional[Candidate] = self.decide(candidates, disable_fallback=True)

            if output is not None:
                generated_sequence.append(output)
                output.transform = self.active_transform
                self.feedback(output)

            else:
                # End phase if no candidates are found
                break

        self.logger.debug(f"---------END handle_scenario_based ->> Return {generated_sequence}")
        return generated_sequence

    def feedback(self, output_event: Optional[Candidate]) -> None:
        self.candidate_selector.feedback(output_event)
        self.prospector.feedback(output_event)

    def decide(self, candidates: Candidates, disable_fallback: bool = False) -> Optional[Candidate]:
        output: Optional[Candidate] = self.candidate_selector.decide(candidates)
        if output is None and not disable_fallback:
            output = self.fallback_selector.decide(candidates)

        return output

    def start(self):
        """ Sets :attr:`self.current_performance_time` to 0."""
        self._performance_time = 0

    def set_equiv_function(self, equiv: Callable[[Label, Label], bool]):
        self.prospector.set_equiv_function(equiv=equiv)

    # TODO : EPSILON POUR LANCER NOUVELLE GENERATION
    def update_performance_time(self, new_time: Optional[int] = None):
        if new_time is not None:
            self._performance_time = new_time
        else:
            self._performance_time = -1

        print(f"NEW PERF TIME = {self._performance_time}")
        # TODO : EPSILON POUR LANCER NOUVELLE GENERATION
        if self.generation_process.generation_time < self._performance_time:
            old = self.generation_process.generation_time
            self.generation_process.update_generation_time(self._performance_time)
            print(f"CHANGED PERF TIME = {old} --> {self.generation_process.generation_time}")

    def inc_performance_time(self, increment: Optional[int] = None):
        old_time = self._performance_time

        new_time: Optional[int] = None

        if increment is not None:
            if old_time > -1:
                new_time = old_time + increment
            else:
                new_time = increment

        self.update_performance_time(new_time=new_time)

    # TODO PLUS GENERAL FAIRE SLOT "POIGNEE" DANS CLASSE TRANSFORM
    # OU METTRE CETTE METHODE DANS CLASSE TRANSFORM
    def formatted_output_couple_content_transfo(self):
        result = []
        for i in range(0, len(self.current_generation_output)):
            # TODO faire plus générique
            # TODO... CAR POUR L'INSTANT SI ON RENCONTRE UN NONE ON OUTPUT TOUT JUSQU'A AVANT... C'EST TOUT
            if self.current_generation_output[i]:
                if self.transfo_current_generation_output[i] and type(
                        self.transfo_current_generation_output[i]) == TransposeTransform:
                    result.append(
                        [self.current_generation_output[i], self.transfo_current_generation_output[i].semitone])
                else:
                    result.append([self.current_generation_output[i], 0])
            else:
                return result
        # print("TRANSPO")
        # print(result)
        return result

    def formatted_output_string(self):
        return format_list_as_list_of_strings(self.current_generation_output)

    def formatted_generation_trace_string(self):
        return format_list_as_list_of_strings(self.generation_trace)

    def encode_memory_with_current_transform(self):
        transform: Transform = self.active_transform
        self.prospector.encode_with_transform(transform)

    def decode_memory_with_current_transform(self):
        transform: Transform = self.active_transform
        self.prospector.decode_with_transform(transform)

    @property
    def performance_time(self) -> int:
        return self._performance_time

    @performance_time.setter
    def performance_time(self, value: int):
        self._performance_time = value
        print("New value of current performance time: {}".format(self._performance_time))
