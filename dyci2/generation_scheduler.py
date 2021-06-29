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
import itertools
from typing import Optional, Callable, Tuple, Type, List

from candidate_selector import CandidateSelector, TempCandidateSelector
from dyci2.query import Query, FreeQuery, LabelQuery, TimeMode
from dyci2.transforms import *
# TODO 2021 : Initially default argument for Generator was (lambda x, y: x == y) --> pb with pickle
# TODO 2021 : (because not serialized ?) --> TODO "Abstract Equiv class" to pass objects and not lambda ?
from factor_oracle_model import FactorOracle
from generation_process import GenerationProcess
from memory import MemoryEvent, Memory
from model import Model
from navigator import Navigator, FactorOracleNavigator
from output import Output
from prospector import Prospector
from utils import format_list_as_list_of_strings, DontKnow


# TODO : SUPPRIMER DANS LA DOC LES FONCTIONS "EQUIV-MOD..." "SEQUENCE TO INTERVAL..."


def basic_equiv(x, y):
    return x == y


# noinspection PyIncorrectDocstring
class GenerationScheduler:
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
    :type current_generation_output: list
    #:param transfo_current_generation_output:
    :type transfo_current_generation_output: list

    :param continuity_with_future:
    :type continuity_with_future: list

    #:param current_transformation_memory:
    :type current_transformation_memory: cf. :mod:`Transforms`
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

    def __init__(self, memory: Memory, model_class: Type[Model] = FactorOracle,
                 navigator_class: Type[Navigator] = FactorOracleNavigator,
                 equiv: Optional[Callable] = (lambda x, y: x == y), authorized_tranformations=(0,),
                 continuity_with_future=(0.0, 1.0)):
        # FIXME[MergeState]: A[], B[], C[], D[]

        if equiv is None:
            equiv = basic_equiv
        self.prospector: Prospector = Prospector(model_class=model_class, navigator_class=navigator_class,
                                                 memory=memory, equiv=equiv)
        self.content_type: Type[MemoryEvent] = memory.content_type  # TODO[C]: Should not be here

        self.uninitialized: bool = True  # TODO[C] Positive name `initialized` would be better + invert all conditions
        self.current_generation_output: List[Optional[Candidate]] = []
        self.transfo_current_generation_output: List[Transform] = []

        self.authorized_transformations: List[DontKnow] = list(authorized_tranformations)
        # TODO: Rather NoTransform?
        self.current_transformation_memory: Optional[Transform] = None
        self.continuity_with_future: DontKnow = list(continuity_with_future)
        self.performance_time: int = 0

        self.generation_process: GenerationProcess = GenerationProcess()
        self.candidate_selector: CandidateSelector = TempCandidateSelector()

    def __setattr__(self, name_attr, val_attr):
        object.__setattr__(self, name_attr, val_attr)
        if name_attr == "current_performance_time":
            print("New value of current performance time: {}".format(self.performance_time))

    def learn_event(self, event: MemoryEvent) -> None:
        # FIXME[MergeState]: A[x], B[], C[], D[]
        """ Learn a new event in the memory (model navigator).
            raises: TypeError if event is incompatible with current memory """
        self.prospector.learn_event(event=event)

    def learn_sequence(self, sequence: List[MemoryEvent]) -> None:
        # FIXME[MergeState]: A[x], B[], C[], D[]
        """ Learn a new sequence in the memory (model navigator).
            raises: TypeError if sequence is incompatible with current memory """
        self.prospector.learn_sequence(sequence=sequence)

    def process_query(self, query: Query) -> int:
        # TODO[B]: This entire function is basically a long list of side effects to distribute all over the system given
        #   various cases. A better solution would be to clean most of these up and pass along. The only important thing
        #   here is go_to_anterior_state_using_execution_trace
        """ raises: RuntimeError if receiving a relative query as the first query. """
        print("\n--------------------")
        print(f"current_performance_time: {self.performance_time}")
        print(f"current_generation_time: {self.generation_process.generation_time}")

        if not self.uninitialized and self.performance_time < 0 and query.time_mode == TimeMode.RELATIVE:
            # TODO: Is this really a good strategy? Or should it just assume this as ABSOLUTE(NOW)?
            raise RuntimeError("Cannot handle a relative query as the first query")

        # TODO[B] If invariant above is useless or if we can handle this elsewhere,
        #  handle query.to_absolute in scheduler instead
        print("PROCESS QUERY\n", query)
        if query.time_mode == TimeMode.RELATIVE:
            query.to_absolute(self.performance_time)
            print("QUERY ABSOLUTE\n", query)

        if self.uninitialized:
            self.performance_time = 0
        generation_index: int = query.start_date
        print(f"generation_index: {generation_index}")
        if 0 < generation_index < self.generation_process.generation_time:
            print(f"USING EXECUTION TRACE : generation_index = {generation_index} : "
                  f"generation_time = {self.generation_process.generation_time}")
            self.prospector.rewind_generation(generation_index - 1)

        # TODO[B] UNSOLVED! Massive side-effect. Exactly what is current_navigation_index?
        self.prospector.navigator.current_navigation_index = generation_index - 1
        # TODO[B]: UNSOLVED! generator_process_query should return output, not store it in current_generation_output
        self._process_query(query)

        print(f"self.current_generation_output {self.current_generation_output}")
        self.generation_process.add_output(generation_index, self.current_generation_output)

        # TODO[B] Can this return generation_index instead? Or is query.start_date changed somewhere along the line?
        return query.start_date

    def _process_query(self, query: Query) -> List[Optional[Candidate]]:
        print("**********************************\nPROCESS QUERY: QUERY = \n**********************************", query)
        print("**********************************\nGENERATION MATCHING QUERY: QUERY = \n**********************", query)
        output: List[Optional[Candidate]]
        # TODO[B]: UNSOLVED! probably unnecessary side-effect but discuss w/ Jerome before removing
        self.transfo_current_generation_output = []
        if isinstance(query, FreeQuery):
            print("GENERATION MATCHING QUERY FREE ...")
            output = self.free_generation(num_events=query.num_events, init=self.uninitialized,
                                          print_info=query.print_info)
            print("... GENERATION MATCHING QUERY FREE OK")
            self.transfo_current_generation_output = [self.current_transformation_memory] * len(output)

        elif isinstance(query, LabelQuery) and len(query.labels) == 1:
            print("GENERATION MATCHING QUERY LABEL ...")
            # TODO[C] Find solution for transforms: since `scenario_based` also calls simply_guided,
            #  transform handling cannot be in `simply_guided`
            self.encode_memory_with_current_transform()
            output = self.simply_guided_generation(required_labels=query.labels, init=self.uninitialized,
                                                   print_info=query.print_info)
            self.decode_memory_with_current_transform()
            self.transfo_current_generation_output = [self.current_transformation_memory] * len(output)
            print("... GENERATION MATCHING QUERY LABEL OK")

        elif isinstance(query, LabelQuery) and len(query.labels) > 1:
            print("GENERATION MATCHING QUERY SCENARIO ...")
            output = self.scenario_based_generation(list_of_labels=query.labels, print_info=query.print_info)
            print("... GENERATION MATCHING QUERY SCENARIO OK")

        else:
            raise RuntimeError(f"Invalid query type: {query.__class__.__name__}")

        # TODO[B] Again with the side effects...
        self.current_generation_output = output
        if len(self.current_generation_output) > 0:
            self.uninitialized = False
        return output

    def free_generation(self, num_events: int, new_max_continuity: Optional[DontKnow] = None,
                        forward_context_length_min: int = 0, init: bool = False, equiv: Callable = None,
                        print_info: bool = False) -> List[Optional[Candidate]]:
        """ Free navigation through the sequence.
        Naive version of the method handling the free navigation in a sequence (random).
        This method has to be overloaded by a model-dependant version when creating a **model navigator** class
        (cf. :mod:`ModelNavigator`).

        :param num_events: length of the generated sequence
        :type num_events: int
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

        self.encode_memory_with_current_transform()
        equiv = self.prospector.l_prepare_navigation([], equiv, new_max_continuity, init)
        sequence: List[Optional[Candidate]] = []
        for i in range(num_events):
            candidates: List[Candidate]
            candidates = self.prospector.navigation_single_step(required_label=None,
                                                                forward_context_length_min=forward_context_length_min,
                                                                equiv=equiv, print_info=print_info, shift_index=i)

            sequence.append(self.candidate_selector.decide(candidates))

        self.decode_memory_with_current_transform()
        return sequence

    def simply_guided_generation(self, required_labels: List[Label],
                                 new_max_continuity: Optional[Tuple[float, float]] = None,
                                 forward_context_length_min: int = 0, init: bool = False,
                                 equiv: Optional[Callable] = None, print_info: bool = False, shift_index: int = 0,
                                 break_when_none: bool = False) -> List[Optional[Candidate]]:
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

        print("HANDLE GENERATION MATCHING LABEL...")

        equiv = self.prospector.l_prepare_navigation(required_labels, equiv, new_max_continuity, init)

        sequence: List[Optional[Output]] = []
        for (i, label) in enumerate(required_labels):
            candidates: List[Candidate]
            candidates = self.prospector.navigation_single_step(required_label=label,
                                                                forward_context_length_min=forward_context_length_min,
                                                                equiv=equiv, print_info=print_info,
                                                                shift_index=i + shift_index)

            if break_when_none and len(candidates) == 0:
                break
            else:
                sequence.append(self.candidate_selector.decide(candidates))

        # TODO: This was most likely a misinterpretation of the original code and should probably be removed
        # sequence: List[Candidate] = [c for c in sequence if c is not None]

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
        print("SCENARIO BASED GENERATION 0")
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
                generated_sequence.append(self.l_scenario_default_case())

        return generated_sequence

    # TODO: We need a strategy to handle the default case in a modular manner
    def l_scenario_default_case(self) -> Optional[Candidate]:
        if self.prospector.navigator.no_empty_event and \
                self.prospector.navigator.current_position_in_sequence < self.prospector.model.index_last_state():
            print("NO EMPTY EVENT")
            next_index: int = self.prospector.navigator.current_position_in_sequence + 1
            self.prospector.navigator.set_current_position_in_sequence_with_sideeffects(next_index)
            # TODO[CRITICAL]: This does not handle transforms coherently with surrounding code
            #  (original code didn't either)
            return Candidate(self.prospector.model.sequence[next_index], next_index, 1.0, NoTransform())

        else:
            print("EMPTY EVENT")
            self.prospector.navigator.set_current_position_in_sequence_with_sideeffects(0)
            return None

    # TODO : PAS OPTIMAL DU TOUT D'ENCODER DECODER A CHAQUE ETAPE !!!!!!!!
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
        print("SCENARIO ONE PHASE 0")

        # Inverse apply transform to memory
        self.decode_memory_with_current_transform()
        self.current_transformation_memory = None

        candidates: List[Candidate] = self.prospector.scenario_based_generation(
            labels=list_of_labels,
            use_intervals=self._use_intervals(),
            continuity_with_future=self.continuity_with_future,
            authorized_transformations=self.authorized_transformations,
            equiv=self.prospector.model.equiv)

        candidate: Optional[Candidate] = self.candidate_selector.decide(candidates)
        if candidate is None:
            return []

        print(f"SCENARIO BASED ONE PHASE SETS POSITION: {candidate.index}")
        self.prospector.navigator.set_current_position_in_sequence_with_sideeffects(candidate.index)
        print(f"current_position_in_sequence: {candidate.index}")

        # TODO: This is *NOT* a good solution for transforms, nor should it be handled here
        if candidate.transform is not None and candidate.transform != 0:
            transform: Transform = TransposeTransform(candidate.transform)
            # NOTE! ContentType was never set in orig. Therefore commented out
            # candidate: [Candidate] = transform.encode(candidate)
            candidate.transform = transform
            # TODO: Side effect
            self.current_transformation_memory = transform

        if print_info:
            print(f"{shift_index} NEW STARTING POINT {candidate.event.label()} (orig. --): {candidate.index}\n"
                  f"length future = {candidate.score} - FROM NOW {self.current_transformation_memory}")

        # TODO: Side effect: This should probably be in GenerationProcess?
        #  Or perhaps not even necessary since stored in Output
        self.transfo_current_generation_output.append(self.current_transformation_memory)

        # Apply transform from initial candidate to memory
        self.encode_memory_with_current_transform()

        # TODO: This is not ok at all!! - pass this as value to simply_guided_generation.
        #  Also move: `no_empty_event` should not part of Navigator but of GenerationScheduler/Generator.
        aux: bool = self.prospector.navigator.no_empty_event
        # In order to begin a new navigation phase when this method returns a "None" event
        self.prospector.navigator.no_empty_event = False

        seq: List[Optional[Candidate]]
        seq = self.simply_guided_generation(required_labels=list_of_labels[1::], init=False,
                                            print_info=print_info,
                                            shift_index=original_query_length - len(list_of_labels) + 1,
                                            break_when_none=True)

        self.prospector.navigator.no_empty_event = aux

        generated_sequence: List[Candidate] = [candidate]
        for output_event in itertools.takewhile(lambda o: o is not None, seq):  # type: Candidate
            output_event.transform = self.current_transformation_memory
            generated_sequence.append(output_event)
            # TODO: Side effect
            self.transfo_current_generation_output.append(self.current_transformation_memory)

        print(f"---------END handle_scenario_based ->> Return {generated_sequence}")
        return generated_sequence

    def start(self):
        """ Sets :attr:`self.current_performance_time` to 0."""
        self.performance_time = 0

    # TODO : EPSILON POUR LANCER NOUVELLE GENERATION
    def update_performance_time(self, new_time: Optional[int] = None):
        if new_time is not None:
            self.performance_time = new_time
        else:
            self.performance_time = -1

        print(f"NEW PERF TIME = {self.performance_time}")
        # TODO : EPSILON POUR LANCER NOUVELLE GENERATION
        if self.generation_process.generation_time < self.performance_time:
            old = self.generation_process.generation_time
            self.generation_process.update_generation_time(self.performance_time)
            print(f"CHANGED PERF TIME = {old} --> {self.generation_process.generation_time}")

    def inc_performance_time(self, increment: Optional[int] = None):
        old_time = self.performance_time

        new_time: Optional[int] = None

        if increment is not None:
            if old_time > -1:
                new_time = old_time + increment
            else:
                new_time = increment

        self.update_performance_time(new_time=new_time)

    # TODO: This should live in Prospector, not GenerationScheduler (may vary between different Prospectors)
    def _use_intervals(self):
        return self.prospector.model.label_type is not None and self.prospector.model.label_type.use_intervals \
               and len(self.authorized_transformations) > 0 and self.authorized_transformations != [0]

    def fonction_test(self):
        return self.prospector.navigator.history_and_taboos

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
        # FIXME[MergeState]: A[X], B[], C[], D[], E[]
        transform: Optional[Transform] = self.current_transformation_memory
        if transform is not None:
            self.prospector.l_encode_with_transform(transform)

    def decode_memory_with_current_transform(self):
        # FIXME[MergeState]: A[X], B[], C[], D[], E[]
        transform: Optional[Transform] = self.current_transformation_memory
        if transform is not None:
            self.prospector.l_decode_with_transform(transform)
