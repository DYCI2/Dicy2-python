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
from typing import Optional, Callable, Type, List

from dyci2.candidate_selector import TempCandidateSelector
from dyci2.dyci2_label import Dyci2Label
from dyci2.dyci2_time import Dyci2Timepoint, TimeMode
from dyci2.dyci2_prospector import Dyci2Prospector
from dyci2.generation_process import GenerationProcess
from dyci2.generator import Dyci2Generator
from dyci2.parameter import Parametric
from dyci2.utils import format_list_as_list_of_strings
from merge.main.candidate import Candidate
from merge.main.corpus import Corpus
from merge.main.corpus_event import CorpusEvent
from merge.main.generation_scheduler import GenerationScheduler
from merge.main.jury import Jury
from merge.main.query import Query


# TODO : SUPPRIMER DANS LA DOC LES FONCTIONS "EQUIV-MOD..." "SEQUENCE TO INTERVAL..."


class Dyci2GenerationScheduler(GenerationScheduler, Parametric):
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

    # :param memory: "Model navigator" inheriting from (a subclass of) :class:`~Model.Model` and (a subclass of)
    # :class:`~Navigator.Navigator`.
    # :type prospector: cf. :mod:`ModelNavigator` and :mod:`MetaModelNavigator`

    # :param initial_query:
    # :type initial_query: bool
    # :param current_generation_query:
    # :type current_generation_query: :class:`~Query.Query`

    # :param current_generation_output:
    # :type current_generation_output: list
    # :param transfo_current_generation_output:
    # :type transfo_current_generation_output: list

    # :param continuity_with_future:
    # :type continuity_with_future: list
    #
    # :param current_transformation_memory:
    # :type active_transform: cf. :mod:`Transforms`
    # :param authorized_tranformations:
    # :type authorized_tranformations: list(int)
    # :param sequence_to_interval_fun:
    # :type sequence_to_interval_fun: function
    # :param equiv_mod_interval:
    # :type equiv_mod_interval: function


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

    def __init__(self,
                 prospector: Dyci2Prospector,
                 jury_type: Type[Jury] = TempCandidateSelector,
                 authorized_tranformations=(0,)):
        self.generator: Dyci2Generator = Dyci2Generator(prospector=prospector,
                                                        jury_type=jury_type,
                                                        authorized_transforms=authorized_tranformations)
        self.logger = logging.getLogger(__name__)
        self._performance_time: int = 0
        self._running: bool = False
        self.generation_process: GenerationProcess = GenerationProcess()

    ################################################################################################################
    # PUBLIC: INHERITED METHODS
    ################################################################################################################

    def process_query(self, query: Query, **kwargs) -> None:
        """ raises: RuntimeError if receiving a relative query as the first query. """
        self.logger.debug("\n--------------------")
        self.logger.debug(f"current_performance_time: {self._performance_time}")
        self.logger.debug(f"current_generation_time: {self.generation_process.generation_time}")

        if not isinstance(query.time, Dyci2Timepoint):
            raise RuntimeError(f"Can only handle queries with time format '{Dyci2Timepoint.__name__}'")

        # TODO[Jerome]: Is this really a good strategy / relevant?
        if self.generator.initialized and not self._running and query.time.time_mode == TimeMode.RELATIVE:
            raise RuntimeError("Cannot handle a relative query as the first query")

        self.logger.debug(f"PROCESS QUERY\n {query}")
        if query.time.time_mode == TimeMode.RELATIVE:
            query.time.to_absolute(self._performance_time)
            self.logger.debug(f"QUERY ABSOLUTE\n {query}")

        # TODO: Is this a good idea?
        if not self.generator.initialized:
            self.start()

        generation_index: int = query.time.start_date
        self.logger.debug(f"generation_index: {generation_index}")
        if 0 < generation_index < self.generation_process.generation_time:
            self.logger.debug(f"USING EXECUTION TRACE : generation_index = {generation_index} : "
                              f"generation_time = {self.generation_process.generation_time}")
            self.generator.prospector.rewind_generation(generation_index - 1)

        # TODO[Jerome] UNSOLVED! Isn't this the same as rewind_generation? Looks very similar to line above.
        self.generator.prospector._navigator.current_navigation_index = generation_index - 1

        output: List[Optional[Candidate]] = self.generator.process_query(query)

        self.logger.debug(f"self.current_generation_output {output}")
        self.generation_process.add_output(generation_index, output)

        # TODO: Handle in parser: This value corresponds to GenerationProcess._start_of_last_sequence
        #       use GenerationScheduler.generation_index() to get this value
        # return generation_index

    def update_performance_time(self, time: Dyci2Timepoint) -> None:
        if not self._running:
            return

        if time.time_mode == TimeMode.RELATIVE:
            time.to_absolute(self._performance_time)

        self._performance_time = time.start_date

        print(f"NEW PERF TIME = {self._performance_time}")
        # TODO : EPSILON POUR LANCER NOUVELLE GENERATION
        if self.generation_process.generation_time < self._performance_time:
            old = self.generation_process.generation_time
            self.generation_process.update_generation_time(self._performance_time)
            print(f"CHANGED PERF TIME = {old} --> {self.generation_process.generation_time}")

    def read_memory(self, corpus: Corpus, **kwargs) -> None:
        self.clear()
        self.generator.read_memory(corpus, **kwargs)

    def learn_event(self, event: CorpusEvent, **kwargs) -> None:
        self.generator.learn_event(event, **kwargs)

    def clear(self) -> None:
        self.generator.clear()

    ################################################################################################################
    # PUBLIC: CLASS-SPECIFIC METHODS
    ################################################################################################################

    def start(self) -> None:
        self.clear()
        self._performance_time = 0
        self._running = True

    def set_equiv_function(self, equiv: Callable[[Dyci2Label, Dyci2Label], bool]) -> None:
        self.generator.prospector.set_equiv_function(equiv=equiv)

    def increment_performance_time(self, increment: int = 1) -> None:
        self.update_performance_time(Dyci2Timepoint(start_date=increment, time_mode=TimeMode.RELATIVE))

    def generation_index(self) -> int:
        return self.generation_process.start_index_of_last_sequence()

    @property
    def performance_time(self) -> int:
        return self._performance_time

    ################################################################################################################
    # TODO: CLEAN UP REQUIRED
    #  Formatting/parsing is generally an IO operation that should live in the Agent/IO class than in the logic class
    ################################################################################################################

    def formatted_output_couple_content_transfo(self):
        return [(c.event, c.transform.renderer_info()) for c in self.generation_process.last_sequence()]

    def formatted_output_string(self):
        return format_list_as_list_of_strings(self.current_generation_output)

    def formatted_generation_trace_string(self):
        return format_list_as_list_of_strings(self.generation_trace)
