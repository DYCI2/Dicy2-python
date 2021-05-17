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

# TODO
#  STOCKER LES TRANSFORMATIONS POUR POUVOIR ENVOYER AUSSI AU FORMAT "index ou valeur du contenu dans son etat original"
#  "transformation à appliquer"
#  QUERY DE TYPE ADD OR REPLACE
#  - go_to_anterior_state_using_execution_trace

# TODO POUR VRAIMENT MIXER SOMAX ET IMPROTEK
#  SLOT DE GENERATION HANDLER (ou de nouvelle classe qui en hérite) : self.alternative_generation_output = None
#  C'EST POUR GUIDAGE DUR ! STOCKER DIFFERENTES POSSIBILITES DONNEES PAR SCENARIO
#  ... SUR TOUTE UNE QUERY ? ENORME SI TOUTE LA GRILLE... SUR UNE PHASE ?? Bof...

# TODO FAIRE UNE CLASSE "SEQUENCETRANSFORMER" QUI CONTIENT TOUS LES PARAMETRES ET METHODES LIES A INTERVALS...
#  A CETTE OCCASION MODIFIER CE QUI EST LA EN DUR, A SAVOIR QUE LE "AUTHORIZED TRANSFO" S'EXPRIMENT UNIQUEMENT EN INT.

# TODO DOCUMENTER VRAIMENT LES SLOTS DE LA CLASSE ET DONC CERTAINS ARGUMENTS DE METHODES ET FONCTIONS
# TODO S'OCCUPER D'UNE CLASSE CONTENT !!


from typing import Optional, Callable, Tuple

# TODO : SUPPRIMER DANS LA DOC LES FONCTIONS "EQUIV-MOD..." "SEQUENCE TO INTERVAL..."

from dyci2.query import *
from dyci2.transforms import *
# TODO 2021 : Initially default argument for Generator was (lambda x, y: x == y) --> pb with pickle
# TODO 2021 : (because not serialized ?) --> TODO "Abstract Equiv class" to pass objects and not lambda ?
from generation_process import GenerationProcess
from memory import MemoryEvent
from prospector import Prospector, implemented_model_navigator_classes
from utils import format_list_as_list_of_strings, DontKnow


def basic_equiv(x, y):
    return x == y


# TODO[E]: Remove noinspection
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

    :param model_navigator:
    :type model_navigator: str

    #:param memory: "Model navigator" inheriting from (a subclass of) :class:`~Model.Model` and (a subclass of)
    :class:`~Navigator.Navigator`.
    :type prospector: cf. :mod:`ModelNavigator` and :mod:`MetaModelNavigator`

    #:param initial_query:
    :type initial_query: bool
    #:param current_generation_query:
    :type current_generation_query: :class:`~Query.Query`

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

    def __init__(self, sequence=(), labels=(), model_navigator="FactorOracleNavigator", equiv=(lambda x, y: x == y),
                 label_type=None, content_type=None, authorized_tranformations=(0,), continuity_with_future=(0.0, 1.0)):
        # FIXME[MergeState]: A[], B[], C[], D[]
        try:
            implemented_model_navigator_classes
        except NameError:
            print("No model navigator available. Please load prospector.py")
            return
        else:
            try:
                assert model_navigator in implemented_model_navigator_classes.keys()
            except AssertionError as exception:
                print("Unknown model navigator. \n Available classes: {}".format(
                    implemented_model_navigator_classes.keys()), exception)
                return
            else:
                if equiv is None:
                    equiv = basic_equiv
                self.model_navigator = model_navigator
                self.prospector: Prospector = implemented_model_navigator_classes[self.model_navigator](
                    sequence=sequence,
                    labels=labels,
                    label_type=label_type,
                    content_type=content_type,
                    equiv=equiv)

        self.content_type = content_type
        self.initial_query = True  # TODO[B] Rename: `restart_generation`? `generation_initialized`?
        self.current_generation_output = []
        self.transfo_current_generation_output = []

        self.authorized_transformations = authorized_tranformations
        # TODO: Rather NoTransform?
        self.current_transformation_memory = None
        self.continuity_with_future = continuity_with_future
        self.performance_time: int = 0

        self.generation_process: GenerationProcess = GenerationProcess()

    def __setattr__(self, name_attr, val_attr):
        object.__setattr__(self, name_attr, val_attr)
        if name_attr == "current_performance_time":
            print("New value of current performance time: {}".format(self.current_performance_time))

    def learn_event(self, event: MemoryEvent) -> None:
        # FIXME[MergeState]: A[x], B[], C[], D[]
        """ Learn a new event in the memory (model navigator)."""
        self.prospector.learn_event(event=event)

    def learn_sequence(self, sequence: List[MemoryEvent]) -> None:
        # FIXME[MergeState]: A[x], B[], C[], D[]
        """ Learn a new sequence in the memory (model navigator)."""
        self.prospector.learn_sequence(sequence=sequence)

    def process_query(self, query: Query, performance_time: int):
        # TODO[B]: This entire function is basically a long list of side effects to distribute all over the system given
        #   various cases. A better solution would be to clean most of these up and pass along. The only important thing
        #   here is go_to_anterior_state_using_execution_trace
        """ raises: """
        print("\n--------------------")
        print(f"current_performance_time: {performance_time}")
        print(f"current_generation_time: {self.generation_process.generation_time}")

        if not self.initial_query and performance_time < 0 and query.time_mode == TimeMode.RELATIVE:
            # TODO[JEROME]: Throw error? What is the purpose of this invariant? For reference: old statement was
            #  if self.initial_query or performance_time >= 0 or query.time_mode == TimeMode.ABSOLUTE
            return query.start_date

        # TODO[B] If invariant above is useless or if we can handle this elsewhere, handle query.to_absolute in scheduler instead
        print("PROCESS QUERY\n", query)
        if query.time_mode == TimeMode.RELATIVE:
            query.to_absolute(performance_time)
            print("QUERY ABSOLUTE\n", query)

        if self.initial_query:
            self.performance_time = 0
        generation_index: int = query.start_date
        print(f"generation_index: {generation_index}")
        if 0 < generation_index < self.generation_process.generation_time:
            print(f"USING EXECUTION TRACE : generation_index = {generation_index} : "
                  f"generation_time = {self.generation_process.generation_time}")
            self.prospector.go_to_anterior_state_using_execution_trace(generation_index - 1)

        # TODO[B] UNSOLVED! Massive side-effect. Exactly what is current_navigation_index?
        self.prospector.current_navigation_index = generation_index - 1
        # TODO[B]: UNSOLVED! generator_process_query should return output, not store it in current_generation_output
        self.generator_process_query(query)

        print(f"self.current_generation_output {self.current_generation_output}")
        self.generation_process.add_output(generation_index, self.current_generation_output)

        # TODO[B] Can this return generation_index instead? Or is query.start_date changed somewhere along the line?
        return query.start_date

    def _process_query(self, query: Query):
        print("**********************************\nPROCESS QUERY: QUERY = \n**********************************", query)
        print("**********************************\nGENERATION MATCHING QUERY: QUERY = \n**********************", query)
        output: Optional[DontKnow]
        # TODO[B]: UNSOLVED! probably unnecessary side-effect but discuss w/ Jerome before removing
        self.transfo_current_generation_output = []
        if isinstance(query, FreeQuery):
            print("GENERATION MATCHING QUERY FREE ...")
            output = self.free_generation(num_events=query.num_events, init=self.initial_query,
                                          print_info=query.print_info)
            print("... GENERATION MATCHING QUERY FREE OK")
            self.transfo_current_generation_output = [self.current_transformation_memory] * len(output)

        elif isinstance(query, LabelQuery) and len(query.labels) == 1:
            print("GENERATION MATCHING QUERY LABEL ...")
            output = result = self.simply_guided_generation(required_labels=query.labels, init=self.initial_query,
                                                            print_info=query.print_info)
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
            self.initial_query = False

    def free_generation(self, num_events: int, new_max_continuity: Optional[DontKnow] = None,
                        forward_context_length_min: int = 0, init: bool = False, equiv: Callable = None,
                        print_info: bool = False) -> List[Optional[DontKnow]]:
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
        # TODO[B] Discuss this with Jerome: how does encoding/decoding impact the actual memory? Is the return value
        #  here (`sequence`) really referring to something else than what's decoded in line below?
        self.encode_memory_with_current_transfo()
        # TODO[B]: Is it possible to get rid of these pre_ functions? Can we pass these parameters
        #  along all the way instead?
        print_info, equiv = self.prospector.l_pre_free_navigation(equiv, new_max_continuity, init)
        sequence: List[Optional[DontKnow]] = []
        generated_indices: List[Optional[int]] = []
        for i in range(num_events):
            generated_indices.append(self.prospector.r_free_navigation_one_step(i, forward_context_length_min,
                                                                                equiv, print_info))
        for generated_index in generated_indices:
            if generated_index is None:
                sequence.append(None)
            else:
                # TODO[B] Handle with proper location of Memory
                sequence.append(self.prospector.model.sequence[generated_index])

        self.decode_memory_with_current_transfo()
        return sequence

    def simply_guided_generation(self, required_labels: List[Label],
                                 new_max_continuity: Optional[Tuple[float, float]] = None,
                                 forward_context_length_min: int = 0, init: bool = False,
                                 equiv: Optional[Callable] = None, print_info: bool = False, shift_index: int = 0,
                                 break_when_none: bool = False):
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
        self.encode_memory_with_current_transfo()

        equiv = self.prospector.l_pre_guided_navigation(required_labels, equiv, new_max_continuity, init)

        generated_indices: List[Optional[int]] = []
        for i in range(len(required_labels)):
            s: Optional[int] = self.prospector.simply_guided_navigation_one_step(required_labels[i], new_max_continuity,
                                                                                 forward_context_length_min, equiv,
                                                                                 print_info, shift_index=i + shift_index)

            if break_when_none and s is None:
                break
            else:
                generated_indices.append(s)

        # TODO[B] Handle with proper location of Memory
        sequence: List[Optional[DontKnow]] = [self.prospector.model.sequence[i] if i is not None else None
                                              for i in generated_indices]

        self.decode_memory_with_current_transfo()
        return sequence

    def scenario_based_generation(self, list_of_labels: List[Label], print_info: bool = False):
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
        # TODO[C] Way too much manipulation and access of `memory` (i.e. Generator) here
        i: int = 0
        generated_sequence = []
        print("SCENARIO BASED GENERATION 0")
        while i < len(list_of_labels):
            # TODO[B]: Handle so that it doesn't return Optional[List[...
            seq: Optional[List[Optional[DontKnow]]]
            seq = self.handle_scenario_based_generation_one_phase(list_of_labels=list_of_labels[i::],
                                                                  print_info=print_info, shift_index=i)

            if seq is not None and len(seq) > 0:
                l = len(seq)
                generated_sequence += seq
                i += l
            else:
                if self.prospector.l_get_no_empty_event() and self.prospector.l_get_position_in_sequence() < self.prospector.l_get_index_last_state():
                    print("NO EMPTY EVENT")
                    next_index: int = self.prospector.l_get_position_in_sequence() + 1
                    generated_sequence.append(self.prospector.l_get_sequence_nonmutable()[next_index])
                    self.prospector.l_set_position_in_sequence(next_index)

                # TODO : + TRANSFORMATION POUR TRANSPO SI NECESSAIRE
                else:
                    print("EMPTY EVENT")
                    generated_sequence.append(None)
                    ###### RELEASE
                    self.prospector.l_set_position_in_sequence(0)
                ###### RELEASE
                i += 1
        # print("SCENARIO BASED GENERATION 1.{}.3".format(i))
        return generated_sequence

    # TODO : PAS OPTIMAL DU TOUT D'ENCODER DECODER A CHAQUE ETAPE !!!!!!!!
    def handle_scenario_based_generation_one_phase(self, list_of_labels: List[Label], print_info: bool = False,
                                                   shift_index: int = 0):
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
        generated_sequence = []
        # s = Recherche prefix (current_scenario) - > Sort 1 ### TODO
        # print("self.memory.history_and_taboos = {}".format(self.memory.history_and_taboos))
        # Remember state at idx 0 is None
        authorized_indexes = self.filter_using_history_and_taboos(
            list(range(len(self.prospector.l_get_labels_nonmutable()))))

        self.decode_memory_with_current_transfo()
        self.current_transformation_memory = None

        if self.prospector.model.label_type is not None and self._use_intervals():
            func_intervals_to_labels: Optional[Callable] = \
                self.prospector.model.label_type.make_sequence_of_intervals_from_sequence_of_labels
            equiv_mod_interval: Optional[Callable] = self.prospector.model.label_type.equiv_mod_interval
        else:
            func_intervals_to_labels = None
            equiv_mod_interval = None

        s, t, length_selected_prefix = self.prospector.scenario_based_generation(self._use_intervals(), list_of_labels,
                                                                                 self.continuity_with_future,
                                                                                 authorized_indexes,
                                                                                 self.authorized_transformations,
                                                                                 func_intervals_to_labels,
                                                                                 equiv_mod_interval,
                                                                                 self.prospector.model.equiv)

        if s is not None:
            print("SCENARIO BASED ONE PHASE SETS POSITION: {}".format(s))
            self.prospector.l_set_position_in_sequence(s)
            print("current_position_in_sequence: {}".format(s))
            # PLUS BESOIN CAR FAIT TOUT SEUL
            # self.memory.history_and_taboos[s] += 1
            if t != 0:
                self.current_transformation_memory = TransposeTransform(t)
                # print(self.memory.sequence[s])
                # TODO FAIRE MIEUX:
                if self.content_type:
                    s_content = self.current_transformation_memory.encode(self.prospector.l_get_sequence_maybemutable()[s])
                else:
                    s_content = self.prospector.l_get_sequence_maybemutable()[s]

                if print_info:
                    print("{} NEW STARTING POINT {} (orig. {}): {}\nlength future = {} - FROM NOW {}"
                          .format(shift_index,
                                  self.current_transformation_memory.encode(self.prospector.l_get_labels_maybemutable()[s]),
                                  self.prospector.l_get_labels_nonmutable()[s],
                                  self.prospector.l_get_position_in_sequence(),
                                  length_selected_prefix,
                                  self.current_transformation_memory))
            else:
                s_content = self.prospector.l_get_sequence_maybemutable()[s]
                if print_info:
                    print("{} NEW STARTING POINT {}: {}\nlength future = {} - FROM NOW No transformation of the memory"
                          .format(shift_index, self.prospector.l_get_labels_nonmutable()[s],
                                  self.prospector.l_get_position_in_sequence(), length_selected_prefix))

            # print("SCENARIO ONE PHASE 4")

            generated_sequence.append(s_content)
            self.transfo_current_generation_output.append(self.current_transformation_memory)
            # PLUS BESOIN CAR FAIT TOUT SEUL
            # self.memory.current_continuity = 0
            # Navigation simple current_scenario jusqua plus rien- > sort l --> faire
            self.encode_memory_with_current_transfo()
            aux = self.prospector.l_get_no_empty_event()
            # In order to begin a new navigation phase when this method returns a "None" event
            self.prospector.l_set_no_empty_event(False)
            seq = self.simply_guided_generation(required_labels=list_of_labels[1::], init=False,
                                                print_info=print_info,
                                                shift_index=len(self.current_generation_query.handle) - len(
                                                    list_of_labels) + 1,
                                                break_when_none=True)
            self.prospector.l_set_no_empty_event(aux)
            # self.decode_memory_with_current_transfo()
            # print("SCENARIO ONE PHASE 5")
            i = 0
            while i < len(seq) and (not (seq[i] is None)):
                generated_sequence.append(seq[i])
                self.transfo_current_generation_output.append(self.current_transformation_memory)
                i += 1

            # self.current_transformation_memory = None
            # print("END PREXIX")
            # print(self.current_transformation_memory)

            # print("---------END handle_scenario_based")
            # return generated_sequence ####  # TODO : + TRANSFORMATION POUR TRANSPO SI NECESSAIRE
            print("---------END handle_scenario_based ->> Return {}".format(generated_sequence))
            return generated_sequence  ####  # TODO : + TRANSFORMATION POUR TRANSPO SI NECESSAIRE

    ################################################################################################################
    #   LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY LEGACY   #
    ################################################################################################################

    def start(self):
        """ Sets :attr:`self.current_performance_time` to 0."""
        self.current_performance_time["event"] = 0
        self.current_performance_time["ms"] = 0
        self.current_performance_time["last_update_event_in_ms"] = 0

    # TODO : EPSILON POUR LANCER NOUVELLE GENERATION
    def update_performance_time(self, date_event=None, date_ms=None):
        # print(self.current_performance_time)
        if not date_event is None:
            self.current_performance_time["event"] = date_event
            if not date_ms is None:
                self.current_performance_time["ms"] = date_ms
                self.current_performance_time["last_update_event_in_ms"] = date_ms
            else:
                self.current_performance_time["ms"] = -1
                self.current_performance_time["last_update_event_in_ms"] = -1
        elif not date_ms is None:
            self.current_performance_time["ms"] = date_ms
            if date_event is None:
                self.current_performance_time["event"] = -1

        print("NEW PERF TIME = {}".format(self.current_performance_time))
        # TODO : EPSILON POUR LANCER NOUVELLE GENERATION
        # #### Release 09/18 #####
        if self.current_generation_time["event"] < self.current_performance_time["event"]:
            old = self.current_generation_time["event"]
            self.current_generation_time["event"] = self.current_performance_time["event"]
            print("CHANGED PERF TIME = {} --> {}".format(old, self.current_generation_time["event"]))

        if self.current_generation_time["ms"] < self.current_performance_time["ms"]:
            old = self.current_generation_time["ms"]
            self.current_generation_time["ms"] = self.current_performance_time["ms"]
        # print("CHANGED PERF TIME = {} --> ".format(old, self.current_generation_time["event"]))

    def inc_performance_time(self, inc_event=None, inc_ms=None):
        old_event = self.current_performance_time["event"]
        old_ms = self.current_performance_time["ms"]

        new_event = None
        new_ms = None

        if not inc_event is None:
            if old_event > -1:
                new_event = old_event + inc_event
            else:
                new_event = inc_event

        if not inc_ms is None:
            if old_ms > -1:
                new_ms = old_ms + inc_ms
            else:
                new_ms = inc_ms

        self.update_performance_time(date_event=new_event, date_ms=new_ms)

    def _use_intervals(self):
        return self.prospector.model.label_type is not None and self.prospector.model.label_type.use_intervals \
               and len(self.authorized_transformations) > 0 and self.authorized_transformations != [0]

    def filter_using_history_and_taboos(self, list_of_indexes):
        return self.prospector.navigator.filter_using_history_and_taboos(list_of_indexes)

    def fonction_test(self):
        return self.prospector.navigator.history_and_taboos

    # TODO : [NONE] au début : UNIQUEMENT POUR ORACLE ! PAS GENERIQUE !
    # TODO : A MODIFIER QUAND LES CONTENTS DANS SEQUENCE AURONT UN TYPE !
    def encode_memory_with_transfo(self, transform):
        """
        Apply the transformation given in argument to :attr:`self.memory.sequence and :attr:`self.memory.label`.

        :param transform:
        :type transform: cf. :mod:`Transforms`

        """
        # print("*********************************************************")
        # print(self.memory.sequence[1::])
        # TODO : Faire mieux
        if self.content_type:
            self.prospector.l_set_sequence([None] + transform.encode_sequence(self.prospector.l_get_sequence_nonmutable()[1::]))
        self.prospector.l_set_labels([None] + transform.encode_sequence(self.prospector.l_get_labels_nonmutable()[1::]))

    # TODO : [NONE] au début : UNIQUEMENT POUR ORACLE ! PAS GENERIQUE !
    # TODO : SERA CERTAINEMENT A MODIFIER QUAND LES CONTENTS DANS SEQUENCE AURONT UN TYPE !
    def decode_memory_with_transfo(self, transform):
        """
        Apply the reciprocal transformation of the transformation given in argument to :attr:`self.memory.sequence` and
        :attr:`self.memory.label`.

        :param transform:
        :type transform: cf. :mod:`Transforms`

        """
        # TODO : Faire mieux
        if self.content_type:
            self.prospector.l_set_sequence([None] + transform.decode_sequence(self.prospector.l_get_sequence_nonmutable()[1::]))
        self.prospector.l_set_labels([None] + transform.decode_sequence(self.prospector.l_get_labels_nonmutable()[1::]))

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

    def encode_memory_with_current_transfo(self):
        # FIXME[MergeState]: A[X], B[], C[], D[], E[]
        current_transfo = self.current_transformation_memory
        if current_transfo is not None:
            self.encode_memory_with_transfo(current_transfo)

    def decode_memory_with_current_transfo(self):
        # FIXME[MergeState]: A[X], B[], C[], D[], E[]
        current_transfo = self.current_transformation_memory
        if current_transfo is not None:
            self.decode_memory_with_transfo(current_transfo)
