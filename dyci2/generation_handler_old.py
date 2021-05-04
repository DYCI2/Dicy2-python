from generation_handler_new import GenerationHandlerNew


class GenerationHandlerOld(GenerationHandlerNew):
    """ The class **GenerationHandler** introduces time management and planning for interactive applications and adds a
    pool of query, concurrency (e.g. processing of concurrent queries), the use of an execution trace etc. to the class
    :class:`~Generator.Generator`.
    More details in **Nika, Bouche, Bresson, Chemillier, Assayag, "Guided improvisation as dynamic calls to an offline
    model", in Proceedings of Sound and Music Computing conference 2015**
    (https://hal.archives-ouvertes.fr/hal-01184642/document), describing the first prototype of "improvisation handler".

    The key differences between :class:`~Generator.Generator` and :class:`~Generator.GenerationHandler` are:
        * :meth:`Generator.receive_query` / :meth:`GenerationHandler.receive_query`
        * :meth:`Generator.process_query` / :meth:`GenerationHandler.process_query`

    :param generation_trace: Whole output: current state of the sequence generated from the beginning
    (:meth:`GenerationHandler.start`).
    :type generation_trace: list

    :param current_performance_time: Current time of the performance, see below.
    :type current_performance_time: dict
    :param current_performance_time["event"]: Time expressed in events.
    :type current_performance_time["event"]: int
    :param current_performance_time["ms"]: Time expressed in ms.
    :type current_performance_time["ms"]: int
    :param current_performance_time["last_update_event_in_ms"]: Date when the last timing information was received.
    :type current_performance_time["last_update_event_in_ms"]: int


    :param current_generation_time: Current time of the generation, see below.
    :type current_generation_time: dict
    :param current_generation_time["event"]: Time expressed in events.
    :type current_generation_time["event"]: int
    :param current_generation_time["ms"]: Time expressed in ms.
    :type current_generation_time["ms"]: int
    :param current_duration_event_ms: If all the events have (more or less) a same duration (e.g. a clock, pulsed music,
    non timed sequences...), this attribute is not None. It is then used to convert events into dates in ms.
    :type current_duration_event_ms: float

    :param query_pool_event: Pool of waiting queries expressed in events.
    :type query_pool_event: list(:class:`~Query.Query`)
    :param query_pool_ms: Pool of waiting queries expressed in ms.
    :type query_pool_ms: list(:class:`~Query.Query`)


    :see also: :mod:`GeneratorBuilder`, automatic instanciation of Generator objects and GenerationHandler objects from
    annotation files.
    :see also: **Tutorial in** :file:`_Tutorials_/Generator_tutorial.py`.


    :Example:

    >>> labels = make_sequence_of_chord_labels(["d m7", "d m7", "g 7", "g 7", "c maj7","c maj7","c# maj7","c# maj7", "d# m7", "d# m7", "g# 7", "g# 7", "c# maj7", "c# maj7"])
    >>> sequence = make_sequence_of_chord_labels(["d m7(1)", "d m7(2)", "g 7(3)", "g 7(4)", "c maj7(5)","c maj7(6)","c# maj7(7)","c# maj7(8)", "d# m7(9)", "d# m7(10)", "g# 7(11)", "g# 7(12)", "c# maj7(13)", "c# maj7(14)"])
    >>>
    >>> print("\\nCreation of a Generation Handler\\nModel type = Factor Oracle\\nSequence: {}\\nLabels: {}".format(sequence, labels))
    >>>
    >>> authorized_intervals = list(range(-6,6))
    >>> generation_handler = GenerationHandlerOld(sequence = sequence, labels = labels, model_type = "FactorOracleNavigator", authorized_tranformations = authorized_intervals, sequence_to_interval_fun = chord_labels_sequence_to_interval)
    >>> generation_handler.memory.avoid_repetitions_mode = 1
    >>> generation_handler.memory.max_continuity = 3
    >>> generation_handler.memory.no_empty_event = False
    >>> generation_handler.start()

    """

    def __init__(self, sequence=[], labels=[], model_navigator="FactorOracleNavigator", equiv=(lambda x, y: x == y),
                 label_type=None, content_type=None, authorized_tranformations=[0], continuity_with_future=[0.0, 1.0]):
        """ Documentation."""
        GenerationHandlerNew.__init__(self, sequence, labels, model_navigator, equiv, label_type, content_type,
                                      authorized_tranformations, continuity_with_future)

        # TODO : REVOIR : CERTAINS ATTRIBUTS NE SERVENT A RIEN

        self.current_performance_time = {"event": -1, "ms": -1, "last_update_event_in_ms": -1}
        self.generation_trace = []
        self.current_generation_time = {"event": -1, "ms": -1}
        self.current_duration_event_ms = 60000 / 60
        # TODO : UTILISER ?
        # self.pulsed = True
        # TODO : A FAIRE (couper runs...)
        self.running_generation = []
        self.query_pool_event = []
        self.query_pool_ms = []

    # TODO : UTILISER ?
    # self.current_scenario = [] #représentation linéaire des query donc on met en tête de liste les somaxismes

    # TODO TRIGGER LE PROCESS DES BONNES QUERIES DANS LA POOL EN FONCTION  --> CF CODE DIMITRI
    def __setattr__(self, name_attr, val_attr):
        object.__setattr__(self, name_attr, val_attr)
        if name_attr == "current_performance_time":
            print("New value of current performance time: {}".format(self.current_performance_time))

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

    # #### Release 09/18 #####

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

    # TODO : POUR L'INSTANT PAS UTILISE DANS LES EXEMPLES, QUERIES PROCESSEES DIRECT
    # TODO : VOIR POUR LES MERGES ET REORDER ETC
    # TODO : ICI MODIFS POUR VRAI IMPROTEK+SOMAX ?
    def receive_query(self, query, print_info=False):
        """

        The key differences between :class:`~Generator.Generator` and :class:`~Generator.GenerationHandler` are:
            * :meth:`Generator.receive_query` / :meth:`GenerationHandler.receive_query`
            * :meth:`Generator.process_query` / :meth:`GenerationHandler.process_query`

        Inserts the received query in the query pool.
        Handles the interaction between (running and/or waiting) queries: merging compatible queries, killing outdated
        queries...
        The queries in the query pool are then run in due time.
        **TODO: for the moment only "append" and immediate processing.**

        :param query:
        :type query: :class:`~Query.Query`

        :Example:

        >>> labels = make_sequence_of_chord_labels(["d m7", "d m7", "g 7", "g 7", "c maj7","c maj7","c# maj7","c# maj7", "d# m7", "d# m7", "g# 7", "g# 7", "c# maj7", "c# maj7"])
        >>> sequence = make_sequence_of_chord_labels(["d m7(1)", "d m7(2)", "g 7(3)", "g 7(4)", "c maj7(5)","c maj7(6)","c# maj7(7)","c# maj7(8)", "d# m7(9)", "d# m7(10)", "g# 7(11)", "g# 7(12)", "c# maj7(13)", "c# maj7(14)"])
        >>>
        >>> print("\\nCreation of a Generation Handler\\nModel type = Factor Oracle\\nSequence: {}\\nLabels: {}".format(sequence, labels))
        >>>
        >>> authorized_intervals = list(range(-6,6))
        >>> generation_handler = GenerationHandlerOld(sequence = sequence, labels = labels, model_type = "FactorOracleNavigator", authorized_tranformations = authorized_intervals, sequence_to_interval_fun = chord_labels_sequence_to_interval)
        >>> generation_handler.memory.avoid_repetitions_mode = 1
        >>> generation_handler.memory.max_continuity = 3
        >>> generation_handler.memory.no_empty_event = False
        >>> generation_handler.start()
        >>>
        >>> scenario = make_sequence_of_chord_labels(["g m7", "g m7", "c 7", "c 7", "f maj7", "f maj7"])
        >>> query= new_temporal_query_sequence_of_events(scenario)
        >>> print("\\n/!\ Receiving and processing a new query: /!\ \\n{}".format(query))
        >>> generation_handler.receive_query(query = query,  print_info = False)
        >>> print("Output of the run: {}".format(generation_handler.current_generation_output))
        >>> print("/!\ Updated buffered improvisation: {} /!\ ".format(generation_handler.generation_trace))
        >>>
        >>> query= new_temporal_query_free_sequence_of_events(length = 3, start_date = 4, start_type = "absolute")
        >>> print("\\n/!\ Receiving and processing a new query: /!\ \\n{}".format(query))
        >>> generation_handler.receive_query(query = query,  print_info = False)
        >>> print("Output of the run: {}".format(generation_handler.current_generation_output))
        >>> print("/!\ Updated buffered improvisation: {} /!\ ".format(generation_handler.generation_trace))

        """

        # IF QUERY POUR MAINTENANT --> PROCESS DIRECT
        if query.scope["unit"] == "event":
            self.query_pool_event.append(query)
            print("LEN QUERY POOL = {}".format(self.query_pool_event))
            self.process_prioritary_query("event", print_info)
            print("LEN QUERY POOL = {}".format(self.query_pool_event))
        elif query.scope["unit"] == "ms":
            self.query_pool_ms.append(query)
            self.process_prioritary_query("ms", print_info)

    # TODO: COMPARER LA PRIORITE DES QUERIES EN EVENT ET EN MS
    def process_prioritary_query(self, unit=None, print_info=False):
        """
        Processes the prioritary query in the query pool.
        """
        if unit:
            if unit == "event":
                try:
                    assert len(self.query_pool_event) > 0
                except AssertionError as exception:
                    print(""" No "event" query to process""", exception)
                else:
                    self.process_query(self.query_pool_event.pop(0), print_info)
            elif unit == "ms":
                try:
                    assert len(self.query_pool_ms) > 0
                except AssertionError as exception:
                    print(""" No "ms" query to process""", exception)
                else:
                    self.process_query(self.query_pool_ms.pop(0), print_info)
        # TODO : Dessous: temporaire
        else:
            try:
                assert len(self.query_pool_event) > 0
            except AssertionError as exception:
                print(""" No "event" query to process""", exception)
            else:
                self.process_query(self.query_pool_event.pop(0), print_info)

    def process_query(self, query, print_info=False):
        """
        The key differences between :class:`~Generator.Generator` and :class:`~Generator.GenerationHandler` are:
            * :meth:`Generator.receive_query` / :meth:`GenerationHandler.receive_query`
            * :meth:`Generator.process_query` / :meth:`GenerationHandler.process_query`

        This methods takes time into account: in addition to what :meth:`Generator.process_query` does,
        it compares the attribute :attr:`Query.start` of the query and :attr:`self.current_performance_time` to call
        :meth:`Navigator.go_to_anterior_state_using_execution_trace` if needed.
        This way, it ensures consistency at tiling time when rewriting previously generated anticipations.
        As in :meth:`Generator.process_query` the output of this query is stored in :attr:`self.current_generation_output`.
        In addition it is inserted at the right place in the whole output history :attr:`self.generation_trace`.

        :param query:
        :type query: :class:`~Query.Query`

        :return: query.start["date"] (converted to "absolute" value)
        :rtype: int
        """

        print("\n--------------------")
        # print("current navigation index:")
        # print(self.memory.current_position_in_sequence)
        print("""current_performance_time: {}""".format(self.current_performance_time["event"]))
        print("""current_generation_time: {}""".format(self.current_generation_time["event"]))

        # print("""self.memory.execution_trace:""")
        # print(self.memory.execution_trace)

        # TODO
        if len(self.running_generation) < 0:
            print("NO PROCESS RUNNING")
        else:
            print(" !!!!! Already {} processes running !!!!!".format((self.running_generation)))

        self.running_generation.append("Proc")

        if self.initial_query or (
                self.current_performance_time["event"] >= 0 and self.current_performance_time["ms"] >= 0) or \
                query.start["type"] == "absolute":
            print("PROCESS QUERY")
            print(query)
            print("Now, {} processes running".format((self.running_generation)))
            if query.start["type"] == "relative":
                # print("QUERY RELATIVE")
                # print(query)
                query.relative_to_absolute(current_performance_time_event=self.current_performance_time["event"],
                                           current_performance_time_ms=self.current_performance_time["ms"])
                # #### Release 09/18 #####
                if query.start["unit"] == "event" and query.start["date"] < self.current_performance_time["event"]:
                    query.start["date"] = self.current_performance_time["event"]
                if query.start["unit"] == "ms" and query.start["date"] < self.current_performance_time["ms"]:
                    query.start["date"] = self.current_performance_time["ms"]
                # #### Release 09/18 #####
                print("QUERY ABSOLUTE")
                print(query)

            if query.start["unit"] == "event":
                if self.initial_query:
                    self.current_performance_time["event"] = 0
                index_for_generation = query.start["date"]
                print("index_for_generation: {}".format(index_for_generation))
                if index_for_generation > 0 and index_for_generation < self.current_generation_time["event"]:
                    # print("PROCESS 0.0")
                    print("""USING EXECUTION TRACE : index_for_generation = {} / self.current_generation_time["event"] 
                          = {}""".format(index_for_generation, self.current_generation_time["event"]))
                    self.memory.go_to_anterior_state_using_execution_trace(index_for_generation - 1)

            elif query.start["unit"] == "ms":
                if query.start["date"] >= self.current_generation_time["ms"]:
                    index_for_generation = len(self.generation_trace)
                else:
                    index_for_generation = self.index_previously_generated_event_date_ms(query.date["ms"])
                    self.memory.go_to_anterior_state_using_execution_trace(index_for_generation - 1)

            self.memory.current_navigation_index = index_for_generation - 1
            GenerationHandlerNew.process_query(self, query, print_info=print_info)

            l = 0
            if not self.current_generation_output is None:
                print("self.current_generation_output {}".format(self.current_generation_output))
                l = len(self.current_generation_output)
                # print("length output = {}".format(l))
                k = 0
                while k < l and not self.current_generation_output[k] is None:
                    k += 1
                l = k
                print("corrected length output = {}".format(l))

            old_gen_time = self.current_generation_time["event"]

            if query.start["unit"] == "event":
                if index_for_generation > len(self.generation_trace):
                    for i in range(len(self.generation_trace), index_for_generation):
                        self.generation_trace.append(None)

                for i in range(0, l):
                    if i + index_for_generation < len(self.generation_trace):
                        self.generation_trace[index_for_generation + i] = self.current_generation_output[i]
                    else:
                        self.generation_trace.append(self.current_generation_output[i])

                self.generation_trace = self.generation_trace[:(index_for_generation + l)]
                self.current_generation_time["event"] = index_for_generation + l
                self.current_generation_time["ms"] = self.estimation_date_ms_of_event(
                    self.current_generation_time["event"])

            if query.start["unit"] == "ms":

                for i in range(0, l):
                    if index_for_generation + i < len(self.generation_trace):
                        self.generation_trace[index_for_generation + i] = self.current_generation_output[i]
                    else:
                        self.generation_trace.append(self.current_generation_output[i])
                self.generation_trace = self.generation_trace[:index_for_generation + l]
                self.current_generation_time["ms"] = index_for_generation + l
                self.current_generation_time["event"] = self.estimation_date_event_of_ms(
                    self.current_generation_time["ms"])

            print("""generation time: {} --> {}""".format(old_gen_time, self.current_generation_time["event"]))

        self.running_generation.pop()
        return query.start["date"]

    def estimation_date_ms_of_event(self, num_event):
        """
        If all the events have (more or less) a same duration (e.g. a clock, pulsed music, non timed sequences...),
        :attr:`self.current_duration_event_ms` is not None. It is then used to convert indexes of events into dates in ms.

        :param num_event: index of event to convert
        :type num_event: int

        :return: estimated corresponding date in ms (or None)
        :rtype: int
        """

        if not self.current_duration_event_ms is None:
            return (num_event - self.current_performance_time["event"]) * (self.current_duration_event_ms) - (
                    self.current_performance_time["ms"] - self.current_performance_time["last_update_event_in_ms"])
        else:
            return None

    def estimation_date_event_of_ms(self, date_ms):
        """
        If all the events have (more or less) a same duration (e.g. a clock, pulsed music, non timed sequences...),
        :attr:`self.current_duration_event_ms` is not None. It is then used to convert dates in ms into indexes of events.

        :param date_ms: date in ms to convert
        :type date_ms: int

        :return: estimated corresponding index of event (or None)
        :rtype: int
        """
        if not self.current_duration_event_ms is None:
            last_known_event = self.current_performance_time["event"]
            last_known_event_date = self.current_performance_time["last_update_event_in_ms"]
            delta = date_ms - last_known_event_date
            return last_known_event + int(delta) / int(self.current_duration_event_ms)
        else:
            return None

    # TODO
    def index_previously_generated_event_date_ms(self, date_query):

        # TODO : voir dans code lisp DIMITRI : additionner les durées potentiellement différentes des éléments générés.
        # ET QUE FAIRE SI TOMBE AU MILIEU D'UN EVENT ??? LE TRONQUER ???? REPRENDRE AU DEBUT
        duration_from_start = 0
        i = 0

        while i < len(self.generation_trace) and duration_from_start < date_query:
            # VRAI TRUC A FAIRE QUAND ON AURA DES EVENTS / CONTENTS AVEC DUREE !!
            # duration_from_start += duration(self.generation_trace[i])
            duration_from_start += self.current_duration_event_ms
            i += 1

        if i < len(self.generation_trace):
            return i - 1
        else:
            return i
