# -*-coding:Utf-8 -*

#############################################################################
# GenerationHandler_tutorial.py 
# Jérôme Nika, IRCAM STMS Lab
# copyleft 2016 - 2017
#############################################################################


""" 
Generation Handler Tutorial
=============================
Tutorial for the class :class:`~Generator.GenerationHander` defined in :mod:`Generator` (cf. also :mod:`OSCAgent_tutorial`).

"""
import random
import sys
import warnings
from typing import List, Tuple

from candidate import Candidate
from dyci2.factor_oracle_model import FactorOracle
from dyci2.factor_oracle_navigator import FactorOracleNavigator
from dyci2.generation_scheduler import GenerationScheduler
from dyci2.label import from_list_to_labels, ChordLabel
from dyci2.memory import Memory, LabelEvent
from dyci2.query import Query, FreeQuery, TimeMode, LabelQuery


def chord_format(lst: List[Tuple[LabelEvent, int]]):
    return [[e.data(), t] for (e, t) in lst]

def candidate_format(lst: List[Candidate]):
    return [e.event.data() for e in lst]


if __name__ == '__main__':

    # logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    random.seed(0)
    warnings.filterwarnings("ignore")
    original_stdout = sys.stdout
    sys.stdout = None

    list_for_labels = ["d m7", "d m7", "g 7", "g 7", "c maj7", "c maj7", "c# maj7", "c# maj7", "d# m7", "d# m7", "g# 7",
                       "g# 7", "c# maj7", "c# maj7"]
    list_for_sequence = ["d m7(1)", "d m7(2)", "g 7(3)", "g 7(4)", "c maj7(5)", "c maj7(6)", "c# maj7(7)", "c# maj7(8)",
                         "d# m7(9)", "d# m7(10)", "g# 7(11)", "g# 7(12)", "c# maj7(13)", "c# maj7(14)"]
    labels: List[ChordLabel] = from_list_to_labels(labels=list_for_labels, label_type=ChordLabel)
    sequence: List[ChordLabel] = from_list_to_labels(labels=list_for_sequence, label_type=ChordLabel)
    memory: Memory = Memory([LabelEvent(event, label) for (event, label) in zip(sequence, labels)],
                            content_type=LabelEvent, label_type=ChordLabel)

    print("\nCreation of a Generation Handler"
          "\nModel type = Factor Oracle\nSequence: {}\nLabels: {}".format(sequence, labels))

    authorized_intervals = list(range(-6, 6))
    generation_scheduler: GenerationScheduler = GenerationScheduler(memory=memory, model_class=FactorOracle,
                                                                    navigator_class=FactorOracleNavigator,
                                                                    authorized_tranformations=authorized_intervals)
    generation_scheduler.prospector._navigator.avoid_repetitions_mode.set(1)
    generation_scheduler.prospector._navigator.max_continuity.set(3)
    warnings.warn("This is not a settable parameter anymore: no_empty_event")
    # generation_scheduler.prospector.navigator.no_empty_event = False

    generation_scheduler.start()

    print("\n --- Starting simulation of interactions (receiving and processing query)...  --- ")

    list_for_scenario: List[str] = ["g m7", "g m7", "c 7", "c 7", "f maj7", "f maj7"]
    labels_for_scenario: List[ChordLabel] = from_list_to_labels(list_for_scenario, ChordLabel)
    query: LabelQuery = LabelQuery(labels_for_scenario, print_info=False)
    # query = new_temporal_query_sequence_of_events(handle=list_for_scenario, label_type=ChordLabel)
    print("\n/!\ Receiving and processing a new query: /!\ \n{}".format(query))
    generation_scheduler.process_query(query=query)
    print("Output of the run: {}".format(generation_scheduler.generation_process.last_sequence()))

    sys.stdout = original_stdout
    print("With transforfmation: {}".format(chord_format(generation_scheduler.formatted_output_couple_content_transfo())))

    sys.stdout = None

    print(
        "/!\ Updated buffered improvisation: {} /!\ ".format(generation_scheduler.generation_process.generation_trace))



    print("\n --- ... and starting simulation of performance time (beat, 60 BPM) --- ")

    print("\n**NEW PERFORMANCE TIME : BEAT {}**\n**PLAYING CORRESPONDING GENERATED EVENT: {}**".format(
        generation_scheduler.performance_time,
        generation_scheduler.generation_process.generation_trace[generation_scheduler.performance_time]))

    # time.sleep(1)
    # generation_handler.current_performance_time["event"] += 1
    # generation_handler.current_performance_time["ms"] += 1000
    # generation_handler.current_performance_time["last_update_event_in_ms"] = generation_handler.current_performance_time["ms"]
    generation_scheduler.inc_performance_time(increment=1)

    # sys.exit(1)

    print("\n**NEW PERFORMANCE TIME : BEAT {}**\n**PLAYING CORRESPONDING GENERATED EVENT: {}**".format(
        generation_scheduler.performance_time,
        generation_scheduler.generation_process.generation_trace[generation_scheduler.performance_time]))



    query: Query = FreeQuery(3, 4, TimeMode.ABSOLUTE, print_info=False)
    # query = new_temporal_query_free_sequence_of_events(length=3, start_date=4, start_type="absolute")
    print("\n/!\ Receiving and processing a new query: /!\ \n{}".format(query))
    generation_scheduler.process_query(query=query)
    print("Output of the run: {}".format(generation_scheduler.generation_process.last_sequence()))

    sys.stdout = original_stdout

    print("With transforfmation: {}".format(chord_format(generation_scheduler.formatted_output_couple_content_transfo())))

    sys.stdout = None

    print(
        "/!\ Updated buffered improvisation: {} /!\ ".format(generation_scheduler.generation_process.generation_trace))

    for i in range(0, 2):
        # time.sleep(1)
        # generation_handler.current_performance_time["event"] += 1
        # generation_handler.current_performance_time["ms"] += 1000
        # generation_handler.current_performance_time["last_update_event_in_ms"] = generation_handler.current_performance_time["ms"]
        generation_scheduler.inc_performance_time(increment=1)
        print("\n**NEW PERFORMANCE TIME : BEAT {}**\n**PLAYING CORRESPONDING GENERATED EVENT: {}**".format(
            generation_scheduler.performance_time,
            generation_scheduler.generation_process.generation_trace[generation_scheduler.performance_time]))

    # TODO : POURQUOI NE MARCHE PAS AVEC TRANSPO MIN DE -2 ????
    # scenario = make_sequence_of_chord_labels(["a maj7", "a maj7"])
    list_for_scenario = ["d m7", "d m7", "d m7"]
    labels_for_scenario: List[ChordLabel] = from_list_to_labels(list_for_scenario, ChordLabel)
    query: LabelQuery = LabelQuery(labels_for_scenario, start_date=2, time_mode=TimeMode.RELATIVE, print_info=False)
    print("\n/!\ Receiving and processing a new query: /!\ \n{}".format(query))
    generation_scheduler.process_query(query=query)

    sys.stdout = original_stdout

    print("Output of the run: {}".format(candidate_format(generation_scheduler.generation_process.last_sequence())))



    print(
        "/!\ Updated buffered improvisation: {} /!\ ".format(candidate_format(generation_scheduler.generation_process.generation_trace)))

    sys.stdout = None
    sys.exit(0)

    for i in range(0, 4):
        # time.sleep(1)
        # generation_handler.current_performance_time["event"] += 1
        # generation_handler.current_performance_time["ms"] += 1000
        # generation_handler.current_performance_time["last_update_event_in_ms"] = generation_handler.current_performance_time["ms"]
        generation_scheduler.increment_performance_time(increment=1)
        print("\n**NEW PERFORMANCE TIME : BEAT {}**\n**PLAYING CORRESPONDING GENERATED EVENT: {}**".format(
            generation_scheduler.performance_time,
            generation_scheduler.generation_process.generation_trace[generation_scheduler.performance_time]))
