import logging
import random
import sys
import typing
import warnings
from typing import List, Tuple
from unittest import TestCase

from dyci2.dyci2_label import ChordLabel, from_list_to_labels
from dyci2.factor_oracle_model import FactorOracle
from dyci2.factor_oracle_navigator import FactorOracleNavigator
from dyci2.generation_scheduler import Dyci2GenerationScheduler
from dyci2.transforms import Transform
from merge.main.candidate import Candidate
from merge.main.corpus import GenericCorpus
from merge.main.corpus_event import GenericCorpusEvent, CorpusEvent
from merge.main.influence import LabelInfluence
from merge.main.query import Query, InfluenceQuery, TriggerQuery
from dyci2.query import TimeMode, Dyci2Time


def chord_format(lst: List[Tuple[GenericCorpusEvent, int]]):
    return [[e.data, t] for (e, t) in lst]


def candidate_format(lst: List[Candidate]):
    return [typing.cast(GenericCorpusEvent, e.event).data for e in lst]


class TestDyci2GeneratorScheduler(TestCase):
    def test_basic(self):
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')
        random.seed(2)
        # warnings.filterwarnings("ignore")
        # original_stdout = sys.stdout
        # sys.stdout = None

        list_for_labels = ["d m7", "d m7", "g 7", "g 7", "c maj7", "c maj7", "c# maj7", "c# maj7", "d# m7", "d# m7",
                           "g# 7",
                           "g# 7", "c# maj7", "c# maj7"]
        list_for_sequence = ["d m7(1)", "d m7(2)", "g 7(3)", "g 7(4)", "c maj7(5)", "c maj7(6)", "c# maj7(7)",
                             "c# maj7(8)",
                             "d# m7(9)", "d# m7(10)", "g# 7(11)", "g# 7(12)", "c# maj7(13)", "c# maj7(14)"]
        labels: List[ChordLabel] = from_list_to_labels(labels=list_for_labels, label_type=ChordLabel)
        sequence: List[ChordLabel] = from_list_to_labels(labels=list_for_sequence, label_type=ChordLabel)
        print(labels)
        print(sequence)
        event: GenericCorpusEvent[ChordLabel] = GenericCorpusEvent(sequence[0], 0, features=None,
                                                                   labels={ChordLabel: labels[0]})
        print(event)
        memory: GenericCorpus[ChordLabel] = GenericCorpus([GenericCorpusEvent(content, i, labels={type(label): label})
                                                           for (i, (content, label)) in
                                                           enumerate(zip(sequence, labels))],
                                                          label_types=[ChordLabel])

        authorized_transf = list(range(-6, 6))
        gen_scheduler: Dyci2GenerationScheduler = Dyci2GenerationScheduler(memory=memory,
                                                                           model_class=FactorOracle,
                                                                           navigator_class=FactorOracleNavigator,
                                                                           label_type=ChordLabel,
                                                                           authorized_tranformations=authorized_transf)
        gen_scheduler.generator.prospector._navigator.avoid_repetitions_mode.set(1)
        gen_scheduler.generator.prospector._navigator.max_continuity.set(3)
        warnings.warn("This is not a settable parameter anymore: no_empty_event")
        # gen_scheduler.prospector.navigator.no_empty_event = False

        gen_scheduler.start()

        print("\n --- Starting simulation of interactions (receiving and processing query)...  --- ")

        list_for_scenario: List[str] = ["g m7", "g m7", "c 7", "c 7", "f maj7", "f maj7"]
        labels_for_scenario: List[ChordLabel] = from_list_to_labels(list_for_scenario, ChordLabel)
        influences_for_scenario: List[LabelInfluence] = [LabelInfluence(label) for label in labels_for_scenario]
        query: InfluenceQuery = InfluenceQuery(influences_for_scenario, time=Dyci2Time())
        # query = new_temporal_query_sequence_of_events(handle=list_for_scenario, label_type=ChordLabel)
        print("\n/!\ Receiving and processing a new query: /!\ \n{}".format(query))
        gen_scheduler.process_query(query=query)
        output: List[Candidate] = gen_scheduler.generation_process.last_sequence()
        events: List[CorpusEvent] = [c.event for c in output]
        transforms: List[Transform] = [c.transform for c in output]
        print("Output of the run:")
        for (e, t) in zip(events, transforms):
            print(f"    {e} {t}")

        print("With transforfmation: {}".format(
            chord_format(gen_scheduler.formatted_output_couple_content_transfo())))

        sys.exit(0)
        sys.stdout = None

        print(
            "/!\ Updated buffered improvisation: {} /!\ ".format(
                gen_scheduler.generation_process.generation_trace))

        print("\n --- ... and starting simulation of performance time (beat, 60 BPM) --- ")

        print("\n**NEW PERFORMANCE TIME : BEAT {}**\n**PLAYING CORRESPONDING GENERATED EVENT: {}**".format(
            gen_scheduler.performance_time,
            gen_scheduler.generation_process.generation_trace[gen_scheduler.performance_time]))

        # time.sleep(1)
        # generation_handler.current_performance_time["event"] += 1
        # generation_handler.current_performance_time["ms"] += 1000
        # generation_handler.current_performance_time["last_update_event_in_ms"] = generation_handler.current_performance_time["ms"]
        gen_scheduler.increment_performance_time(increment=1)

        # sys.exit(1)

        print("\n**NEW PERFORMANCE TIME : BEAT {}**\n**PLAYING CORRESPONDING GENERATED EVENT: {}**".format(
            gen_scheduler.performance_time,
            gen_scheduler.generation_process.generation_trace[gen_scheduler.performance_time]))

        query: Query = TriggerQuery(3, Dyci2Time(start_date=4, time_mode=TimeMode.ABSOLUTE))
        # query = new_temporal_query_free_sequence_of_events(length=3, start_date=4, start_type="absolute")
        print("\n/!\ Receiving and processing a new query: /!\ \n{}".format(query))
        gen_scheduler.process_query(query=query)
        print("Output of the run: {}".format(gen_scheduler.generation_process.last_sequence()))

        print("With transforfmation: {}".format(
            chord_format(gen_scheduler.formatted_output_couple_content_transfo())))

        sys.stdout = None

        print(
            "/!\ Updated buffered improvisation: {} /!\ ".format(
                gen_scheduler.generation_process.generation_trace))

        for i in range(0, 2):
            # time.sleep(1)
            # generation_handler.current_performance_time["event"] += 1
            # generation_handler.current_performance_time["ms"] += 1000
            # generation_handler.current_performance_time["last_update_event_in_ms"] = generation_handler.current_performance_time["ms"]
            gen_scheduler.increment_performance_time(increment=1)
            print("\n**NEW PERFORMANCE TIME : BEAT {}**\n**PLAYING CORRESPONDING GENERATED EVENT: {}**".format(
                gen_scheduler.performance_time,
                gen_scheduler.generation_process.generation_trace[gen_scheduler.performance_time]))

        # TODO : POURQUOI NE MARCHE PAS AVEC TRANSPO MIN DE -2 ????
        # scenario = make_sequence_of_chord_labels(["a maj7", "a maj7"])
        list_for_scenario = ["d m7", "d m7", "d m7"]
        labels_for_scenario: List[ChordLabel] = from_list_to_labels(list_for_scenario, ChordLabel)
        influences_for_scenario: List[LabelInfluence] = [LabelInfluence(label) for label in labels_for_scenario]
        query: InfluenceQuery = InfluenceQuery(influences_for_scenario,
                                               time=Dyci2Time(start_date=2, time_mode=TimeMode.RELATIVE))
        print("\n/!\ Receiving and processing a new query: /!\ \n{}".format(query))
        gen_scheduler.process_query(query=query)

        print("Output of the run: {}".format(
            candidate_format(gen_scheduler.generation_process.last_sequence())))

        print(
            "/!\ Updated buffered improvisation: {} /!\ ".format(
                candidate_format(gen_scheduler.generation_process.generation_trace)))
