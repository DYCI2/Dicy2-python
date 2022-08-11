# -*-coding:Utf-8 -*


#############################################################################
# agent.py
# Jérôme Nika, IRCAM STMS LAB - Joakim Borg, IRCAM STMS LAB
# copyleft 2016 - 2020
#############################################################################


"""
OSC AGENT
===================
Class defining an OSC server embedding an instance of class :class:`~Generator.GenerationHandler`.
See the different tutorials accompanied with Max patches.

"""

import logging
import random
from multiprocessing import Process
from typing import Optional, Any, Union, List, Tuple, Dict, Type, Callable

from maxosc import Sender, SendFormat
from maxosc.caller import Caller
from maxosc.exceptions import MaxOscError
from maxosc.maxformatter import MaxFormatter
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from dyci2 import save_send_format
from generation_scheduler import GenerationScheduler
# TODO[JB]: This is a placeholder for all places where you're expected to specify the real type of the input value!
from dyci2.dyci2_label import Dyci2Label
from dyci2.dyci2_corpus_event import MemoryEvent, BasicEvent
from dyci2.dyci2_corpus import Memory
from dyci2.parameter import Parameter
from dyci2.query import Query, FreeQuery, TimeMode, LabelQuery

TODO_INSERTTYPE = Union[None, List, Tuple, Dict, int, float, str]

def basic_equiv(x, y):
    return x == y


class Target:
    WARNING_ADDRESS = "/warning"

    def __init__(self, port: int, ip: str):
        self._client: Sender = Sender(ip=ip, port=port, send_format=SendFormat.FLATTEN, cnmat_compatibility=True,
                                      warning_address=Target.WARNING_ADDRESS)

    def send(self, address: str, content: Any, **_kwargs):
        self._client.send(address, content)


class Server(Process, Caller):
    DEFAULT_IP = "127.0.0.1"
    SERVER_ADDRESS = "/server"
    DEFAULT_INPORT = 4567
    DEFAULT_OUTPORT = 1234
    DEBUG = True

    def __init__(self, inport: int = DEFAULT_INPORT, outport: int = DEFAULT_OUTPORT, *args, **kwargs):
        Process.__init__(self, *args, **kwargs)
        Caller.__init__(self, parse_parenthesis_as_list=True, discard_duplicate_args=False)

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format='%(message)s')

        self._inport: int = inport
        self._outport: int = outport
        self._server: Optional[BlockingOSCUDPServer] = None  # Initialized on `run` call
        self._client: Target = Target(self._outport, Server.DEFAULT_IP)

    def run(self) -> None:
        """ raises: OSError is server already is in use """
        osc_dispatcher: Dispatcher = Dispatcher()
        osc_dispatcher.map(self.SERVER_ADDRESS, self.__process_osc)
        osc_dispatcher.set_default_handler(self.__unmatched_osc)
        self._server: BlockingOSCUDPServer = BlockingOSCUDPServer((Server.DEFAULT_IP, self._inport), osc_dispatcher)
        self._server.serve_forever()

    def __process_osc(self, _address, *args) -> None:
        """
         raises:
            MaxOscError: Raised if input in any way is incorrectly formatted, if function doesn't exist
                     or has invalid argument names/values.
            Exception: Any uncaught exception by the function called will be raised here.
        """
        args_str: str = MaxFormatter.format_as_string(*args)
        try:
            self.call(args_str)
        except MaxOscError as e:
            self.logger.error(f"Incorrectly formatted input: {str(e)}.")
            return
        except Exception as e:
            self.logger.error(e)
            self.logger.debug(repr(e))
            if self.DEBUG:
                raise

    def __unmatched_osc(self, address: str, *_args, **_kwargs) -> None:
        self.logger.error(f"OSC address {address} is not registered. Use {self.SERVER_ADDRESS} for communication.")

    def stop_server(self):
        self._server.shutdown()

    def output_random(self, type_received_elements: str, received_elements: List[Any]):
        for i in range(len(received_elements)):
            print(f"Element {i}: type = {type_received_elements}, value= {received_elements[i]}")
        self._client.send("/random", random.random())


class OSCAgent(Server):
    DEFAULT_OSC_MAX_LEN = 100
    PATH_SEPARATOR = "::"

    def __init__(self, inport: int = Server.DEFAULT_INPORT, outport: int = Server.DEFAULT_OUTPORT,
                 equiv: Optional[Callable] = None,
                 label_type: Type[Dyci2Label] = Dyci2Label, content_type: Type[MemoryEvent] = BasicEvent,
                 authorized_transformations: Union[list, tuple] = (0,),
                 continuity_with_future: Tuple[float, float] = (0.0, 1.0),
                 max_length_osc_mess: int = DEFAULT_OSC_MAX_LEN, *args, **kwargs):
        Server.__init__(self, inport=inport, outport=outport, *args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.generation_handler: GenerationScheduler = GenerationScheduler(
            memory=Memory.new_empty(content_type=content_type, label_type=label_type),
            equiv=equiv,
            authorized_tranformations=authorized_transformations,
            continuity_with_future=continuity_with_future)

        self.max_length_osc_mess: int = max_length_osc_mess

    def run(self) -> None:
        self.generation_handler.start()
        Server.run(self)

    def start(self):
        Server.start(self)

    def set_performance_time(self, time_in_event: int):
        time_in_event = int(time_in_event)
        self.generation_handler.update_performance_time(new_time=time_in_event)

    # TODO: NOTE! Breaking change, original code commented out below
    def send_init_control_parameters(self):
        parameters: List[Tuple[List[str], Parameter]] = self.generation_handler.get_parameters()
        for parameter_path, parameter in parameters:
            path: str = self.PATH_SEPARATOR.join(parameter_path)
            value: Any = parameter.get()
            self._client.send("/control_parameter", [path, value])

    # def send_init_control_parameters(self):
    #     message: list = []
    #     count_types: dict = {}
    #     for slot in self.generation_handler.prospector.control_parameters:
    #         value = self.generation_handler.prospector.__dict__[slot]
    #         type_param: str = str(type(value))
    #         if type_param in count_types.keys():
    #             count_types[type_param] += 1
    #         else:
    #             count_types[type_param] = 1
    #         message.extend([type_param, count_types[type_param], slot, value])
    #     print(message)
    #     self._client.send("/control_parameters", message)

    # TODO: NOTE! Breaking change: will have to address full parameter path from max
    def set_control_parameter(self, parameter_path_str: str, value: Any):
        try:
            parameter_path: List[str] = parameter_path_str.split(self.PATH_SEPARATOR)
            self.generation_handler.set_parameter(parameter_path, value)
        except (ValueError, KeyError) as e:
            self.logger.error(f"Could not set control parameter: {str(e)}")

    def set_delta_transformation(self, delta: int):
        # TODO: is a range past 12 valid or should this one check for range [-12, 12]?
        self.generation_handler.authorized_transforms = list(range(-delta, delta))
        # TODO: UI feedback (use self.logger.info(...))

    # TODO: This part of the code has not been updated and will need a proper test case before this can be done
    # def load_generation_handler_from_json_file(self, dict_memory: TODO_INSERTTYPE, keys_label: TODO_INSERTTYPE,
    #                                            keys_content: TODO_INSERTTYPE):
    #     label_type: Type[label] = label
    #     # TODO[JB]: Handle this with a simple Label.from_string implementation instead
    #     # TODO 2021 : CHECK THAT IT WORKS...
    #     label_type = label.from_string(str(keys_label))
    #     # exec("%s = %s" % ("label_type", keys_label))
    #     content_type: Optional[TODO_INSERTTYPE] = None
    #     # TODO[JB]: Handle this with a simple TODO_INSERTTYPE.from_string implementation instead
    #     if keys_content != "state":
    #         # TODO 2021 : CHECK THAT IT WORKS...
    #         content_type = label.from_string(str(keys_content))
    #         # exec("%s = %s" % ("content_type", keys_content))
    #
    #     # TODO: Manage parameters from max
    #     self.generation_handler = generator_builder.new_generation_handler_from_json_file(
    #         path_json_file=dict_memory, keys_labels=keys_label,
    #         keys_contents=keys_content, model_navigator="FactorOracleNavigator",
    #         label_type=label_type, authorized_tranformations=(0,),
    #         continuity_with_future=(0.0, 1.0), content_type=content_type)
    #
    #     self._client.send("/new_generator_built_from_memory", str(dict_memory))
    #     self.send_init_control_parameters()
    #     l_dates, l_labels, length, l_pos = save_send_format.load_dates_json_memory_in_antescofo(dict_memory, keys_label)
    #     try:
    #         assert length == len(l_dates) and length == len(l_pos) and length == len(l_labels)
    #     except AssertionError as e:
    #         print("Sequence of dates and sequence of labels have different lengths", e)
    #     else:
    #         self._client.send("/antescofo", ["/length_mem_Voice", length])
    #
    #         pos = 0
    #         while pos < length:
    #             time.sleep(0.2)
    #             end_temp: int = min(pos + self.max_length_osc_mess - 1, length - 1)  # TODO[JB]: Dont use temp in name
    #             print(f"{pos}------>{end_temp}")
    #             self._client.send("/antescofo", ["/start_pos_load_Voice", pos])
    #             time.sleep(0.1)
    #
    #             l_dates_temp: TODO_INSERTTYPE = l_dates[pos:end_temp + 1]  # TODO[JB]: Dont use temp in name
    #             self._client.send("/antescofo", ["/load_dates_Voice", l_dates_temp])
    #             time.sleep(0.1)
    #
    #             l_labels_temp: TODO_INSERTTYPE = l_labels[pos:end_temp + 1]  # TODO[JB]: Dont use temp in name
    #             self._client.send("/antescofo", ["/load_labels_Voice", l_labels_temp])
    #             time.sleep(0.1)
    #
    #             l_pos_temp: TODO_INSERTTYPE = l_pos[pos:end_temp + 1]
    #             self._client.send("/antescofo", ["/load_pos_in_scenario_Voice", l_pos_temp])
    #             time.sleep(0.1)
    #
    #             pos = end_temp + 1

    # TODO: This part of the code has not been updated and will need a proper test case before this can be done
    # def new_empty_memory(self, keys_label: TODO_INSERTTYPE, keys_content: TODO_INSERTTYPE):
    #     # TODO[JB]: The following lines are just code duplication from load_generation_handler_from_json_file
    #     # TODO 2021 : CHECK THAT IT WORKS...
    #     label_type: Type[Label] = Label.from_string(str(keys_label))
    #     # exec("%s = %s" % ("label_type", keys_label))
    #     content_type: Optional[TODO_INSERTTYPE] = None
    #     # TODO[JB]: Handle this with a simple TODO_INSERTTYPE.from_string implementation instead
    #     if keys_content != "state":
    #         # TODO 2021 : CHECK THAT IT WORKS...
    #         content_type = label.from_string(str(keys_content))
    #         # exec("%s = %s" % ("content_type", keys_content))
    #
    #     self.generation_handler: GenerationScheduler = GenerationScheduler(label_type=label_type,
    #                                                                        content_type=content_type)
    #     self._client.send("/new_empty_memory", keys_label)
    #     self.send_init_control_parameters()

    # TODO: Note: Changed signature. Update max code accordingly. Old:
    # def learn_event(self, label_type_str: str, label_value: Any, content_type_str: str, content_value: str):
    def learn_event(self, label_type_str: str, label_value: Any, content_value: str):
        try:
            label_type: Type[Dyci2Label] = Dyci2Label.from_string(str(label_type_str))
            label: Dyci2Label = label_type(label_value)
        except ValueError as e:
            self.logger.error(f"{str(e)}. Could not learn event")
            return

        content: MemoryEvent = BasicEvent(data=content_value, label=label)

        self.generation_handler.learn_event(event=content)
        index_last_state: int = self.generation_handler.prospector._model.index_last_state()
        label_last_state: Dyci2Label = self.generation_handler.prospector._model.labels[index_last_state]
        content_last_state: Any = self.generation_handler.prospector._model.sequence[index_last_state]
        self.logger.debug(f"index last state = {index_last_state}")
        self.logger.debug(f"associated label = {label_last_state} ({type(label_last_state)})")
        self.logger.debug(f"associated content = {content_last_state} ({type(content_last_state)}")

        self._client.send("/new_event_learned", "bang")
        self.logger.info(f"Learned event {content}.")

    # TODO: Note: Changed signature. Update max code accordingly
    def handle_new_query(self, name: str, start_date: int, start_type: str, query_scope: int,
                         label_type_str: Optional[str] = None, labels_str: Optional[List[str]] = None):
        query_as_list: List[Union[str, int]] = [name, start_date, start_type, query_scope, label_type_str, labels_str]
        self.logger.debug(f"query_as_list = {query_as_list}")
        for i, v in enumerate(query_as_list):
            self.logger.debug(f"Element {i}  = {v}")

        try:
            time_mode: TimeMode = TimeMode.from_string(start_type)
        except ValueError as e:
            self.logger.error(f"{str(e)}. Query was ignored.")
            return

        if label_type_str is None and labels_str is None and query_scope > 0:
            query: Query = FreeQuery(num_events=query_scope, start_date=start_date, time_mode=time_mode)
        elif label_type_str is not None and labels_str is not None and query_scope == len(labels_str):
            try:
                label_type: Type[Dyci2Label] = Dyci2Label.from_string(label_type_str)
                labels: List[Dyci2Label] = [label_type(s) for s in labels_str]
                query: Query = LabelQuery(labels=labels, start_date=start_date, time_mode=time_mode)
            except ValueError as e:
                self.logger.error(f"{str(e)}. Query was ignored.")
                return
        else:
            self.logger.error(f"Invalid query format. Query was ignored.")
            return

        self._client.send("/server_received_query", str(name))

        # TODO: Format the output

        abs_start_date: int = self.generation_handler.process_query(query=query)
        self.logger.debug(f"Output of the run of {name}: {self.generation_handler.generation_process.last_sequence()}")

        # TODO: Note! This format has also changed. Old:
        # message: list = [str(name), abs_start_date, start_unit, "absolute", scope_duration, scope_unit,
        #                  self.generation_handler.formatted_output_string()]
        message: List[Any] = [str(name), abs_start_date, "absolute", query_scope,
                              self.generation_handler.formatted_output_string()]
        self._client.send("/result_run_query", message)
        self._client.send("/updated_buffered_impro", self.generation_handler.formatted_generation_trace_string())
        self._send_to_antescofo(self.generation_handler.formatted_output_couple_content_transfo(), abs_start_date)

    # TODO: Design test case and update/simplify
    def _send_to_antescofo(self, original_output: List[TODO_INSERTTYPE], abs_start_date: TODO_INSERTTYPE):
        if len(original_output) > 0:
            i = 0
            e = original_output[i]
            list_to_send = []
            while i < len(original_output) and e is not None:
                e = original_output[i]
                if e is not None:
                    list_to_send.append(e)
                i = i + 1

            # TODO: What to do with None?
            if len(list_to_send) > 0:
                print("... SENT TO MAX : {}".format(list_to_send))
                map_antescofo = save_send_format.write_list_as_antescofo_map(list_to_send, abs_start_date)
                self._client.send("/antescofo", ["/updated_buffered_impro_with_info_transfo", map_antescofo])
