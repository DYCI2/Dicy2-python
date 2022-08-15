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
from multiprocessing import Process
from typing import Optional, Any, Union, List, Tuple, Type

from maxosc import Sender, SendFormat
from maxosc.caller import Caller
from maxosc.exceptions import MaxOscError
from maxosc.maxformatter import MaxFormatter
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from dyci2 import save_send_format
from dyci2.corpus_event import Dyci2CorpusEvent
from dyci2.dyci2_time import Dyci2Timepoint
from dyci2.label import Dyci2Label
from dyci2.parameter import Parameter
from dyci2.prospector import Dyci2Prospector, FactorOracleProspector
from generation_scheduler import Dyci2GenerationScheduler
from merge.main.corpus import GenericCorpus, Corpus
from merge.main.exceptions import InputError, StateError


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

    def __init__(self, inport: int = DEFAULT_INPORT, outport: int = DEFAULT_OUTPORT, **kwargs):
        Process.__init__(self, **kwargs)
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

    def stop_server(self):
        self._server.shutdown()

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


class OSCAgent(Server):
    DEFAULT_OSC_MAX_LEN = 100
    PATH_SEPARATOR = "::"

    def __init__(self,
                 inport: int = Server.DEFAULT_INPORT,
                 outport: int = Server.DEFAULT_OUTPORT,
                 label_type: Type[Dyci2Label] = Dyci2Label,
                 max_length_osc_mess: int = DEFAULT_OSC_MAX_LEN,
                 **kwargs):
        Server.__init__(self, inport=inport, outport=outport, **kwargs)
        self.logger = logging.getLogger(__name__)

        # TODO: ContentType was removed. If need to reimplement this constraint, see https://trello.com/c/BhfKYtSP

        corpus: GenericCorpus = GenericCorpus([], label_types=[label_type])
        prospector: Dyci2Prospector = FactorOracleProspector(corpus=corpus, label_type=label_type, **kwargs)
        self.generation_scheduler: Dyci2GenerationScheduler = Dyci2GenerationScheduler(prospector=prospector)

        self.max_length_osc_mess: int = max_length_osc_mess

    ################################################################################################################
    # PROCESS CONTROL (DO NOT CALL OVER OSC)
    ################################################################################################################

    def run(self) -> None:
        """ Note: `run()` should never be explicitly called! It's called internally by calling `start()` """
        self.generation_scheduler.start()
        Server.run(self)

    def start(self) -> None:
        Server.start(self)

    ################################################################################################################
    # MAIN RUNTIME CONTROL (OSC)
    ################################################################################################################

    def query(self,
              name: str,
              start_date: int,
              start_type: str,
              query_scope: int,
              label_type_str: Optional[str] = None,
              labels_str: Optional[List[str]] = None):
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

        abs_start_date: int = self.generation_scheduler.process_query(query=query)
        self.logger.debug(
            f"Output of the run of {name}: {self.generation_scheduler.generation_process.last_sequence()}")

        # TODO: Note! This format has also changed. Old:
        # message: list = [str(name), abs_start_date, start_unit, "absolute", scope_duration, scope_unit,
        #                  self.generation_handler.formatted_output_string()]
        message: List[Any] = [str(name), abs_start_date, "absolute", query_scope,
                              self.generation_scheduler.formatted_output_string()]
        self._client.send("/result_run_query", message)
        self._client.send("/updated_buffered_impro", self.generation_scheduler.formatted_generation_trace_string())
        self._send_to_antescofo(self.generation_scheduler.formatted_output_couple_content_transfo(), abs_start_date)

    def set_performance_time(self, new_time: int):
        if (isinstance(new_time, int) or isinstance(new_time, float)) and new_time >= 0:
            self.generation_scheduler.update_performance_time(time=Dyci2Timepoint(start_date=int(new_time)))
        else:
            raise InputError(f"set_performance_time can only handle integers larger than or equal to 0")

    ################################################################################################################
    # MODIFY STATE (OSC)
    ################################################################################################################

    def new_empty_memory(self, label_type: str = Dyci2Label.__class__.__name__) -> None:
        label_type: Type[Dyci2Label] = Dyci2Label.from_string(str(label_type))
        # TODO: Don't know how `content_type` works: need explicit protocol
        # content_type: Optional[TODO_INSERTTYPE] = None
        # if keys_content != "state":
        #     content_type = Label.from_string(str(keys_content))
        #     #exec("%s = %s" % ("content_type", keys_content))

        corpus: GenericCorpus = GenericCorpus([], label_types=[label_type])

        # TODO: This solution (which was used in old versions of DYCI2) will
        #  reset all parameters passed to the GenerationScheduler = not a good idea. Better to implement this with
        #  >>> self.generation_handler.read_memory(corpus)
        prospector: Dyci2Prospector = FactorOracleProspector(corpus=corpus, label_type=label_type)
        self.generation_scheduler: Dyci2GenerationScheduler = Dyci2GenerationScheduler(prospector)

        self._client.send("/new_empty_memory", str(label_type))
        self.send_init_control_parameters()

    def learn_event(self, label_type_str: str, label_value: Any, content_value: str) -> None:
        try:
            label_type: Type[Dyci2Label] = Dyci2Label.from_string(str(label_type_str))
            label: Dyci2Label = label_type(label_value)
        except ValueError as e:
            self.logger.error(f"{str(e)}. Could not learn event")
            return

        corpus: Optional[Corpus] = self.generation_scheduler.corpus
        if corpus is None:
            self.logger.error(f"A corpus must be initialized before learning events")
            return

        index: int = len(corpus)
        event: Dyci2CorpusEvent = Dyci2CorpusEvent(data=content_value, label=label, index=index)

        try:
            self.generation_scheduler.learn_event(event=event)
        except (TypeError, StateError) as e:
            self.logger.error(f"{str(e)}. Could not learn event")
            return

        self.logger.debug(f"index last state = {index}")
        self.logger.debug(f"associated label = {label} ({type(label)})")
        self.logger.debug(f"associated event = {event.renderer_info()} ({type(event)}")

        self._client.send("/new_event_learned", "bang")
        self.logger.info(f"Learned event {event}.")

    ################################################################################################################
    # PARAMETERS AND TEMPORAL CONTROL (OSC)
    ################################################################################################################

    def set_control_parameter(self, parameter_path_str: str, value: Any) -> None:
        try:
            parameter_path: List[str] = parameter_path_str.split(self.PATH_SEPARATOR)
            self.generation_scheduler.set_parameter(parameter_path, value)
        except (ValueError, KeyError) as e:
            self.logger.error(f"Could not set control parameter: {str(e)}")

    def send_init_control_parameters(self) -> None:
        parameters: List[Tuple[List[str], Parameter]] = self.generation_scheduler.get_parameters()
        for parameter_path, parameter in parameters:
            path: str = self.PATH_SEPARATOR.join(parameter_path)
            value: Any = parameter.get()
            self._client.send("/control_parameter", [path, value])

    def set_delta_transformation(self, delta: int) -> None:
        self.generation_scheduler.authorized_transforms = list(range(-delta, delta))
        # TODO: UI feedback (use self.logger.info(...))

    ################################################################################################################
    # PRIVATE
    ################################################################################################################

    # TODO: Design test case and update/simplify
    def _send_to_antescofo(self, original_output, abs_start_date):
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
