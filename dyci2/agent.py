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
from abc import abstractmethod
from multiprocessing import Process
from typing import Optional, Any, Union, List, Tuple, Type

from maxosc.caller import Caller
from maxosc.exceptions import MaxOscError
from maxosc.maxformatter import MaxFormatter
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

from dyci2.corpus_event import Dyci2CorpusEvent
from dyci2.dyci2_time import Dyci2Timepoint, TimeMode
from dyci2.generation_scheduler import Dyci2GenerationScheduler
from dyci2.label import Dyci2Label, ListLabel
from dyci2.osc_protocol import OscSendProtocol
from dyci2.parameter import Parameter
from dyci2.prospector import Dyci2Prospector, FactorOracleProspector
from dyci2.utils import FormattingUtils, GenerationTraceFormatter
from merge.io.osc_sender import OscLogForwarder, OscSender
from merge.main.candidate import Candidate
from merge.main.corpus import GenericCorpus, Corpus
from merge.main.exceptions import StateError, LabelError, QueryError, TimepointError
from merge.main.influence import Influence
from merge.main.query import Query, TriggerQuery, InfluenceQuery


def basic_equiv(x, y):
    return x == y


class Server(Process, Caller):
    DEFAULT_IP = "127.0.0.1"
    SERVER_ADDRESS = "/server"
    DEFAULT_INPORT = 4567
    DEFAULT_OUTPORT = 1234
    DEBUG = True

    def __init__(self,
                 inport: int = DEFAULT_INPORT,
                 outport: int = DEFAULT_OUTPORT,
                 osc_log_address: Optional[str] = None,
                 log_level: int = logging.DEBUG,
                 **kwargs):
        Process.__init__(self)
        Caller.__init__(self, parse_parenthesis_as_list=True, discard_duplicate_args=False)

        self.logger = logging.getLogger(__name__)
        self._log_level: int = log_level
        logging.basicConfig(level=log_level, format='%(message)s')
        self.osc_log_address: Optional[str] = osc_log_address

        self._inport: int = inport
        self._outport: int = outport
        self._server: Optional[BlockingOSCUDPServer] = None  # Initialized on `run` call
        self._client: OscSender = OscSender(ip=Server.DEFAULT_IP, port=self._outport)

    @abstractmethod
    def _on_log_handler_added(self, handler: logging.Handler) -> None:
        pass

    def run(self) -> None:
        """ raises: OSError is server already is in use """
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=self._log_level, format='%(message)s')
        if self.osc_log_address is not None:
            osc_log_handler = OscLogForwarder(self._client, self.osc_log_address)
            self.logger.addHandler(osc_log_handler)
            self._on_log_handler_added(osc_log_handler)

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
            self.logger.error(f"Incorrectly formatted input: {str(e)}")
            return
        except Exception as e:
            self.logger.error(e)
            if self.DEBUG:
                raise

    def __unmatched_osc(self, address: str, *_args, **_kwargs) -> None:
        self.logger.error(f"OSC address {address} is not registered. Use {self.SERVER_ADDRESS} for communication")


class ValueErorr:
    pass


class OSCAgent(Server):
    DEFAULT_OSC_MAX_LEN = 100
    PATH_SEPARATOR = "::"

    def __init__(self,
                 inport: int = Server.DEFAULT_INPORT,
                 outport: int = Server.DEFAULT_OUTPORT,
                 label_type: Type[Dyci2Label] = ListLabel,
                 max_length_osc_mess: int = DEFAULT_OSC_MAX_LEN,
                 **kwargs):
        Server.__init__(self, inport=inport, outport=outport, **kwargs)
        self.logger = logging.getLogger(__name__)

        # TODO: ContentType was removed. If need to reimplement this constraint, see https://trello.com/c/BhfKYtSP

        corpus: GenericCorpus = GenericCorpus([], label_types=[label_type])
        prospector: Dyci2Prospector = FactorOracleProspector(corpus=corpus, label_type=label_type)
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
              labels_data: Optional[List[str]] = None):
        query_as_list: List[Union[str, int]] = [name, start_date, start_type, query_scope, label_type_str, labels_data]
        self.logger.debug(f"query_as_list = {query_as_list}")
        for i, v in enumerate(query_as_list):
            self.logger.debug(f"Element {i}  = {v}")

        if not (isinstance(start_date, int) or isinstance(start_date, float)) or start_date < 0:
            self.logger.error(f"Start date must be greater than or equal to 0. Query was ignored")
            return

        try:
            time_mode: TimeMode = TimeMode.from_string(start_type)
        except ValueError as e:
            self.logger.error(f"{str(e)}. Query was ignored")
            return

        timepoint: Dyci2Timepoint = Dyci2Timepoint(start_date=int(start_date), time_mode=time_mode)

        if label_type_str is None and labels_data is None and isinstance(query_scope, int) and query_scope > 0:
            query: Query = TriggerQuery(content=query_scope, time=timepoint)

        elif label_type_str is not None and labels_data is not None and query_scope == len(labels_data):
            try:
                label_type: Type[Dyci2Label] = Dyci2Label.type_from_string(label_type_str)
                labels: List[Dyci2Label] = [label_type.parse(s) for s in labels_data]
                query: Query = InfluenceQuery(content=Influence.from_labels(labels), time=timepoint)
            except ValueError as e:
                self.logger.error(f"{str(e)}. Query was ignored")
                return
        else:
            self.logger.error(f"Invalid query format. Query was ignored")
            return

        self._client.send(OscSendProtocol.SERVER_RECEIVED_QUERY, str(name))

        try:
            self.generation_scheduler.process_query(query=query)
        except (QueryError, StateError, TimepointError) as e:
            self.logger.error(f"{str(e)}. Query was ignored")
            return

        abs_start_date: int = self.generation_scheduler.generation_index()

        output_sequence: List[Optional[Candidate]] = self.generation_scheduler.generation_process.last_sequence()

        self.logger.debug(f"Output of the run of {name}: "
                          f"'{FormattingUtils.output_without_transforms(output_sequence, use_max_format=False)}'")

        # TODO[Jerome]: Add support for transforms
        message: List[Any] = [str(name), abs_start_date, "absolute", query_scope,
                              FormattingUtils.output_without_transforms(output_sequence, use_max_format=True)]

        self._client.send(OscSendProtocol.QUERY_RESULT, message)

    def set_performance_time(self, new_time: int) -> None:
        if (isinstance(new_time, int) or isinstance(new_time, float)) and new_time >= 0:
            timepoint = Dyci2Timepoint(start_date=int(new_time))
            self.generation_scheduler.update_performance_time(time=timepoint)
            self._client.send(OscSendProtocol.PERFORMANCE_TIME, self.generation_scheduler.performance_time)
        else:
            self.logger.error(f"set_performance_time can only handle integers larger than or equal to 0")
            return

    def increment_performance_time(self, increment: int = 1) -> None:
        if not isinstance(increment, int):
            self.logger.error(f"increment_performance_time can only handle integers")
            return

        self.generation_scheduler.increment_performance_time(increment=increment)
        self._client.send(OscSendProtocol.PERFORMANCE_TIME, self.generation_scheduler.performance_time)

    ################################################################################################################
    # MODIFY STATE (OSC)
    ################################################################################################################

    def reset_memory(self,
                     label_type: str = Dyci2Label.__class__.__name__,
                     content_type: str = Dyci2CorpusEvent.__class__.__name__) -> None:
        # TODO: content_type is unused and only added as a placeholder for now

        try:
            label_type: Type[Dyci2Label] = Dyci2Label.type_from_string(str(label_type))
        except ValueError as e:
            self.logger.error(f"{str(e)}. Could not create new memory")
            return

        corpus: GenericCorpus = GenericCorpus([], label_types=[label_type])

        self.generation_scheduler.read_memory(corpus, override=True)

        self._client.send(OscSendProtocol.RESET_MEMORY, label_type.__name__)
        self.send_init_control_parameters()

    def learn_event(self, label_type_str: str, label_value: Any, content_value: str) -> None:
        try:
            label_type: Type[Dyci2Label] = Dyci2Label.type_from_string(str(label_type_str))
            label: Dyci2Label = label_type.parse(label_value)
        except (ValueError, LabelError) as e:
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
        except LabelError:
            self.logger.error(f"Corpus can only handle events of type(s) '"
                              f"{','.join([t.__name__ for t in corpus.label_types])}'. Could not learn event")
            return

        self.logger.debug(f"index last state = {index}")
        self.logger.debug(f"associated label = {label} ({label.__class__.__name__})")
        self.logger.debug(f"associated event = {event.renderer_info()} ({event.__class__.__name__})")

        self._client.send(OscSendProtocol.EVENT_LEARNED, "bang")
        self.logger.info(f"Learned event '{event.renderer_info()}'")

    ################################################################################################################
    # PARAMETERS AND STATE CONTROL (OSC)
    ################################################################################################################

    def set_control_parameter(self, parameter_path_str: str, value: Any) -> None:
        try:
            parameter_path: List[str] = parameter_path_str.split(self.PATH_SEPARATOR)
            param: Parameter = self.generation_scheduler.set_parameter(parameter_path, value)
            self.logger.debug(f"parameter '{parameter_path_str}' set to {param.value}")
        except (ValueError, KeyError) as e:
            self.logger.error(f"Could not set control parameter: {str(e)}")
            return

    def send_init_control_parameters(self) -> None:
        parameters: List[Tuple[List[str], Parameter]] = self.generation_scheduler.get_parameters()
        for parameter_path, parameter in parameters:
            path: str = self.PATH_SEPARATOR.join(parameter_path)
            value: Any = parameter.get()
            self._client.send(OscSendProtocol.CONTROL_PARAMETER, [path, value])

    def set_delta_transformations(self, delta: int) -> None:
        if delta < 0:
            self.logger.error("Value must be greater than or equal to 0. No transformations were set")
            return

        self.generation_scheduler.authorized_transforms = list(range(-delta, delta))

        if delta == 0:
            self.logger.debug(f"Transforms disabled")
        else:
            transforms_str: str = ', '.join([str(t) for t in self.generation_scheduler.authorized_transforms])
            self.logger.debug(f"Transforms {transforms_str} enabled")

    # TODO: Temp solution to handle logging over OSC
    def set_log_level(self, log_level: int) -> None:
        self.logger.setLevel(log_level)
        self.generation_scheduler.logger.setLevel(log_level)
        self.generation_scheduler.generation_process.logger.setLevel(log_level)
        self.generation_scheduler.generator.logger.setLevel(log_level)
        self.generation_scheduler.generator.prospector.logger.setLevel(log_level)
        self.generation_scheduler.generator.prospector.navigator.logger.setLevel(log_level)
        self.generation_scheduler.generator.prospector.model.logger.setLevel(log_level)
        self._log_level = log_level
        self.logger.debug(f"log level set to {logging.getLevelName(log_level)}")

    def clear(self) -> None:
        self.generation_scheduler.clear()
        self._client.send(OscSendProtocol.CLEAR, "bang")
        self._client.send(OscSendProtocol.PERFORMANCE_TIME, self.generation_scheduler.performance_time)

    ################################################################################################################
    # QUERY STATE
    ################################################################################################################

    def query_generation_trace(self, keyword: str = "", start: Optional[int] = None, end: Optional[int] = None):
        generation_trace: List[Optional[Candidate]] = self.generation_scheduler.generation_process.generation_trace
        if len(generation_trace) == 0:
            self.logger.error("No generation trace exists yet")
            return

        try:
            msg: List[Any] = GenerationTraceFormatter.query(keyword, generation_trace, start=start, end=end)
            self._client.send(OscSendProtocol.QUERY_GENERATION_TRACE, *msg)
        except QueryError as e:
            self.logger.error(str(e))
            return

    ################################################################################################################
    # PRIVATE
    ################################################################################################################

    def _on_log_handler_added(self, handler: logging.Handler) -> None:
        # TODO: Temporary, ugly solution for logging all messages to OSC
        self.generation_scheduler.logger.addHandler(handler)
        self.generation_scheduler.generation_process.logger.addHandler(handler)
        self.generation_scheduler.generator.logger.addHandler(handler)
        self.generation_scheduler.generator.prospector.logger.addHandler(handler)
        self.generation_scheduler.generator.prospector.navigator.logger.addHandler(handler)
        self.generation_scheduler.generator.prospector.model.logger.addHandler(handler)
