import argparse
import asyncio
import logging
import multiprocessing
from typing import Dict, Optional, Type, Tuple

from dyci2.agent import OscAgent
from dyci2.label import Dyci2Label, ListLabel
from dyci2.osc_protocol import OscSendProtocol
from dyci2.signals import Signal
from merge.io.async_osc import AsyncOscWithStatus, AsyncOsc
from merge.main.exceptions import LabelError


class Dyci2Server(AsyncOsc):
    DEFAULT_ADDRESS = "/server"
    DEFAULT_RECV_PORT = 4566
    DEFAULT_SEND_PORT = 1233
    STATUS_INTERVAL = 1.0

    def __init__(self, recv_port: int, send_port: int, ip: str):
        super().__init__(recv_port=recv_port,
                         send_port=send_port,
                         ip=ip,
                         default_address=self.DEFAULT_ADDRESS,
                         log_to_osc=True,
                         osc_log_address="/logging",
                         prepend_address_on_osc_call=False)
        self.agents: Dict[int, Tuple[OscAgent, multiprocessing.Queue]] = {}

    def exit(self) -> None:
        for agent, queue in self.agents.values():
            queue.put(Signal.TERMINATE)
            agent.join()
        self.stop()

    async def _main_loop(self):
        self.default_log_config()
        self.logger.info("DYCI2 server started")
        while self.running:
            self.send("bang", address=OscSendProtocol.STATUS)
            await asyncio.sleep(self.STATUS_INTERVAL)

        self.logger.info("DYCI2 server terminated")
        self.send("bang", address=OscSendProtocol.TERMINATED)

    def create_agent(self,
                     recv_port: int,
                     send_port: int,
                     label_type_str: Optional[str] = None,
                     identifier: Optional[int] = None,
                     override: bool = False) -> None:

        if label_type_str is None:
            label_type: Type[Dyci2Label] = ListLabel
        else:
            try:
                label_type: Type[Dyci2Label] = Dyci2Label.type_from_string(label_type_str)
            except LabelError as e:
                self.logger.error(f"{str(e)}. No agent was created")
                return

        if identifier is None:
            identifier = 1 if len(self.agents) == 0 else max(self.agents.keys()) + 1

        elif not isinstance(identifier, int):
            self.logger.error(f"The identifier must be an integer. No agent was created")
            return
        # else: identifier is valid

        if identifier in self.agents:
            if override:
                self._delete_agent(identifier)
            else:
                self.logger.error(f"An agent with the identifier {identifier} already exists. "
                                  f"Use override=True to override")
                return

        if self._port_in_use(send_port):
            self.logger.error(f"Port {send_port} is already in use by an agent. No agent was created")
            return

        if self._port_in_use(recv_port):
            self.logger.error(f"Port {recv_port} is already in use by an agent. No agent was created")
            return

        queue: multiprocessing.Queue = multiprocessing.Queue()
        agent: OscAgent = OscAgent(recv_port=recv_port,
                                   send_port=send_port,
                                   ip=self.ip,
                                   label_type=label_type,
                                   server_control_queue=queue)

        agent.start()

        self.agents[identifier] = agent, queue
        self.logger.info(f"Created agent {identifier} with inport {recv_port} and outport {send_port}")
        self.send(identifier, recv_port, send_port, address=OscSendProtocol.CREATE_AGENT)

    def delete_agent(self, identifier: int) -> None:
        try:
            self.logger.info(f"Deleting agent {identifier}...")
            self._delete_agent(identifier)
            self.logger.info(f"Agent {identifier} was deleted")
            self.send(identifier, address=OscSendProtocol.DELETE_AGENT)
        except KeyError:
            self.logger.error(f"No agent with identifier {identifier} exists: Could not delete agent")
            return

    def query_agents(self):
        if len(self.agents) == 0:
            self.send("None", address=OscSendProtocol.QUERY_AGENTS)
        for identifier, (agent, _) in self.agents.items():
            self.send(identifier, agent.recv_port, agent.send_port, address=OscSendProtocol.QUERY_AGENTS)

    def _delete_agent(self, identifier: int):
        """ raises: KeyError if no agent exists for identifier """
        agent, queue = self.agents[identifier]
        queue.put(Signal.TERMINATE)
        agent.join(timeout=5)
        del self.agents[identifier]

    def _port_in_use(self, port: int) -> bool:
        return (port in [agent.recv_port for agent, _ in self.agents.values()] or
                port in [agent.send_port for agent, _ in self.agents.values()])


if __name__ == '__main__':
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s]: %(message)s')

    parser = argparse.ArgumentParser(description='Launch and manage a DYCI2 server')
    parser.add_argument('--recvport', metavar='RECV_PORT', type=int,
                        help='input port used by the server', default=Dyci2Server.DEFAULT_RECV_PORT)
    parser.add_argument('--sendport', metavar='OUT_PORT', type=int, default=Dyci2Server.DEFAULT_SEND_PORT,
                        help='output port used by the server')
    parser.add_argument('--ip', metavar='IP', type=AsyncOsc.parse_ip, default="127.0.0.1",
                        help='ip address of the max client')

    args = parser.parse_args()

    server: Dyci2Server = Dyci2Server(args.recvport, args.sendport, args.ip)
    server.start()

