import argparse
import asyncio
import logging
import multiprocessing
from typing import Dict, Optional, Type, Tuple

from dyci2.agent import Agent
from dyci2.label import Dyci2Label, ListLabel
from dyci2.osc_protocol import SendProtocol
from dyci2.signals import Signal
from merge.io.async_osc import AsyncOsc
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
                         osc_log_address=self.DEFAULT_ADDRESS,
                         prepend_address_on_osc_call=False)
        # Key: osc_address for that particular agent
        self.agents: Dict[str, Tuple[Agent, multiprocessing.Queue]] = {}

    ################################################################################################################
    # PROCESS CONTROL (DO NOT CALL OVER OSC)
    ################################################################################################################

    async def _main_loop(self) -> None:
        self.default_log_config()
        self.logger.info("DYCI2 server started")
        self.send(SendProtocol.INITIALIZED)
        while self.running:
            self.send(SendProtocol.STATUS, "bang")
            await asyncio.sleep(self.STATUS_INTERVAL)

        self.logger.info("DYCI2 server terminated")
        self.send(SendProtocol.TERMINATED, "bang")

    def _unmatched_osc(self, address: str, *args) -> None:
        if address in self.agents:
            queue: multiprocessing.Queue = self.agents[address][1]
            queue.put(args)
        else:
            self.logger.error(f"Unknown OSC address {address}. (Did you initialize the agent?)")

    # def _process_osc(self, address: str, *args) -> None:
    #     if address == self.DEFAULT_ADDRESS:
    #         super()._process_osc(address, *args)
    #     elif address in self.agents:
    #         queue: multiprocessing.Queue = self.agents[address][1]
    #         queue.put(args)
    #     else:
    #         self.logger.error(f"Unknown OSC address {address}. Message was ignored")

    ################################################################################################################
    # OSC MESSAGES
    ################################################################################################################

    def create_agent(self,
                     agent_osc_address: str,
                     label_type_str: Optional[str] = None,
                     override: bool = False) -> None:

        if label_type_str is None:
            label_type: Type[Dyci2Label] = ListLabel
        else:
            try:
                label_type: Type[Dyci2Label] = Dyci2Label.type_from_string(label_type_str)
            except LabelError as e:
                self.logger.error(f"{str(e)}. No agent was created")
                return

        if not AsyncOsc.is_valid_osc_address(agent_osc_address):
            self.logger.error(f"Invalid OSC address ({agent_osc_address}). No agent was created")
            return

        if agent_osc_address in self.agents:
            if override:
                self._delete_agent(agent_osc_address)
            else:
                self.logger.error(f"An agent with the address {agent_osc_address} already exists.")
                return

        queue: multiprocessing.Queue = multiprocessing.Queue()
        agent: Agent = Agent(osc_address=agent_osc_address,
                             send_port=self.send_port,
                             ip=self.ip,
                             server_control_queue=queue,
                             label_type=label_type)

        agent.start()

        self.agents[agent_osc_address] = agent, queue
        self.logger.info(f"Created new agent at '{agent_osc_address}'")
        self.send(SendProtocol.CREATE_AGENT, agent_osc_address)

    def delete_agent(self, agent_osc_address: str) -> None:
        try:
            self.logger.info(f"Attempting to deleting agent {agent_osc_address}...")
            self._delete_agent(agent_osc_address)
            self.logger.info(f"Agent {agent_osc_address} was deleted")
            self.send(SendProtocol.DELETE_AGENT, agent_osc_address)
        except KeyError:
            self.logger.error(f"No agent with address {agent_osc_address} exists. Could not delete agent")
            return

    def query_agents(self):
        if len(self.agents) == 0:
            self.send(SendProtocol.QUERY_AGENTS, "None")
        else:
            self.send(SendProtocol.QUERY_AGENTS, *[agent_osc_address for agent_osc_address in self.agents.keys()])

    def exit(self) -> None:
        for agent, queue in self.agents.values():
            queue.put(Signal.TERMINATE)
            agent.join()
        self.stop()

    ################################################################################################################
    # PRIVATE
    ################################################################################################################

    def _delete_agent(self, agent_osc_address: str) -> None:
        """ raises: KeyError if no agent exists for identifier """
        agent, queue = self.agents[agent_osc_address]
        queue.put(Signal.TERMINATE)
        agent.join(timeout=5)
        del self.agents[agent_osc_address]

    # def _port_in_use(self, port: int) -> bool:
    #     return (port in [agent.recv_port for agent, _ in self.agents.values()] or
    #             port in [agent.send_port for agent, _ in self.agents.values()])


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

    parser_args = parser.parse_args()

    server: Dyci2Server = Dyci2Server(parser_args.recvport, parser_args.sendport, parser_args.ip)
    server.start()
