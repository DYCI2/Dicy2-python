from enum import Enum


class OscSendProtocol:
    CREATE_AGENT = "create_agent"
    DELETE_AGENT = "delete_agent"
    QUERY_AGENTS = "query_agents"

    STATUS = "status"
    TERMINATED = "terminated"
    INITIALIZED = "initialized"

    SERVER_RECEIVED_QUERY = "server_received_query"
    QUERY_RESULT = "query_result"
    PERFORMANCE_TIME = "performance_time"
    CLEAR = "clear"
    EVENT_LEARNED = "new_event_learned"
    CONTROL_PARAMETER = "control_parameter"
    RESET_STATE = "reset_state"

    QUERY_GENERATION_TRACE = "query_generation_trace"

    # LOGGING: keywords are "critical", "error", "info", "warning" and "debug"


class Signal(Enum):
    """ Signals for communicating between processes through the main multiprocessing.Queue """
    TERMINATE = 1
