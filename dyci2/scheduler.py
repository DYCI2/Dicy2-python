from abc import ABC, abstractmethod

from generation_engine import GenerationEngine
from query import Query


class Scheduler(ABC):
    def __init__(self, generation_engine: GenerationEngine):
        self.generation_engine: GenerationEngine = generation_engine

    @abstractmethod
    def process_query(self, query: Query):
        # TODO: With a proper scheduling solution, this should probably be split in
        #  `schedule_query(Query) -> None` and `update_time(int time) -> Event/Output`
        """ Abstract method for adding query to scheduler

        raises: RuntimeError if unable to schedule query
        """

    @abstractmethod
    def start(self):
        """ Abstract method for starting the scheduler """


class InstantScheduler(Scheduler):
    def __init__(self, generation_engine: GenerationEngine):
        super().__init__(generation_engine=generation_engine)
        self.performance_time: int = 0

    def process_query(self, query: Query):
        self.generation_engine.process_query(query, self.performance_time)

    def start(self):
        self.performance_time = 0
