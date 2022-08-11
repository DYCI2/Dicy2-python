from enum import Enum

from merge.stubs.timepoint import Timepoint


class TimeMode(Enum):
    RELATIVE = "relative"
    ABSOLUTE = "absolute"

    @classmethod
    def from_string(cls, time_mode: str) -> 'TimeMode':
        """ raises: ValueError if input doesn't match an existing TimeMode """
        return TimeMode(time_mode.lower())


class Dyci2Timepoint(Timepoint):
    def __init__(self, start_date: int = 0, time_mode: TimeMode = TimeMode.ABSOLUTE):
        self.start_date: int = start_date
        self.time_mode: TimeMode = time_mode

    def to_absolute(self, performance_time: int) -> None:
        if self.time_mode == TimeMode.RELATIVE:
            self.start_date: int = max(performance_time + self.start_date, performance_time)
            self.time_mode = TimeMode.ABSOLUTE
