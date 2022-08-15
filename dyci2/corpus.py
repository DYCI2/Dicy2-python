from typing import List, Type


# TODO[B1]: Fully replaced with FreeCorpus in shared architecture
# class Memory:
#     def __init__(self, events: List[MemoryEvent], content_type: Type[MemoryEvent], label_type: Type[Label]):
#         self.events: List[MemoryEvent] = events
#         self.content_type: Type[MemoryEvent] = content_type
#         self.label_type: Type[Label] = label_type
#
#     @classmethod
#     def new_empty(cls, content_type: Type[MemoryEvent], label_type: Type[Label]):
#         return cls([], content_type, label_type)
#
#     def append(self, event: MemoryEvent):
#         self.events.append(event)
#
#     def length(self) -> int:
#         return len(self.events)