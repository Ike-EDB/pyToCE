"""Status values returned by the public workflows."""

from typing import Literal, TypeAlias

OperationStatus: TypeAlias = Literal["computed", "updated", "reused"]
ItemStatus: TypeAlias = Literal["computed", "updated", "reused"]
