from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel


@dataclass
class BaseStep(ABC):
    config: BaseModel
    name: str

    @abstractmethod
    def run(self) -> Any:
        """Executes the logic of the pipeline step.

        This method must be implemented by all concrete subclasses of BaseStep.
        It should define the specific operations to be performed in the pipeline step.

        Returns:
            Any: The result of the step execution. The return type may vary depending on the step's
            purpose.
        """
        pass
