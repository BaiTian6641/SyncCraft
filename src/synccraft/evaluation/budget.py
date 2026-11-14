"""Raw operation accounting; intentionally no uncalibrated 'equivalent FLOPs'."""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class CostLedger:
    accepted_tokens: int = 0
    discarded_tokens: int = 0
    repair_tokens: int = 0
    projection_steps: int = 0
    compiler_calls: int = 0
    initial_audit_calls: int = 0
    reaudit_calls: int = 0

    def __post_init__(self):
        self._validate()

    def _validate(self):
        if any(type(getattr(self, key)) is not int or getattr(self, key) < 0
               for key in self.__dataclass_fields__):
            raise ValueError("all counters must be nonnegative integers")

    def record(self, counter: str, amount: int = 1):
        self._validate()
        if counter not in self.__dataclass_fields__:
            raise ValueError(f"unknown operation counter: {counter}")
        if type(amount) is not int or amount < 0:
            raise ValueError("counts must be nonnegative integers; rollback cannot refund cost")
        object.__setattr__(self, counter, getattr(self, counter) + amount)

    @property
    def total_generated_tokens(self):
        return self.accepted_tokens + self.discarded_tokens + self.repair_tokens

    def to_dict(self):
        self._validate()
        return {**asdict(self), "total_generated_tokens": self.total_generated_tokens}

    def within(self, limits: dict[str, int]):
        values = self.to_dict()
        for key, value in limits.items():
            if key not in values or type(value) is not int or value < 0:
                raise ValueError("invalid budget dimension or limit")
        return all(values[key] <= value for key, value in limits.items())
