"""Raw operation accounting; intentionally no uncalibrated 'equivalent FLOPs'."""

from __future__ import annotations

from dataclasses import asdict, dataclass

_COUNTER_ALIASES: dict[str, str] = {
    "initial_audits": "initial_audit_calls",
    "reaudits": "reaudit_calls",
    "re_audits": "reaudit_calls",
}


@dataclass(frozen=True)
class CostLedger:
    accepted_tokens: int = 0
    discarded_tokens: int = 0
    repair_tokens: int = 0
    projection_steps: int = 0
    compiler_calls: int = 0
    initial_audit_calls: int = 0
    reaudit_calls: int = 0

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if any(
            type(getattr(self, key)) is not int or getattr(self, key) < 0
            for key in self.__dataclass_fields__
        ):
            raise ValueError("all counters must be nonnegative integers")

    def record(self, counter: str, amount: int = 1) -> None:
        self._validate()
        canonical = _COUNTER_ALIASES.get(counter, counter)
        if canonical not in self.__dataclass_fields__:
            raise ValueError(f"unknown operation counter: {counter}")
        if type(amount) is not int or amount < 0:
            raise ValueError("counts must be nonnegative integers; rollback cannot refund cost")
        object.__setattr__(self, canonical, getattr(self, canonical) + amount)

    @property
    def initial_audits(self) -> int:
        return self.initial_audit_calls

    @property
    def reaudits(self) -> int:
        return self.reaudit_calls

    @property
    def total_generated_tokens(self) -> int:
        return self.accepted_tokens + self.discarded_tokens + self.repair_tokens

    def to_dict(self) -> dict[str, int]:
        self._validate()
        return {**asdict(self), "total_generated_tokens": self.total_generated_tokens}

    def within(self, limits: dict[str, int]) -> bool:
        values = self.to_dict()
        for raw_key, value in limits.items():
            key = _COUNTER_ALIASES.get(raw_key, raw_key)
            if key not in values or type(value) is not int or value < 0:
                raise ValueError("invalid budget dimension or limit")
            if values[key] > value:
                return False
        return True
