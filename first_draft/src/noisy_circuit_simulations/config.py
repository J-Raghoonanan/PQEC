"""Typed configuration for the circuit-level PQES simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class NoiseFamily(str, Enum):
    """Local channel applied after gates and during memory intervals."""

    DEPOLARIZING = "depolarizing"
    DEPHASING = "dephasing"


class CircuitKind(str, Enum):
    """Workload block repeated between purification placements."""

    IDLE = "idle"
    ECHO = "echo"
    BRICKWORK = "brickwork"


class PurificationSchedule(str, Enum):
    """Placement of a matched total number of PQES levels."""

    NONE = "none"
    INTERLEAVED = "interleaved"
    END_ONLY = "end_only"


@dataclass(frozen=True)
class HardwareNoiseSpec:
    """Independent error probabilities for the modeled hardware locations."""

    family: NoiseFamily = NoiseFamily.DEPOLARIZING
    p_1q: float = 0.0
    p_2q: float = 0.0
    p_cswap: float = 0.0
    p_memory: float = 0.0
    p_measure: float = 0.0
    p_reset: float = 0.0

    @classmethod
    def uniform(
        cls,
        family: NoiseFamily,
        p: float,
    ) -> "HardwareNoiseSpec":
        """Use the same location error probability everywhere."""
        return cls(
            family=family,
            p_1q=p,
            p_2q=p,
            p_cswap=p,
            p_memory=p,
            p_measure=p,
            p_reset=p,
        )

    @property
    def base_error_rate(self) -> float:
        """Primary x-axis value used by uniform-p grid sweeps."""
        return float(self.p_1q)

    def validate(self) -> None:
        for name in (
            "p_1q",
            "p_2q",
            "p_cswap",
            "p_memory",
            "p_measure",
            "p_reset",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")


@dataclass(frozen=True)
class CircuitSpec:
    """Ideal preparation and workload circuit configuration."""

    M: int = 1
    cycles: int = 5
    kind: CircuitKind = CircuitKind.BRICKWORK
    seed: int = 7
    rotation_scale: float = 1.0

    def validate(self) -> None:
        if self.M < 1:
            raise ValueError(f"M must be positive, got {self.M}")
        if self.cycles < 1:
            raise ValueError(f"cycles must be positive, got {self.cycles}")
        if self.rotation_scale < 0:
            raise ValueError("rotation_scale must be nonnegative")


@dataclass(frozen=True)
class PurificationSpec:
    """PQES depth and placement."""

    schedule: PurificationSchedule = PurificationSchedule.INTERLEAVED
    levels_per_cycle: int = 1

    def validate(self) -> None:
        if self.levels_per_cycle < 0:
            raise ValueError("levels_per_cycle must be nonnegative")
        if (
            self.schedule == PurificationSchedule.NONE
            and self.levels_per_cycle != 0
        ):
            raise ValueError("schedule='none' requires levels_per_cycle=0")


@dataclass(frozen=True)
class ExperimentSpec:
    """Complete specification for one circuit-level trajectory."""

    circuit: CircuitSpec
    hardware: HardwareNoiseSpec
    purification: PurificationSpec
    out_dir: Path = field(
        default_factory=lambda: Path("data/noisy_circuit_simulations")
    )

    def validate(self) -> None:
        self.circuit.validate()
        self.hardware.validate()
        self.purification.validate()
        if self.circuit.M > 4:
            raise ValueError(
                "The exact noisy SWAP instrument scales as 2^(1+2M); "
                "M > 4 is intentionally disabled for grid runs."
            )

    @property
    def total_purification_depth(self) -> int:
        if self.purification.schedule == PurificationSchedule.NONE:
            return 0
        return self.circuit.cycles * self.purification.levels_per_cycle

    @property
    def leaf_copies(self) -> int:
        return 2**self.total_purification_depth

    def synthesize_run_id(self) -> str:
        p = self.hardware.base_error_rate
        return "_".join(
            [
                f"M{self.circuit.M}",
                f"T{self.circuit.cycles}",
                self.circuit.kind.value,
                self.hardware.family.value,
                f"p{p:.6f}",
                self.purification.schedule.value,
                f"L{self.purification.levels_per_cycle}",
                f"s{self.circuit.seed}",
            ]
        )


__all__ = [
    "CircuitKind",
    "CircuitSpec",
    "ExperimentSpec",
    "HardwareNoiseSpec",
    "NoiseFamily",
    "PurificationSchedule",
    "PurificationSpec",
]
