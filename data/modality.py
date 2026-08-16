"""Stable MRI modality mapping for BrainMAP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

"""
维护 MRI 模态名称与整数标签之间的固定双向映射，例如 t1n、t1c、t2w、t2f 分别映射为 0、1、2、3，
为预训练中的模态分类监督提供标签。
"""

DEFAULT_MODALITIES = ("t1n", "t1c", "t2w", "t2f")


@dataclass(frozen=True)
class ModalityMapper:
    """Bidirectional mapping between modality names and stable integer indices."""

    modalities: List[str]

    def __post_init__(self) -> None:
        if not self.modalities:
            raise ValueError("modalities must not be empty")
        normalized = [str(item).lower() for item in self.modalities]
        if len(set(normalized)) != len(normalized):
            raise ValueError(f"Duplicate modalities are not allowed: {self.modalities}")
        object.__setattr__(self, "modalities", normalized)
        object.__setattr__(self, "_name_to_index", {name: idx for idx, name in enumerate(normalized)})

    @classmethod
    def default(cls) -> "ModalityMapper":
        return cls(list(DEFAULT_MODALITIES))

    @classmethod
    def from_config(cls, data_config: Dict[str, object]) -> "ModalityMapper":
        modalities = data_config.get("modalities", list(DEFAULT_MODALITIES))
        if not isinstance(modalities, Iterable) or isinstance(modalities, (str, bytes)):
            raise TypeError("data.modalities must be a sequence of modality names")
        return cls([str(item) for item in modalities])

    @property
    def num_modalities(self) -> int:
        return len(self.modalities)

    def to_index(self, modality: str) -> int:
        name = str(modality).lower()
        try:
            return self._name_to_index[name]
        except KeyError as exc:
            raise KeyError(f"Unknown modality={modality!r}; expected one of {self.modalities}") from exc

    def to_name(self, index: int) -> str:
        if index < 0 or index >= len(self.modalities):
            raise IndexError(f"Modality index out of range: {index}")
        return self.modalities[index]


if __name__ == "__main__":
    mapper = ModalityMapper.default()
    assert mapper.to_index("t1n") == 0
    assert mapper.to_index("t1c") == 1
    assert mapper.to_index("t2w") == 2
    assert mapper.to_index("t2f") == 3
    assert mapper.to_name(2) == "t2w"
    print("modality sanity check passed")
