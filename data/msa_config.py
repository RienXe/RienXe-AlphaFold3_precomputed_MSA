import dataclasses
import datetime
from typing import Self


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class BlastpConfig:

  binary_path: str
  query: str
  database_config: str
  outfmt:str

@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ReformatConfig:
  PATH_FAS_MSA: str
  binary_path: str
  reformat_target: str
  fas_path: str
  a3m_dir:str