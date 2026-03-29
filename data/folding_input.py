# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Model input dataclass."""

from collections.abc import Collection, Iterator, Mapping, Sequence
import functools
import dataclasses
import gzip
import json
import logging
import lzma
import pathlib
import random
import re
import string
from typing import Any, Final, Self, TypeAlias, cast

from constants import residue_names


BondAtomId: TypeAlias = tuple[str, int, str]

JSON_DIALECT: Final[str] = 'alphafold3'
JSON_VERSIONS: Final[tuple[int, ...]] = (1, 2)
JSON_VERSION: Final[int] = JSON_VERSIONS[-1]

ALPHAFOLDSERVER_JSON_DIALECT: Final[str] = 'alphafoldserver'
ALPHAFOLDSERVER_JSON_VERSION: Final[int] = 1

@functools.lru_cache(maxsize=256)
def int_id_to_str_id(num: int) -> str:
  """Encodes a number as a string, using reverse spreadsheet style naming.

  Args:
    num: A positive integer.

  Returns:
    A string that encodes the positive integer using reverse spreadsheet style,
    naming e.g. 1 = A, 2 = B, ..., 27 = AA, 28 = BA, 29 = CA, ... This is the
    usual way to encode chain IDs in mmCIF files.
  """
  if num <= 0:
    raise ValueError(f'Only positive integers allowed, got {num}.')

  num = num - 1  # 1-based indexing.
  output = []
  while num >= 0:
    output.append(chr(num % 26 + ord('A')))
    num = num // 26 - 1
  return ''.join(output)


def _validate_keys(actual: Collection[str], expected: Collection[str]):
  """Validates that the JSON doesn't contain any extra unwanted keys."""
  if bad_keys := set(actual) - set(expected):
    raise ValueError(f'Unexpected JSON keys in: {", ".join(sorted(bad_keys))}')


def _read_file(path: pathlib.Path, json_path: pathlib.Path | None) -> str:
  """Reads a maybe compressed (gzip, xz, zstd) file from the given path.

  Args:
    path: The path to the file to read. This can be either absolute path, or a
      path relative to the JSON file path.
    json_path: The path to the JSON file. If None, the path must be absolute.

  Returns:
    The contents of the file.
  """
  if not path.is_absolute():
    if json_path is None:
      raise ValueError('json_path must be specified if path is not absolute.')
    path = (json_path.parent / path).resolve()

  with open(path, 'rb') as f:
    first_six_bytes = f.read(6)
    f.seek(0)

    # Detect the compression type using the magic number in the header.
    if first_six_bytes[:2] == b'\x1f\x8b':
      with gzip.open(f, 'rt') as gzip_f:
        return cast(str, gzip_f.read())
    elif first_six_bytes == b'\xfd\x37\x7a\x58\x5a\x00':
      with lzma.open(f, 'rt') as xz_f:
        return cast(str, xz_f.read())
    else:
      return f.read().decode('utf-8')


class Template:
  """Structural template input."""

  __slots__ = ('_mmcif', '_query_to_template')

  def __init__(self, *, mmcif: str, query_to_template_map: Mapping[int, int]):
    """Initializes the template.

    Args:
      mmcif: The structural template in mmCIF format. The mmCIF should have only
        one protein chain.
      query_to_template_map: A mapping from query residue index to template
        residue index.
    """
    self._mmcif = mmcif
    # Needed to make the Template class hashable.
    self._query_to_template = tuple(query_to_template_map.items())

  @property
  def query_to_template_map(self) -> Mapping[int, int]:
    return dict(self._query_to_template)

  @property
  def mmcif(self) -> str:
    return self._mmcif

  def __hash__(self) -> int:
    return hash((self._mmcif, tuple(sorted(self._query_to_template))))

  def __eq__(self, other: Self) -> bool:
    mmcifs_equal = self._mmcif == other._mmcif
    maps_equal = sorted(self._query_to_template) == sorted(
        other._query_to_template
    )
    return mmcifs_equal and maps_equal


class ProteinChain:
  """Protein chain input."""

  __slots__ = (
      '_id',
      '_sequence',
      '_ptms',
      '_paired_msa',
      '_unpaired_msa',
      '_templates',
  )

  def __init__(
      self,
      *,
      id: str,  # pylint: disable=redefined-builtin
      sequence: str,
      ptms: Sequence[tuple[str, int]],
      paired_msa: str | None = None,
      unpaired_msa: str | None = None,
      templates: Sequence[Template] | None = None,
  ):
    """Initializes a single protein chain input.

    Args:
      id: Unique protein chain identifier.
      sequence: The amino acid sequence of the chain.
      ptms: A list of tuples containing the post-translational modification type
        and the (1-based) residue index where the modification is applied.
      paired_msa: Paired A3M-formatted MSA for this chain. This MSA is not
        deduplicated and will be used to compute paired features. If None, this
        field is unset and must be filled in by the data pipeline before
        featurisation. If set to an empty string, it will be treated as a custom
        MSA with no sequences.
      unpaired_msa: Unpaired A3M-formatted MSA for this chain. This will be
        deduplicated and used to compute unpaired features. If None, this field
        is unset and must be filled in by the data pipeline before
        featurisation. If set to an empty string, it will be treated as a custom
        MSA with no sequences.
      templates: A list of structural templates for this chain. If None, this
        field is unset and must be filled in by the data pipeline before
        featurisation. The list can be empty or contain up to 20 templates.
    """
    if not all(res.isalpha() for res in sequence):
      raise ValueError(f'Protein must contain only letters, got "{sequence}"')
    if any(not 0 < mod[1] <= len(sequence) for mod in ptms):
      raise ValueError(f'Invalid protein modification index: {ptms}')
    if any(mod[0].startswith('CCD_') for mod in ptms):
      raise ValueError(
          f'Protein ptms must not contain the "CCD_" prefix, got {ptms}'
      )
    # Use hashable containers for ptms and templates.
    self._id = id
    self._sequence = sequence
    self._ptms = tuple(ptms)
    self._paired_msa = paired_msa
    self._unpaired_msa = unpaired_msa
    self._templates = tuple(templates) if templates is not None else None

  @property
  def id(self) -> str:
    return self._id

  @property
  def sequence(self) -> str:
    """Returns a single-letter sequence, taking modifications into account.

    Uses 'X' for all unknown residues.
    """
    return ''.join([
        residue_names.letters_three_to_one(r, default='X')
        for r in self.to_ccd_sequence()
    ])

  @property
  def ptms(self) -> Sequence[tuple[str, int]]:
    return self._ptms

  @property
  def paired_msa(self) -> str | None:
    return self._paired_msa

  @property
  def unpaired_msa(self) -> str | None:
    return self._unpaired_msa

  @property
  def templates(self) -> Sequence[Template] | None:
    return self._templates

  def __len__(self) -> int:
    return len(self._sequence)

  def __eq__(self, other: Self) -> bool:
    return (
        self._id == other._id
        and self._sequence == other._sequence
        and self._ptms == other._ptms
        and self._paired_msa == other._paired_msa
        and self._unpaired_msa == other._unpaired_msa
        and self._templates == other._templates
    )

  def __hash__(self) -> int:
    return hash((
        self._id,
        self._sequence,
        self._ptms,
        self._paired_msa,
        self._unpaired_msa,
        self._templates,
    ))

  def hash_without_id(self) -> int:
    """Returns a hash ignoring the ID - useful for deduplication."""
    return hash((
        self._sequence,
        self._ptms,
        self._paired_msa,
        self._unpaired_msa,
        self._templates,
    ))

  @classmethod
  def from_alphafoldserver_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str
  ) -> Self:
    """Constructs ProteinChain from the AlphaFoldServer JSON dict."""
    _validate_keys(
        json_dict.keys(),
        {
            'sequence',
            'glycans',
            'modifications',
            'count',
            'maxTemplateDate',
            'useStructureTemplate',
        },
    )
    sequence = json_dict['sequence']

    if 'glycans' in json_dict:
      raise ValueError(
          f'Specifying glycans in the `{ALPHAFOLDSERVER_JSON_DIALECT}` format'
          ' is not supported.'
      )

    if 'maxTemplateDate' in json_dict:
      raise ValueError(
          f'Specifying maxTemplateDate in the `{ALPHAFOLDSERVER_JSON_DIALECT}`'
          ' format is not supported, use the --max_template_date flag instead.'
      )

    templates = None  # Search for templates unless explicitly disabled.
    if not json_dict.get('useStructureTemplate', True):
      templates = []  # Do not use any templates.

    ptms = [
        (mod['ptmType'].removeprefix('CCD_'), mod['ptmPosition'])
        for mod in json_dict.get('modifications', [])
    ]
    return cls(id=seq_id, sequence=sequence, ptms=ptms, templates=templates)

  @classmethod
  def from_dict(
      cls,
      json_dict: Mapping[str, Any],
      json_path: pathlib.Path | None = None,
      seq_id: str | None = None,
  ) -> Self:
    """Constructs ProteinChain from the AlphaFold JSON dict."""
    json_dict = json_dict['protein']
    _validate_keys(
        json_dict.keys(),
        {
            'id',
            'sequence',
            'modifications',
            'unpairedMsa',
            'unpairedMsaPath',
            'pairedMsa',
            'pairedMsaPath',
            'templates',
        },
    )

    sequence = json_dict['sequence']
    ptms = [
        (mod['ptmType'], mod['ptmPosition'])
        for mod in json_dict.get('modifications', [])
    ]

    unpaired_msa = json_dict.get('unpairedMsa', None)
    unpaired_msa_path = json_dict.get('unpairedMsaPath', None)
    if unpaired_msa and unpaired_msa_path:
      raise ValueError('Only one of unpairedMsa/unpairedMsaPath can be set.')
    elif unpaired_msa_path:
      unpaired_msa = _read_file(pathlib.Path(unpaired_msa_path), json_path)

    paired_msa = json_dict.get('pairedMsa', None)
    paired_msa_path = json_dict.get('pairedMsaPath', None)
    if paired_msa and paired_msa_path:
      raise ValueError('Only one of pairedMsa/pairedMsaPath can be set.')
    elif paired_msa_path:
      paired_msa = _read_file(pathlib.Path(paired_msa_path), json_path)

    raw_templates = json_dict.get('templates', None)

    return cls(
        id=seq_id or json_dict['id'],
        sequence=sequence,
        ptms=ptms,
        paired_msa=paired_msa,
        unpaired_msa=unpaired_msa,
        templates=raw_templates,
    )

  def to_dict(
      self, seq_id: str | Sequence[str] | None = None
  ) -> Mapping[str, Mapping[str, Any]]:
    """Converts ProteinChain to an AlphaFold JSON dict."""
    if self._templates is None:
      templates = None
    else:
      templates = [
          {
              'mmcif': template.mmcif,
              'queryIndices': list(template.query_to_template_map.keys()),
              'templateIndices': (
                  list(template.query_to_template_map.values()) or None
              ),
          }
          for template in self._templates
      ]
    contents = {
        'id': seq_id or self._id,
        'sequence': self._sequence,
        'modifications': [
            {'ptmType': ptm[0], 'ptmPosition': ptm[1]} for ptm in self._ptms
        ],
        'unpairedMsa': self._unpaired_msa,
        'pairedMsa': self._paired_msa,
        'templates': templates,
    }
    return {'protein': contents}

  def to_ccd_sequence(self) -> Sequence[str]:
    """Converts to a sequence of CCD codes."""
    ccd_coded_seq = [
        residue_names.PROTEIN_COMMON_ONE_TO_THREE.get(res, residue_names.UNK)
        for res in self._sequence
    ]
    for ptm_code, ptm_index in self._ptms:
      ccd_coded_seq[ptm_index - 1] = ptm_code
    return ccd_coded_seq

  def fill_missing_fields(self) -> Self:
    """Fill missing MSA and template fields with default values."""
    return ProteinChain(
        id=self.id,
        sequence=self._sequence,
        ptms=self._ptms,
        unpaired_msa=self._unpaired_msa or '',
        paired_msa=self._paired_msa or '',
        templates=self._templates or None,
    )


class RnaChain:
  """RNA chain input."""

  __slots__ = ('_id', '_sequence', '_modifications', '_unpaired_msa')

  def __init__(
      self,
      *,
      id: str,  # pylint: disable=redefined-builtin
      sequence: str,
      modifications: Sequence[tuple[str, int]],
      unpaired_msa: str | None = None,
  ):
    """Initializes a single strand RNA chain input.

    Args:
      id: Unique RNA chain identifier.
      sequence: The RNA sequence of the chain.
      modifications: A list of tuples containing the modification type and the
        (1-based) residue index where the modification is applied.
      unpaired_msa: Unpaired A3M-formatted MSA for this chain. This will be
        deduplicated and used to compute unpaired features. If None, this field
        is unset and must be filled in by the data pipeline before
        featurisation. If set to an empty string, it will be treated as a custom
        MSA with no sequences.
    """
    if not all(res.isalpha() for res in sequence):
      raise ValueError(f'RNA must contain only letters, got "{sequence}"')
    if any(not 0 < mod[1] <= len(sequence) for mod in modifications):
      raise ValueError(f'Invalid RNA modification index: {modifications}')
    if any(mod[0].startswith('CCD_') for mod in modifications):
      raise ValueError(
          'RNA modifications must not contain the "CCD_" prefix, got'
          f' {modifications}'
      )
    self._id = id
    self._sequence = sequence
    # Use hashable container for modifications.
    self._modifications = tuple(modifications)
    self._unpaired_msa = unpaired_msa

  @property
  def id(self) -> str:
    return self._id

  @property
  def sequence(self) -> str:
    """Returns a single-letter sequence, taking modifications into account.

    Uses 'N' for all unknown residues.
    """
    return ''.join([
        residue_names.letters_three_to_one(r, default='N')
        for r in self.to_ccd_sequence()
    ])

  @property
  def modifications(self) -> Sequence[tuple[str, int]]:
    return self._modifications

  @property
  def unpaired_msa(self) -> str | None:
    return self._unpaired_msa

  def __len__(self) -> int:
    return len(self._sequence)

  def __eq__(self, other: Self) -> bool:
    return (
        self._id == other._id
        and self._sequence == other._sequence
        and self._modifications == other._modifications
        and self._unpaired_msa == other._unpaired_msa
    )

  def __hash__(self) -> int:
    return hash(
        (self._id, self._sequence, self._modifications, self._unpaired_msa)
    )

  def hash_without_id(self) -> int:
    """Returns a hash ignoring the ID - useful for deduplication."""
    return hash((self._sequence, self._modifications, self._unpaired_msa))

  @classmethod
  def from_alphafoldserver_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str
  ) -> Self:
    """Constructs RnaChain from the AlphaFoldServer JSON dict."""
    _validate_keys(json_dict.keys(), {'sequence', 'modifications', 'count'})
    sequence = json_dict['sequence']
    modifications = [
        (mod['modificationType'].removeprefix('CCD_'), mod['basePosition'])
        for mod in json_dict.get('modifications', [])
    ]
    return cls(id=seq_id, sequence=sequence, modifications=modifications)

  @classmethod
  def from_dict(
      cls,
      json_dict: Mapping[str, Any],
      json_path: pathlib.Path | None = None,
      seq_id: str | None = None,
  ) -> Self:
    """Constructs RnaChain from the AlphaFold JSON dict."""
    json_dict = json_dict['rna']
    _validate_keys(
        json_dict.keys(),
        {'id', 'sequence', 'unpairedMsa', 'unpairedMsaPath', 'modifications'},
    )
    sequence = json_dict['sequence']
    modifications = [
        (mod['modificationType'], mod['basePosition'])
        for mod in json_dict.get('modifications', [])
    ]

    unpaired_msa = json_dict.get('unpairedMsa', None)
    unpaired_msa_path = json_dict.get('unpairedMsaPath', None)
    if unpaired_msa and unpaired_msa_path:
      raise ValueError('Only one of unpairedMsa/unpairedMsaPath can be set.')
    elif unpaired_msa_path:
      unpaired_msa = _read_file(pathlib.Path(unpaired_msa_path), json_path)

    return cls(
        id=seq_id or json_dict['id'],
        sequence=sequence,
        modifications=modifications,
        unpaired_msa=unpaired_msa,
    )

  def to_dict(
      self, seq_id: str | Sequence[str] | None = None
  ) -> Mapping[str, Mapping[str, Any]]:
    """Converts RnaChain to an AlphaFold JSON dict."""
    contents = {
        'id': seq_id or self._id,
        'sequence': self._sequence,
        'modifications': [
            {'modificationType': mod[0], 'basePosition': mod[1]}
            for mod in self._modifications
        ],
        'unpairedMsa': self._unpaired_msa,
    }
    return {'rna': contents}

  def to_ccd_sequence(self) -> Sequence[str]:
    """Converts to a sequence of CCD codes."""
    mapping = {r: r for r in residue_names.RNA_TYPES}  # Same 1-letter and CCD.
    ccd_coded_seq = [
        mapping.get(res, residue_names.UNK_RNA) for res in self._sequence
    ]
    for ccd_code, modification_index in self._modifications:
      ccd_coded_seq[modification_index - 1] = ccd_code
    return ccd_coded_seq

  def fill_missing_fields(self) -> Self:
    """Fill missing MSA fields with default values."""
    return RnaChain(
        id=self.id,
        sequence=self.sequence,
        modifications=self.modifications,
        unpaired_msa=self._unpaired_msa or '',
    )


class DnaChain:
  """Single strand DNA chain input."""

  __slots__ = ('_id', '_sequence', '_modifications')

  def __init__(
      self,
      *,
      id: str,  # pylint: disable=redefined-builtin
      sequence: str,
      modifications: Sequence[tuple[str, int]],
  ):
    """Initializes a single strand DNA chain input.

    Args:
      id: Unique DNA chain identifier.
      sequence: The DNA sequence of the chain.
      modifications: A list of tuples containing the modification type and the
        (1-based) residue index where the modification is applied.
    """
    if not all(res.isalpha() for res in sequence):
      raise ValueError(f'DNA must contain only letters, got "{sequence}"')
    if any(not 0 < mod[1] <= len(sequence) for mod in modifications):
      raise ValueError(f'Invalid DNA modification index: {modifications}')
    if any(mod[0].startswith('CCD_') for mod in modifications):
      raise ValueError(
          'DNA modifications must not contain the "CCD_" prefix, got'
          f' {modifications}'
      )
    self._id = id
    self._sequence = sequence
    # Use hashable container for modifications.
    self._modifications = tuple(modifications)

  @property
  def id(self) -> str:
    return self._id

  @property
  def sequence(self) -> str:
    """Returns a single-letter sequence, taking modifications into account.

    Uses 'N' for all unknown residues.
    """
    return ''.join([
        residue_names.letters_three_to_one(r, default='N')
        for r in self.to_ccd_sequence()
    ])

  def __len__(self) -> int:
    return len(self._sequence)

  def __eq__(self, other: Self) -> bool:
    return (
        self._id == other._id
        and self._sequence == other._sequence
        and self._modifications == other._modifications
    )

  def __hash__(self) -> int:
    return hash((self._id, self._sequence, self._modifications))

  def modifications(self) -> Sequence[tuple[str, int]]:
    return self._modifications

  def hash_without_id(self) -> int:
    """Returns a hash ignoring the ID - useful for deduplication."""
    return hash((self._sequence, self._modifications))

  @classmethod
  def from_alphafoldserver_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str
  ) -> Self:
    """Constructs DnaChain from the AlphaFoldServer JSON dict."""
    _validate_keys(json_dict.keys(), {'sequence', 'modifications', 'count'})
    sequence = json_dict['sequence']
    modifications = [
        (mod['modificationType'].removeprefix('CCD_'), mod['basePosition'])
        for mod in json_dict.get('modifications', [])
    ]
    return cls(id=seq_id, sequence=sequence, modifications=modifications)

  @classmethod
  def from_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str | None = None
  ) -> Self:
    """Constructs DnaChain from the AlphaFold JSON dict."""
    json_dict = json_dict['dna']
    _validate_keys(json_dict.keys(), {'id', 'sequence', 'modifications'})
    sequence = json_dict['sequence']
    modifications = [
        (mod['modificationType'], mod['basePosition'])
        for mod in json_dict.get('modifications', [])
    ]
    return cls(
        id=seq_id or json_dict['id'],
        sequence=sequence,
        modifications=modifications,
    )

  def to_dict(
      self, seq_id: str | Sequence[str] | None = None
  ) -> Mapping[str, Mapping[str, Any]]:
    """Converts DnaChain to an AlphaFold JSON dict."""
    contents = {
        'id': seq_id or self._id,
        'sequence': self._sequence,
        'modifications': [
            {'modificationType': mod[0], 'basePosition': mod[1]}
            for mod in self._modifications
        ],
    }
    return {'dna': contents}

  def to_ccd_sequence(self) -> Sequence[str]:
    """Converts to a sequence of CCD codes."""
    ccd_coded_seq = [
        residue_names.DNA_COMMON_ONE_TO_TWO.get(res, residue_names.UNK_DNA)
        for res in self._sequence
    ]
    for ccd_code, modification_index in self._modifications:
      ccd_coded_seq[modification_index - 1] = ccd_code
    return ccd_coded_seq


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Ligand:
  """Ligand input.

  Attributes:
    id: Unique ligand "chain" identifier.
    ccd_ids: The Chemical Component Dictionary or user-defined CCD IDs of the
      chemical components of the ligand. Typically, this is just a single ID,
      but some ligands are composed of multiple components. If that is the case,
      a bond linking these components should be added to the bonded_atom_pairs
      Input field.
    smiles: The SMILES representation of the ligand.
  """

  id: str
  ccd_ids: Sequence[str] | None = None
  smiles: str | None = None

  def __post_init__(self):
    if (self.ccd_ids is None) == (self.smiles is None):
      raise ValueError('Ligand must have one of CCD ID or SMILES set.')

    # Use hashable types for ccd_ids.
    if self.ccd_ids is not None:
      object.__setattr__(self, 'ccd_ids', tuple(self.ccd_ids))

  def __len__(self) -> int:
    if self.ccd_ids is not None:
      return len(self.ccd_ids)
    else:
      return 1

  def hash_without_id(self) -> int:
    """Returns a hash ignoring the ID - useful for deduplication."""
    return hash((self.ccd_ids, self.smiles))

  @classmethod
  def from_alphafoldserver_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str
  ) -> Self:
    """Constructs Ligand from the AlphaFoldServer JSON dict."""
    # Ligand can be specified either as a ligand, or ion (special-case).
    _validate_keys(json_dict.keys(), {'ligand', 'ion', 'count'})
    if 'ligand' in json_dict:
      return cls(id=seq_id, ccd_ids=[json_dict['ligand'].removeprefix('CCD_')])
    elif 'ion' in json_dict:
      return cls(id=seq_id, ccd_ids=[json_dict['ion']])
    else:
      raise ValueError(f'Unknown ligand type: {json_dict}')

  @classmethod
  def from_dict(
      cls, json_dict: Mapping[str, Any], seq_id: str | None = None
  ) -> Self:
    """Constructs Ligand from the AlphaFold JSON dict."""
    json_dict = json_dict['ligand']
    _validate_keys(json_dict.keys(), {'id', 'ccdCodes', 'smiles'})
    if json_dict.get('ccdCodes') and json_dict.get('smiles'):
      raise ValueError(
          'Ligand cannot have both CCD code and SMILES set at the same time, '
          f'got CCD: {json_dict["ccdCode"]} and SMILES: {json_dict["smiles"]}'
      )

    if 'ccdCodes' in json_dict:
      return cls(id=seq_id or json_dict['id'], ccd_ids=json_dict['ccdCodes'])
    elif 'smiles' in json_dict:
      return cls(id=seq_id or json_dict['id'], smiles=json_dict['smiles'])
    else:
      raise ValueError(f'Unknown ligand type: {json_dict}')

  def to_dict(
      self, seq_id: str | Sequence[str] | None = None
  ) -> Mapping[str, Mapping[str, Any]]:
    """Converts Ligand to an AlphaFold JSON dict."""
    contents = {'id': seq_id or self.id}
    if self.ccd_ids is not None:
      contents['ccdCodes'] = self.ccd_ids
    if self.smiles is not None:
      contents['smiles'] = self.smiles
    return {'ligand': contents}


def _sample_rng_seed() -> int:
  """Sample a random seed for AlphaFoldServer job."""
  # See https://alphafoldserver.com/faq#what-are-seeds-and-how-are-they-set.
  return random.randint(0, 2**32 - 1)


def _validate_user_ccd_keys(keys: Sequence[str], component_name: str) -> None:
  """Validates the keys of the user-defined CCD dictionary."""
  mandatory_keys = (
      '_chem_comp.id',
      '_chem_comp.name',
      '_chem_comp.type',
      '_chem_comp.formula',
      '_chem_comp.mon_nstd_parent_comp_id',
      '_chem_comp.pdbx_synonyms',
      '_chem_comp.formula_weight',
      '_chem_comp_atom.comp_id',
      '_chem_comp_atom.atom_id',
      '_chem_comp_atom.type_symbol',
      '_chem_comp_atom.charge',
      '_chem_comp_atom.pdbx_model_Cartn_x_ideal',
      '_chem_comp_atom.pdbx_model_Cartn_y_ideal',
      '_chem_comp_atom.pdbx_model_Cartn_z_ideal',
      '_chem_comp_bond.atom_id_1',
      '_chem_comp_bond.atom_id_2',
      '_chem_comp_bond.value_order',
      '_chem_comp_bond.pdbx_aromatic_flag',
  )
  if missing_keys := set(mandatory_keys) - set(keys):
    raise ValueError(
        f'Component {component_name} in the user-defined CCD is missing these'
        f' keys: {missing_keys}'
    )


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class Input:
  """AlphaFold input.

  Attributes:
    name: The name of the target.
    chains: Protein chains, RNA chains, DNA chains, or ligands.
    protein_chains: Protein chains.
    rna_chains: RNA chains.
    dna_chains: Single strand DNA chains.
    ligands: Ligand (including ion) inputs.
    rng_seeds: Random number generator seeds, one for each model execution.
    bonded_atom_pairs: A list of tuples of atoms that are bonded to each other.
      Each atom is defined by a tuple of (chain_id, res_id, atom_name). Chain
      IDs must be set if there are any bonded atoms. Residue IDs are 1-indexed.
      Atoms in ligands defined by SMILES can't be bonded since SMILES doesn't
      define unique atom names.
    user_ccd: Optional user-defined chemical component dictionary in the CIF
      format. This can be used to provide additional CCD entries that are not
      present in the default CCD and thus define arbitrary new ligands. This is
      more expressive than SMILES since it allows to name all atoms within the
      ligand which in turn makes it possible to define bonds using those atoms.
  """

  name: str
  chains: Sequence[ProteinChain | RnaChain | DnaChain | Ligand]
  rng_seeds: Sequence[int]
  bonded_atom_pairs: Sequence[tuple[BondAtomId, BondAtomId]] | None = None
  user_ccd: str | None = None

  def __post_init__(self):
    if not self.rng_seeds:
      raise ValueError('Input must have at least one RNG seed.')

    if not self.name.strip() or not self.sanitised_name():
      raise ValueError(
          'Input name must be non-empty and contain at least one valid'
          ' character (letters, numbers, dots, dashes, underscores).'
      )

    chain_ids = [c.id for c in self.chains]
    if any(not c.id.isalpha() or c.id.islower() for c in self.chains):
      raise ValueError(f'IDs must be upper case letters, got: {chain_ids}')
    if len(set(chain_ids)) != len(chain_ids):
      raise ValueError('Input JSON contains sequences with duplicate IDs.')

    # Use hashable types for chains, rng_seeds, and bonded_atom_pairs.
    object.__setattr__(self, 'chains', tuple(self.chains))
    object.__setattr__(self, 'rng_seeds', tuple(self.rng_seeds))
    if self.bonded_atom_pairs is not None:
      object.__setattr__(
          self, 'bonded_atom_pairs', tuple(self.bonded_atom_pairs)
      )

  @property
  def protein_chains(self) -> Sequence[ProteinChain]:
    return [chain for chain in self.chains if isinstance(chain, ProteinChain)]

  @property
  def rna_chains(self) -> Sequence[RnaChain]:
    return [chain for chain in self.chains if isinstance(chain, RnaChain)]

  @property
  def dna_chains(self) -> Sequence[DnaChain]:
    return [chain for chain in self.chains if isinstance(chain, DnaChain)]

  @property
  def ligands(self) -> Sequence[Ligand]:
    return [chain for chain in self.chains if isinstance(chain, Ligand)]

  def sanitised_name(self) -> str:
    """Returns sanitised version of the name that can be used as a filename."""
    lower_spaceless_name = self.name.lower().replace(' ', '_')
    allowed_chars = set(string.ascii_lowercase + string.digits + '_-.')
    return ''.join(l for l in lower_spaceless_name if l in allowed_chars)

  @classmethod
  def from_alphafoldserver_fold_job(cls, fold_job: Mapping[str, Any]) -> Self:
    """Constructs Input from an AlphaFoldServer fold job."""

    # Validate the fold job has the correct format.
    _validate_keys(
        fold_job.keys(),
        {'name', 'modelSeeds', 'sequences', 'dialect', 'version'},
    )
    if 'dialect' not in fold_job and 'version' not in fold_job:
      dialect = ALPHAFOLDSERVER_JSON_DIALECT
      version = ALPHAFOLDSERVER_JSON_VERSION
    elif 'dialect' in fold_job and 'version' in fold_job:
      dialect = fold_job['dialect']
      version = fold_job['version']
    else:
      raise ValueError(
          'AlphaFold Server input JSON must either contain both `dialect` and'
          ' `version` fields, or neither. If neither is specified, it is'
          f' assumed that `dialect="{ALPHAFOLDSERVER_JSON_DIALECT}"` and'
          f' `version="{ALPHAFOLDSERVER_JSON_VERSION}"`.'
      )

    if dialect != ALPHAFOLDSERVER_JSON_DIALECT:
      raise ValueError(
          f'AlphaFold Server input JSON has unsupported dialect: {dialect}, '
          f'expected {ALPHAFOLDSERVER_JSON_DIALECT}.'
      )

    # For now, there is only one AlphaFold Server JSON version.
    if version != ALPHAFOLDSERVER_JSON_VERSION:
      raise ValueError(
          f'AlphaFold Server input JSON has unsupported version: {version}, '
          f'expected {ALPHAFOLDSERVER_JSON_VERSION}.'
      )

    # Parse the chains.
    chains = []
    for sequence in fold_job['sequences']:
      if 'proteinChain' in sequence:
        for _ in range(sequence['proteinChain'].get('count', 1)):
          chains.append(
              ProteinChain.from_alphafoldserver_dict(
                  sequence['proteinChain'],
                  seq_id=int_id_to_str_id(len(chains) + 1),
              )
          )
      elif 'rnaSequence' in sequence:
        for _ in range(sequence['rnaSequence'].get('count', 1)):
          chains.append(
              RnaChain.from_alphafoldserver_dict(
                  sequence['rnaSequence'],
                  seq_id=int_id_to_str_id(len(chains) + 1),
              )
          )
      elif 'dnaSequence' in sequence:
        for _ in range(sequence['dnaSequence'].get('count', 1)):
          chains.append(
              DnaChain.from_alphafoldserver_dict(
                  sequence['dnaSequence'],
                  seq_id=int_id_to_str_id(len(chains) + 1),
              )
          )
      elif 'ion' in sequence:
        for _ in range(sequence['ion'].get('count', 1)):
          chains.append(
              Ligand.from_alphafoldserver_dict(
                  sequence['ion'],
                  seq_id=int_id_to_str_id(len(chains) + 1),
              )
          )
      elif 'ligand' in sequence:
        for _ in range(sequence['ligand'].get('count', 1)):
          chains.append(
              Ligand.from_alphafoldserver_dict(
                  sequence['ligand'],
                  seq_id=int_id_to_str_id(len(chains) + 1),
              )
          )
      else:
        raise ValueError(f'Unknown sequence type: {sequence}')

    if 'modelSeeds' in fold_job and fold_job['modelSeeds']:
      rng_seeds = [int(seed) for seed in fold_job['modelSeeds']]
    else:
      rng_seeds = [_sample_rng_seed()]

    return cls(name=fold_job['name'], chains=chains, rng_seeds=rng_seeds)

  @classmethod
  def from_json(
      cls, json_str: str, json_path: pathlib.Path | None = None
  ) -> Self:
    """Loads the input from the AlphaFold JSON string."""
    raw_json = json.loads(json_str)

    _validate_keys(
        raw_json.keys(),
        {
            'dialect',
            'version',
            'name',
            'modelSeeds',
            'sequences',
            'bondedAtomPairs',
            'userCCD',
        },
    )

    if 'dialect' not in raw_json or 'version' not in raw_json:
      raise ValueError(
          'AlphaFold 3 input JSON must contain `dialect` and `version` fields.'
      )

    if raw_json['dialect'] != JSON_DIALECT:
      raise ValueError(
          'AlphaFold 3 input JSON has unsupported dialect:'
          f' {raw_json["dialect"]}, expected {JSON_DIALECT}.'
      )

    if raw_json['version'] not in JSON_VERSIONS:
      raise ValueError(
          'AlphaFold 3 input JSON has unsupported version:'
          f' {raw_json["version"]}, expected one of {JSON_VERSIONS}.'
      )

    if 'sequences' not in raw_json:
      raise ValueError('AlphaFold 3 input JSON does not contain any sequences.')

    if 'modelSeeds' not in raw_json or not raw_json['modelSeeds']:
      raise ValueError(
          'AlphaFold 3 input JSON must specify at least one rng seed in'
          ' `modelSeeds`.'
      )

    sequences = raw_json['sequences']

    # Make sure sequence IDs are all set.
    raw_sequence_ids = [next(iter(s.values())).get('id') for s in sequences]
    if all(raw_sequence_ids):
      sequence_ids = []
      for sequence_id in raw_sequence_ids:
        if isinstance(sequence_id, list):
          sequence_ids.append(sequence_id)
        else:
          sequence_ids.append([sequence_id])
    else:
      raise ValueError(
          'AlphaFold 3 input JSON contains sequences with unset IDs.'
      )

    flat_seq_ids = []
    for seq_ids in sequence_ids:
      flat_seq_ids.extend(seq_ids)

    chains = []
    for seq_ids, sequence in zip(sequence_ids, sequences, strict=True):
      if len(sequence) != 1:
        raise ValueError(f'Chain {seq_ids} has more than 1 sequence.')
      for seq_id in seq_ids:
        if 'protein' in sequence:
          chains.append(ProteinChain.from_dict(sequence, json_path, seq_id))
        elif 'rna' in sequence:
          chains.append(RnaChain.from_dict(sequence, json_path, seq_id))
        elif 'dna' in sequence:
          chains.append(DnaChain.from_dict(sequence, seq_id=seq_id))
        elif 'ligand' in sequence:
          chains.append(Ligand.from_dict(sequence, seq_id=seq_id))
        else:
          raise ValueError(f'Unknown sequence type: {sequence}')

    smiles_ligand_ids = set(
        c.id for c in chains if isinstance(c, Ligand) and c.smiles is not None
    )
    chain_lengths = {chain.id: len(chain) for chain in chains}
    bonded_atom_pairs = None
    if bonds := raw_json.get('bondedAtomPairs'):
      bonded_atom_pairs = []
      for bond in bonds:
        if len(bond) != 2:
          raise ValueError(f'Bond {bond} must have 2 atoms, got {len(bond)}.')
        bond_beg, bond_end = bond
        if (
            len(bond_beg) != 3
            or not isinstance(bond_beg[0], str)
            or not isinstance(bond_beg[1], int)
            or not isinstance(bond_beg[2], str)
        ):
          raise ValueError(
              f'Atom {bond_beg} in bond {bond} must have 3 components: '
              '(chain_id: str, res_id: int, atom_name: str).'
          )
        if (
            len(bond_end) != 3
            or not isinstance(bond_end[0], str)
            or not isinstance(bond_end[1], int)
            or not isinstance(bond_end[2], str)
        ):
          raise ValueError(
              f'Atom {bond_end} in bond {bond} must have 3 components: '
              '(chain_id: str, res_id: int, atom_name: str).'
          )
        if bond_beg[0] not in flat_seq_ids or bond_end[0] not in flat_seq_ids:
          raise ValueError(f'Invalid chain ID(s) in bond {bond}')
        if (
            not 0 < bond_beg[1] <= chain_lengths[bond_beg[0]]
            or not 0 < bond_end[1] <= chain_lengths[bond_end[0]]
        ):
          raise ValueError(f'Invalid residue ID(s) in bond {bond}')
        if bond_beg[0] in smiles_ligand_ids:
          raise ValueError(
              f'Bond {bond} involves an unsupported SMILES ligand {bond_beg[0]}'
          )
        if bond_end[0] in smiles_ligand_ids:
          raise ValueError(
              f'Bond {bond} involves an unsupported SMILES ligand {bond_end[0]}'
          )
        bonded_atom_pairs.append((tuple(bond_beg), tuple(bond_end)))

      if len(bonded_atom_pairs) != len(set(bonded_atom_pairs)):
        raise ValueError(f'Bonds are not unique: {bonded_atom_pairs}')

    return cls(
        name=raw_json['name'],
        chains=chains,
        rng_seeds=[int(seed) for seed in raw_json['modelSeeds']],
        bonded_atom_pairs=bonded_atom_pairs,
        user_ccd=raw_json.get('userCCD'),
    )


  def to_json(self) -> str:
    """Converts Input to an AlphaFold JSON."""
    deduped_chains = {}
    deduped_chain_ids = {}
    for chain in self.chains:
      print('chain', chain)
      print('hash no id',chain.hash_without_id())
      deduped_chains[chain.hash_without_id()] = chain
      deduped_chain_ids.setdefault(chain.hash_without_id(), []).append(chain.id)

    sequences = []
    for chain_content_hash, ids in deduped_chain_ids.items():
      chain = deduped_chains[chain_content_hash]
      sequences.append(chain.to_dict(seq_id=ids if len(ids) > 1 else ids[0]))

    alphafold_json = json.dumps(
        {
            'dialect': JSON_DIALECT,
            'version': JSON_VERSION,
            'name': self.name,
            'sequences': sequences,
            'modelSeeds': self.rng_seeds,
            'bondedAtomPairs': self.bonded_atom_pairs,
            'userCCD': self.user_ccd,
        },
        indent=2,
    )
    # Remove newlines from the query/template indices arrays. We match the
    # queryIndices/templatesIndices with a non-capturing group. We then match
    # the entire region between the square brackets by looking for lines
    # containing only whitespace, number, or a comma.
    return re.sub(
        r'("(?:queryIndices|templateIndices)": \[)([\s\n\d,]+)(\],?)',
        lambda mtch: mtch[1] + re.sub(r'\n\s+', ' ', mtch[2].strip()) + mtch[3],
        alphafold_json,
    )

  def fill_missing_fields(self) -> Self:
    """Fill missing MSA and template fields with default values."""
    with_missing_fields = [
        c.fill_missing_fields()
        if isinstance(c, (ProteinChain, RnaChain))
        else c
        for c in self.chains
    ]
    return dataclasses.replace(self, chains=with_missing_fields)

  def with_multiple_seeds(self, num_seeds: int) -> Self:
    """Returns a copy of the input with num_seeds rng seeds."""
    if num_seeds <= 1:
      raise ValueError('Number of seeds must be greater than 1.')
    if len(self.rng_seeds) != 1:
      raise ValueError('Input must have one rng seed to set multiple seeds.')

    return dataclasses.replace(
        self,
        rng_seeds=list(range(self.rng_seeds[0], self.rng_seeds[0] + num_seeds)),
    )


def load_fold_inputs_from_path(json_path: pathlib.Path, output_path: pathlib.Path) -> Iterator[Input]:
  """Loads multiple fold inputs from a JSON string."""
  with open(json_path, 'r') as f:
    json_str = f.read()

  # Parse the JSON string, so we can detect its format.
  raw_json = json.loads(json_str)

  if isinstance(raw_json, list):
    # AlphaFold Server JSON.
    logging.info('Loading %d fold jobs from %s', len(raw_json), json_path)
    for fold_job_idx, fold_job in enumerate(raw_json):
      try:
        result_foldjob = Input.from_alphafoldserver_fold_job(fold_job)
        job_name = result_foldjob.name.lower()
        path_possible_output = output_path / f'{job_name}'
        path_possible_complete = path_possible_output/f'{job_name}_model.cif'
        print('path_possible_complete', path_possible_complete)
        if path_possible_complete.exists():
          logging.info('Output file %s already exists, skipping', path_possible_complete)
          print(f'Output file {path_possible_complete} already exists, skipping')
          continue
        else:
          yield result_foldjob
        # yield Input.from_alphafoldserver_fold_job(fold_job)
      except ValueError as e:
        raise ValueError(
            f'Failed to load fold job {fold_job_idx} from {json_path}'
            f' (AlphaFold Server dialect): {e}'
        ) from e
  else:
    # AlphaFold 3 JSON.
    try:
      result_foldjob = Input.from_json(json_str, json_path)
      path_possible_output = output_path / f'{result_foldjob.name}'
      path_possible_complete = path_possible_output/f'{result_foldjob.name}_model.cif'
      if path_possible_complete.exists():
        logging.info('Output file %s already exists, skipping', path_possible_complete)
      else:
        yield result_foldjob
      # yield Input.from_json(json_str, json_path)
    except ValueError as e:
      raise ValueError(
          f'Failed to load input from {json_path} (AlphaFold 3 dialect): {e}'
      ) from e


def load_fold_inputs_from_dir(input_dir: pathlib.Path, output_path: pathlib.Path) -> Iterator[Input]:
  """Loads multiple fold inputs from all JSON files in a given input_dir.

  Args:
    input_dir: The directory containing the JSON files.

  Yields:
    The fold inputs from all JSON files in the input directory.
  """
  for file_path in sorted(input_dir.glob('*.json')):
    if not file_path.is_file():
      continue

    yield from load_fold_inputs_from_path(file_path, output_path)