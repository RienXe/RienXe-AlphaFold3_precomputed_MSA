# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Functions for running the MSA and template tools for the AlphaFold model."""

from concurrent import futures
import dataclasses
import datetime
import functools
import logging
import time

from data import folding_input
from data import msa_config
from utils import get_msa_path
from data.folding_input import _read_file as read_a3m_file
from pathlib import Path

# Cache to avoid re-running the MSA tools for the same sequence in homomers.
@functools.cache
def protein_blast(sequence:str):
  try:
    blastp_hits = get_msa_path.blastp_seq(sequence=sequence,blastp_config= blastp_config)

    print('result ',blastp_hits.stdout.strip().split("\n"))

    besthit = get_msa_path.get_thebest_hit(blastp_hits, id_cutoff=90.0)

    print('besthit' ,besthit)
  except Exception as e:
    print(e)
    besthit = []
    
  return besthit
    



# @functools.cache
def _get_protein_a3m_path(
  sequence: str,
  blastp_config: msa_config.BlastpConfig,
  reformat_config: msa_config.ReformatConfig,
) -> str:

    """Processes a single protein chain."""
    logging.info('Getting protein MSAs path for sequence %s', sequence)
    msa_start_time = time.time()

    besthit = protein_blast(sequence)

    if len(besthit) > 0 :
        list_key = blastp_config.outfmt.split(' ')[1:]

        print('list_key',list_key)
        dict_hits = get_msa_path.hits_to_dict(besthit, list_key)

        print(dict_hits)
        logging.info(f'{dict_hits}')
    else:
      dict_hits = {}

    if len(dict_hits)>0:
      path_msa = get_msa_path.get_msa_a3mpath(sequence = sequence,
                                              dict_hits = dict_hits,
                                              reformat_config = reformat_config,
                                              id_cutoff=90.0)
    else:
      path_msa = ''

    return path_msa


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class DataPipelineConfig:
    PATH_FAS_MSA:str
    MSA_mode:str

    #Blastp
    binary_blastp_script:str
    blastp_query: str
    blastp_db: str
    blastp_outfmt: str

    #Reformat
    binary_reformat_script: str
    reformat_target: str
    reformat_fas_path: str
    reformat_a3m_dir: str  

    #Output directory
    path_output_dir: str
    path_rewrotejson_dir:str
    path_completejson_dir:str
    path_slurm_logs:str

    path_slurm_logsab5000:str
    path_slurm_bash:str
    path_slurm_bash_cpu:str
    

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class ConfigureForSlurm:
    path_array_slurm_dir: str
    path_slurm_array: str
    path_exectued_jobs: str
    path_output_log: str
    path_error_log: str
    interval:int


class DataPipeline:
  """Runs the alignment tools and assembles the input features."""

  def __init__(self, data_pipeline_config: DataPipelineConfig):
    """Initializes the data pipeline with default configurations."""
    self._blastp_config = msa_config.BlastpConfig(
      binary_path = data_pipeline_config.binary_blastp_script,
      query = data_pipeline_config.blastp_query,
      database_config = data_pipeline_config.blastp_db,
      outfmt = data_pipeline_config.blastp_outfmt
    )

    self._reformat_config = msa_config.ReformatConfig(
      PATH_FAS_MSA = data_pipeline_config.PATH_FAS_MSA,
      binary_path = data_pipeline_config.binary_reformat_script,
      reformat_target = data_pipeline_config.reformat_target,
      fas_path = data_pipeline_config.reformat_fas_path,
      a3m_dir = data_pipeline_config.reformat_a3m_dir,
    )

  def process_protein_chain(
      self, chain: folding_input.ProteinChain
  ) -> folding_input.ProteinChain:
    """Processes a single protein chain."""
    has_unpaired_msa = chain.unpaired_msa is not None
    has_paired_msa = chain.paired_msa is not None
    has_templates = chain.templates is not None

    print(chain.sequence)
    if not has_unpaired_msa and not has_paired_msa :
      # MSA None - search. Templates either [] - don't search, or None - search.
      unpaired_msa_path = _get_protein_a3m_path(
          sequence=chain.sequence,
          blastp_config=self._blastp_config,
          reformat_config=self._reformat_config,
      )
      if unpaired_msa_path != '':
        unpaired_msa_path = Path(unpaired_msa_path)
        unpaired_msa = read_a3m_file(unpaired_msa_path,None)
        paired_msa = unpaired_msa
      else:
        unpaired_msa = None
        paired_msa = None

      # if templates has been decided
      if chain.templates:
        return folding_input.ProteinChain(
          id=chain.id,
          sequence=chain.sequence,
          ptms=chain.ptms,
          unpaired_msa=unpaired_msa,
          paired_msa=paired_msa,
          templates= chain.templates,)
      else:
        return folding_input.ProteinChain(
          id=chain.id,
          sequence=chain.sequence,
          ptms=chain.ptms,
          unpaired_msa=unpaired_msa,
          paired_msa=paired_msa,)
    else:
      return folding_input.ProteinChain(
        id=chain.id,
        sequence=chain.sequence,
        ptms=chain.ptms,
        unpaired_msa=chain.unpaired_msa,
        paired_msa=chain.paired_msa,
    )


  def process(self, fold_input: folding_input.Input) -> folding_input.Input:
    """Runs MSA and template tools and returns a new Input with the results."""
    processed_chains = []
    for chain in fold_input.chains:
      print(f'Running data pipeline for chain {chain.id}...')
      process_chain_start_time = time.time()
      match chain:
        case folding_input.ProteinChain():
          processed_chains.append(self.process_protein_chain(chain))
        case _:
          processed_chains.append(chain)
      print(
          f'Running data pipeline for chain {chain.id} took'
          f' {time.time() - process_chain_start_time:.2f} seconds',
      )

    return dataclasses.replace(fold_input, chains=processed_chains)
