import functools
import subprocess
import tempfile
import os
import parasail
from utils import fastapy
from collections.abc import Mapping, Sequence
from data import msa_config
from utils import subprocess_utils
from pathlib import Path
import logging
import json

def modify_sequence(seq2):
    # Handle leading lowercase and '-' characters
    start_index = 0
    for char in seq2:
        if char.islower() or char == '-':
            start_index += 1
        else:
            break

    # Handle trailing lowercase and '-' characters
    end_index = len(seq2)
    for char in reversed(seq2):
        if char.islower() or char == '-':
            end_index -= 1
        else:
            break

    # Create the modified sequence
    modified_seq2 = '-' * start_index + seq2[start_index:end_index] + '-' * (len(seq2) - end_index)
    return modified_seq2

# Load JSON file
def load_json_file(file_path):
    """Load a JSON file and return its content."""
    with open(file_path, 'r') as file:
        return json.load(file)


def align_pair(seq_query, seq_ref):
    # Define gap penalties
    gap_open = 20
    gap_extend = 1
    seq_ref = seq_ref.replace('-', '').upper()

    # Perform a global alignment (Needleman–Wunsch) with traceback
    result = parasail.sg_qe_de_trace(seq_query, seq_ref, gap_open, gap_extend, parasail.blosum62)

    aligned_seq1 = result.traceback.query
    aligned_seq2 = result.traceback.ref
    # Identify continuous leading gaps in seq1
    start_index = len(aligned_seq1) - len(aligned_seq1.lstrip('-'))  # First non-gap character

    # Identify continuous trailing gaps in seq1
    end_index = len(aligned_seq1.rstrip('-'))  # Last non-gap character + 1

    # Trim seq2 only based on continuous gaps at the start or end
    trimmed_seq2 = aligned_seq2[start_index:end_index]
    trimmed_seq2 = modify_sequence(trimmed_seq2)
    return aligned_seq1, trimmed_seq2


def find_nongap_indices(s:str) -> Sequence[str]:
    """
    Return a list of indices where s has a non-gap character.
    """
    return [i for i, c in enumerate(s) if c != '-']



def extra_gaps_info(s1:str, s2:str) -> Sequence[tuple[int,int]] :
    """
    Given two aligned strings s1 and s2 (with the same ungapped sequence),
    compute where extra gaps (i.e. '-' characters) were added in s2 compared
    to s1. Returns a list of tuples (index, count) where:
      - index: the column in s2 at which extra gaps were inserted
      - count: how many extra '-' characters were inserted at that position.
    
    This function checks leading, internal, and trailing regions.
    """
    # Get the positions of the real residues in both alignments.
    indices1 = find_nongap_indices(s1)
    indices2 = find_nongap_indices(s2)
    
    # Sanity check: both must yield the same number of residues.
    if len(indices1) != len(indices2):
        raise ValueError("The two strings do not have the same underlying sequence!")
    
    gap_info = []
    
    # --- Leading gaps (before the first real residue) ---
    lead_gap_s1 = indices1[0]
    lead_gap_s2 = indices2[0]
    if lead_gap_s2 > lead_gap_s1:
        gap_info.append((0, lead_gap_s2 - lead_gap_s1))
    
    # --- Internal gaps: compare the positions of each residue ---
    # For each residue (after the first) the difference between indices in s2 and s1
    # tells you how many extra gap columns have been introduced before that residue.
    prev_diff = indices2[0] - indices1[0]
    for k in range(1, len(indices1)):
        diff = indices2[k] - indices1[k]
        if diff != prev_diff:
            # When diff increases, extra gaps have been inserted.
            # We assume extra gaps occur in a contiguous block just before the k-th residue.
            extra = diff - prev_diff
            if extra > 0:
                # The extra gaps are at the position in s2 where the k-th residue appears,
                # but they were inserted just before that residue.
                gap_index = indices2[k] - extra
                gap_info.append((gap_index, extra))
            # Update prev_diff to the new value.
            prev_diff = diff
    
    # --- Trailing gaps (after the last real residue) ---
    trail_gap_s1 = len(s1) - indices1[-1] - 1  # gaps after last residue in s1
    trail_gap_s2 = len(s2) - indices2[-1] - 1  # gaps after last residue in s2
    if trail_gap_s2 > trail_gap_s1:
        # Extra trailing gaps start right after the last residue in s2.
        gap_info.append((indices2[-1] + 1, trail_gap_s2 - trail_gap_s1))
    
    return gap_info

def get_refseq(path_msa:str) -> str:
    msa_records = list(fastapy.parse(path_msa))
    if not msa_records:
        raise ValueError("No sequences found in the MSA file.")

    for record in msa_records:
        return str(record.seq)

@functools.cache
def realign_process(path_msa:str, seq_query:str) -> list[fastapy.Record]:
    path_json_dict = os.path.abspath('./constants/converters/organism_mnemonic.json')
    print(path_json_dict)
    dict_organism = load_json_file(path_json_dict)

    # --- STEP 1: READ THE MSA ---
    # Parse the jackhmmer MSA (assumed to be in FASTA format)
    msa_records = list(fastapy.parse(path_msa))
    if not msa_records:
        raise ValueError("No sequences found in the MSA file.")

    records = [fastapy.Record(seq = seq_query, id='query', desc = 'query' )]

    for record in msa_records:
        new_align = ''
        new_align = record.seq
        record_id = record.id
        record_id_new = record_id.strip().replace('_', '')[:10]
        record_desc = record.desc
        organism = record_desc.split(':')[0].strip()
        if organism == 'na':
            organism = record_desc.split(':')[1].strip()
        elif organism not in dict_organism and record_id != 'query':
            print('===============================================================================================')
            print(f"Organism '{organism}' not found in the dictionary.")
            organism = record_desc.split(':')[1].strip()
            print(f"Using alternative organism name: '{organism}'")
            # break
        
        SpeciesID = dict_organism.get(organism, '')
        record_desc_new = f"sp|{record_id_new}|{SpeciesID}_{SpeciesID}|{record_desc}"
        # print(record_id[:10])
        # print(record_desc)
        # print(organism)
        # print(SpeciesID)
        # print(record_desc_new)
        if record_desc_new:
            record_id = record_desc_new
            record.desc = record_desc_new
        else:
            record_id = record_id
            record.desc = record_desc
        # for index, count in gaps:
        #     to_insert = ''.join(["-"]*count)
        #     new_align = new_align[:index] + to_insert + new_align[index:]

        # records.append(fastapy.Record(seq = new_align, id=record.id,  desc = record.desc))            
        _, trimmed_ref = align_pair(seq_query, new_align)
        # print(record.id,'trimmed_seq',trimmed_ref)
        records.append(fastapy.Record(seq = trimmed_ref, id=record_desc_new,  desc = record_desc_new))
    return records



def write_realign(path_msa:str, seq_query:str, path_output:str) -> None:
    records = realign_process(path_msa, seq_query)
    fastapy.write(path_output, records)
    
    #To check the MSA before and after convert to a3m
    # path_name = Path(path_output).name
    # fastapy.write(f'/mnt/DATA/ren/alphafold/alphafold3/process/temp_fas/{path_name}', records)
    

# path_msa = '/4TB/uniprot_human/reviewed/profile/Q9Y6Y8.fas'
# output_file = '/4TB/uniprot_human/reviewed/profile/Q9Y6Y8_new.fas'



def blastp_seq(sequence:str,
               blastp_config: msa_config.BlastpConfig,
               ):
    
    """
    Run BLASTP on a given query sequence and return hits with 100% identity.
    
    Parameters:
        query_sequence (str): Protein sequence (amino acids) as a string.
        db (str): BLAST database to search against.
        blastp_path (str): Path to the BLASTP executable (default assumes it's in your PATH).
        additional_options (list): Optional list of additional command-line options for BLASTP.
        
    Returns:
        list: A list of hits (each hit is represented as a list of fields) where the
              percent identity (pident) is 100.0. Returns an empty list if no such hit is found.
    """
    # Write the query sequence to a temporary FASTA file
    with tempfile.TemporaryDirectory() as query_tmp_dir:
        input_fasta_path = os.path.join(query_tmp_dir, 'query.fasta')
        subprocess_utils.create_query_fasta_file(
            sequence=sequence, path=input_fasta_path
        )

        # Build the BLASTP command line
        command = [blastp_config.binary_path,
                "-query", input_fasta_path,
                "-db", blastp_config.database_config,
                "-outfmt", blastp_config.outfmt,]

        print('command',command)
        # Run BLASTP
        try:
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # If BLASTP returns an error, print it and clean up the temporary file
            print("BLASTP execution failed with error:")
            print(e.stderr)
            # os.remove(input_fasta_path)
            return []
        # # Clean up the temporary query file
        # os.remove(input_fasta_path)

    return result

def get_thebest_hit(blastp_hits,id_cutoff=90) -> Sequence[str]:
    # Parse the BLAST output and filter for 100% identity matches
    perfect_hits = []
    for line in blastp_hits.stdout.strip().split("\n"):
        if not line:
            continue  # skip empty lines
        fields = line.split("\t")
        # print(fields)
        try:
            pident = float(fields[2])
        except (ValueError, IndexError):
            continue  # skip if conversion fails or the line doesn't have enough fields
        
        if pident >= id_cutoff:
            # Optionally, you could also check that the alignment length covers the full query.
            perfect_hits.append(fields)
            return perfect_hits
    return perfect_hits


def hits_to_dict(hits:Sequence[str], keys:Sequence[str]) -> Mapping[str,str]: 
    dict_hits = {}
    for hit in hits:
        sseqid = hit[1]
        if '|' in sseqid:
            parts = sseqid.split('|')
            # Expecting format like "sp|P12345|ENTRY_NAME" or "tr|P12345|ENTRY_NAME"
            if len(parts) >= 2:
                uniprot_id = parts[1]
            else:
                uniprot_id = sseqid
        else:
            uniprot_id = sseqid
            
        uniprot_id = uniprot_id.upper()
        dict_hits[uniprot_id] = dict(zip(keys, hit))
        return dict_hits


def convert_fas_to_a3m(fas_path:str, a3m_path:str, reformat_config:msa_config.ReformatConfig) -> None:
    try:
        # Ensure the input file exists
        if not os.path.exists(fas_path):
            raise FileNotFoundError(f"Input file '{fas_path}' does not exist.")
        binary_reformat_path = reformat_config.binary_path
        reformat_target = reformat_config.reformat_target

        # Call the reformat.pl script
        cmd = ['perl',binary_reformat_path,reformat_target, fas_path, a3m_path]
        subprocess_utils.run(
                      cmd=cmd,
          cmd_name=f'reformat ({os.path.basename(fas_path)})',
          log_stdout=False,
          log_stderr=True,
          log_on_process_error=True,
        )

    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")
        print("stderr:", e.stderr)
    except Exception as e:
        print(f"An error occurred: {e}")




def get_msa_a3mpath(sequence:str,
                    dict_hits:Mapping[str,Mapping[str,str]],
                    reformat_config: msa_config.ReformatConfig,
                    id_cutoff:float|None=90.0):

    PATH_FAS_MSA = Path(reformat_config.PATH_FAS_MSA)
    path_a3m_dir = Path(reformat_config.a3m_dir)

    # Extract Uniprot IDs from the hits.
    uniprot_id = [k for k in dict_hits][0]
    
    print("Found referenced Uniprot hit(s):")
    seq_identity = float(dict_hits[uniprot_id]['pident'])
    seq_query = sequence
    seq_cov = float(dict_hits[uniprot_id]['qcovs'])
    slen = int(dict_hits[uniprot_id]['slen'])
    qlen = int(dict_hits[uniprot_id]['qlen'])

    fas_filename = f'{uniprot_id}.fas'
    path_msafas = PATH_FAS_MSA/fas_filename

    path_a3m = ''

    print('slen', slen, 'qlen', qlen)
    print('path_msafas',path_msafas)
    if path_msafas.exists():
        seq_ref = get_refseq(str(path_msafas))

       # if seq_identity == 100.0 and seq_cov == 100.0 and slen == qlen:   
           # logging.info("Found 100  %%\\ seq identity pre-computed")
           # print("Found 100  %%\\ seq identity pre-computed")
           # a3m_name = f'{uniprot_id}.a3m'
           # path_a3m = path_a3m_dir/a3m_name
           # try:
           #     convert_fas_to_a3m(fas_path = str(path_msafas), 
           #                     a3m_path = str(path_a3m),
           #                     reformat_config= reformat_config)
           # except Exception as e:
           #     print(e)
           #     return str(path_a3m)
           # else:
           #     return str(path_a3m)

        if seq_identity >= id_cutoff:
            logging.info(f'Found higher than {id_cutoff}  %%\\ seq identity pre-computed')
            logging.info(f'sequence identity {seq_identity}, coverage is {seq_cov}')
            print(f'Found higher than {id_cutoff}  %%\\ seq identity pre-computed')

            realign_filename = f'{uniprot_id}_realign.fas'
            realign_a3mfilename = f'{uniprot_id}_realign.a3m'
            
            path_realign_a3m = path_a3m_dir/realign_a3mfilename
            with tempfile.TemporaryDirectory() as query_tmp_dir:
                temp_realign_path = os.path.join(query_tmp_dir, realign_filename)
                try:
                    write_realign(str(path_msafas), 
                                    seq_query, 
                                    temp_realign_path )

                    convert_fas_to_a3m(fas_path = str(temp_realign_path), 
                                    a3m_path = str(path_realign_a3m),
                                    reformat_config= reformat_config)
                    
                except Exception as e:
                    print('error is ',e)
                    return str(path_a3m)
                else:
                    os.remove(temp_realign_path)
                    return str(path_realign_a3m)
        else:
            logging.info(f"Not find sequence with higher than {id_cutoff}  %%\\ identity MSA hits found.")
            return str(path_a3m)
    else:
        logging.info("No matched pre-computed MSA hits found.")
        return str(path_a3m)   
