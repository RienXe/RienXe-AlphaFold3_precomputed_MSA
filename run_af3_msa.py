
"""
Loads Input Data: Reads one or more JSON files containing folding inputs.
Processes Data: Optionally runs a data pipeline to preprocess the input.
Writes Intermediate Files: Creates JSON files with filled or reformatted data.
Submits Jobs: Sets up directories and submits MSA (multiple sequence alignment) jobs via SLURM.
Parallel Execution: Uses a ProcessPoolExecutor to process multiple fold inputs concurrently.
"""
import os
import pathlib
from collections.abc import Collection, Iterator, Mapping, Sequence
from data import folding_input, pipeline
from utils import slurm_handle
import datetime
import logging
import time
import argparse
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Tuple
import subprocess
from utils.slurm_manager import create_slurm_manager


DEFAULT_OMTFMT = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen qcovs qseq sseq"
DEFUALT_REFORMAT_TARGET = 'a3m'


def get_lens(fold_input:folding_input.Input) -> int:
    """
    Calculates the total length of all sequences across protein, DNA, and RNA chains
    from the given folding input.

    Args:
        fold_input (folding_input.Input): An object that contains multiple attributes, only three of them will be calculated:
            - protein_chains: A list of protein chains, where each chain contains a sequence.
            - dna_chains: A list of DNA chains, where each chain contains a sequence.
            - rna_chains: A list of RNA chains, where each chain contains a sequence.

    Returns:
        int: The total length of all sequences across the protein, DNA, and RNA chains.
             This is computed by summing up the lengths of the sequences in the chains.
    """
    total_length = 0
    for type_chain in [fold_input.protein_chains, fold_input.dna_chains, fold_input.rna_chains]:
        for chain in type_chain:
            total_length += len(chain.sequence)
    return total_length


def write_fold_input_json(
    fold_input: folding_input.Input,
    output_dir: os.PathLike[str] | str,
) -> str:
    """
    Purpose:
        Serializes a fold input object into JSON format and writes it to a specified output directory.
        If a file with the same name already exists, the function appends a timestamp to create a unique filename.

    Parameters:
        fold_input: An instance of folding_input.Input containing the model input data. It must support the methods sanitised_name() and to_json().
        output_dir: The directory path where the JSON file will be saved. Can be provided as either a string or an os.PathLike object.

    Returns:
        string: A string representing the full path to the written JSON file.
"""

    """Writes the input JSON to the output directory."""
    path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_primary.json')
    if os.path.exists(path):
        path = os.path.join(output_dir, f'{fold_input.sanitised_name()}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}_primary.json')
    
    print(f'Writing model input JSON to {path}')
    with open(path, 'wt') as f:
        f.write(fold_input.to_json())
    return path


# Define your function to process each fold input
def process_fold_input_task(args:Tuple[str, pipeline.DataPipelineConfig, str]) -> None:
    """
    Purpose:
    A wrapper function designed to be used as a worker task for parallel execution. It unpacks a tuple of arguments and calls the main processing function.

    Parameters:
        args: A tuple containing three elements:
        fold_input: An instance of folding_input.Input.
        data_pipeline_config: An instance of pipeline.DataPipelineConfig or None if no data pipeline should be applied.
        path_outdir: The output directory path where final results will be saved.

    """

    fold_input, data_pipeline_config, path_outdir = args

    process_fold_input(fold_input=fold_input, 
                       data_pipeline_config=data_pipeline_config, 
                       output_dir=path_outdir)
    

def get_json_names(fold_inputs:folding_input.Input):
    """
    Purpose:
        Intended to iterate over a collection of fold inputs and extract their sanitized names.

    Parameters:
        fold_inputs: An iterable (e.g., list) of folding_input.Input objects.
    Returns:
        A list of names obtained by calling sanitised_name() on each fold input.
    """
    list_names = []
    for fold_input in fold_inputs:
        list_names.append(fold_input.sanitised_name())
    return list_names


def process_fold_input(
    fold_input: folding_input.Input,
    data_pipeline_config: pipeline.DataPipelineConfig | None,
    output_dir: os.PathLike[str] | str,
) -> None:
    """
    Purpose:
        Processes a single fold input by:
            Running the optional data pipeline.
            Filling in any missing fields in the input.
            Writing an intermediate JSON file.
            Creating appropriate output directories for the complete JSON and SLURM logs.
            Submitting an MSA job via SLURM.

    Parameters:
        fold_input: An instance of folding_input.Input representing the input data.
        data_pipeline_config: An optional DataPipelineConfig instance for preprocessing the input. If None, the data pipeline is skipped.
        output_dir: The base output directory for results.
    """

    print(f'\nRunning fold job {fold_input.name}...')

    if not fold_input.chains:
        raise ValueError('Fold input has no chains.')

    print(f'Output will be written in {output_dir}')

    if data_pipeline_config is None:
        print('Skipping data pipeline...')
    else:
        print('Running data pipeline...')
        fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)

    job_name = fold_input.sanitised_name()
    path_output = os.path.join(output_dir, job_name)
    # Check if the output directory already exists, if so skip the fold input
    if os.path.exists(path_output):
        return print(f'Output directory {path_output} already exists. Skipping fold input {fold_input.name}.')
        
    
    #intermdiate path
    path_rewrotejson_dir = data_pipeline_config.path_rewrotejson_dir

    path_completejson_dir = data_pipeline_config.path_completejson_dir
    path_slurm_logsab5000 = data_pipeline_config.path_slurm_logsab5000
    path_slurm_bash = data_pipeline_config.path_slurm_bash
    path_slurm_bash_cpu = data_pipeline_config.path_slurm_bash_cpu

    path_slurm_logs = data_pipeline_config.path_slurm_logs

    #fill the missing field of json
    fold_input = fold_input.fill_missing_fields()

    #calculate total tokens of proteins
    total_lens_noligand = get_lens(fold_input)
    print('total_lens_noligand',total_lens_noligand)

    #write the intermediate json file into the intermediate path
    path_rewrotejson = write_fold_input_json(fold_input, path_rewrotejson_dir)
    job_name = path_rewrotejson.split(path_rewrotejson_dir)[1].strip('/').split('.')[0]


    #create folder for MSA output of json file, if folder is already existed    
    dir_completejson = os.path.join(path_completejson_dir, job_name)

    if os.path.exists(dir_completejson) and os.listdir(dir_completejson):
        new_output_dir = (
            f'{dir_completejson}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        print(
            f'Output will be written in {new_output_dir} since {dir_completejson} is'
            ' non-empty.'
        )
        dir_completejson = new_output_dir

    else:
        print(f'Output will be written in {dir_completejson}')
    
    os.makedirs(dir_completejson, exist_ok=True)
    
    dict_paths = {'path_rewrotejson':path_rewrotejson, 
                  'dir_completejson': dir_completejson,
                  'path_slurm_logsab5000': path_slurm_logsab5000,
                  'path_slurm_bash': path_slurm_bash,
                  'output_dir': data_pipeline_config.path_output_dir,
                  'job_name_original':fold_input.sanitised_name(),
                  'path_slurm_bash_cpu': path_slurm_bash_cpu}
    
    result_msa = slurm_handle.submit_msajob_toslurm(dict_paths, path_slurm_logs ,job_name,total_lens_noligand)
    logging.info(result_msa)



def main():
    """
    Purpose:
        Serves as the entry point of the script. It parses command-line arguments, loads the fold inputs from the specified path, sets up the data pipeline configuration, and dispatches the processing tasks.

    Behavior:
        Argument Parsing: Uses argparse to handle parameters such as input path, output directory, database paths, and paths to required binary scripts.
    
    Input Loading:
        If the input path is a directory, it loads multiple fold inputs using folding_input.load_fold_inputs_from_dir.
        If the input is a single JSON file, it loads it using folding_input.load_fold_inputs_from_path.
        Validates that the input file has a .json suffix.
    
    Directory Setup:
        Creates an intermediate result directory with a timestamp.
        Sets up subdirectories for rewritten JSON, complete JSON, and SLURM logs.
    
    Data Pipeline Configuration:
        Constructs a DataPipelineConfig object with paths and parameters from the command-line arguments.
    
    Parallel Processing:
        Creates a list of tasks (each a tuple of (fold_input, data_pipeline_config, path_outdir)) and uses a ProcessPoolExecutor to process each task concurrently.

    Usage:
        python run_af3_msa.py --path_input path/to/input --path_outdir path/to/output --path_intermediate_result path/to/intermediate_result --path_db path/to/uniprot_db --path_msa_db path/to/msa_db --binary_blastp_script path/to/blastp --binary_reformat_script path/to/reformat.pl
    shortest usage:
        python run_af3_msa.py --path_input path/to/input --path_outdir path/to/output --path_intermediate_result path/to/intermediate result
    """

    parser = argparse.ArgumentParser(description="Process json input data")
    parser.add_argument('--path_input', type=str, required=True, help='Path to the input json file or directory of multiple json')
    parser.add_argument('--path_outdir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--path_intermediate_result', type=str, required=True, default='/projects/perrakis_alphafold/alphafold3/process/', 
                            help='Path to the intermediate result. Default path is /projects/perrakis_alphafold/alphafold3/process/')
    parser.add_argument('--path_db', type=str, required=False, help = 'Path to the UniProt fasta database to find match uniprot entry. Default path is ./review_database/uniprotkb_taxonomy_id_9606_AND_reviewed_2025_02_06.fasta', 
                        default= './review_database/uniprotkb_taxonomy_id_9606_AND_reviewed_2025_02_06.fasta')
    parser.add_argument('--path_msa_db', type=str, required=False, default= '/data/groups/public/archive/humanPPI_MSA/humanPPI_MSA/MSA/',
                         help='Path to the pre-computed MSA database of msa with name as uniprot id. Default path is /data/groups/public/archive/humanPPI_MSA/humanPPI_MSA/MSA/')
    parser.add_argument('--binary_blastp_script', type=str, required=False, default='./blast/bin/blastp',
                        help='Path to the BLASTP binary script. Default path is ./blast/bin/blastp')
    parser.add_argument('--binary_reformat_script', type=str, required=False, default='./reformat.pl',
                         help='Path to the reformat.pl binary script. Default path is ./reformat.pl')

    parser.add_argument('--gpu_restriction',type = int, required=False, default=4,
                         help='gpu restriction for job more than 5000 tokens. Default is 4, which means the job will be submitted to slurm with 4 gpus.')

    args = parser.parse_args()

    path_input = pathlib.Path(args.path_input).resolve()
    path_outdir = pathlib.Path(args.path_outdir).resolve()
    path_intermediate_result = os.path.abspath(args.path_intermediate_result)
    

    if path_input.is_dir():
        fold_inputs = folding_input.load_fold_inputs_from_dir(path_input, path_outdir)

        
    elif path_input.is_file():
        if path_input.suffix == '.json':
            fold_inputs = folding_input.load_fold_inputs_from_path(path_input,path_outdir)

        else:
            raise AssertionError(
            f'{str(path_input)} file must be .json'
            )

    else:
            raise AssertionError(
            f'{str(path_input)} must be either .json file or directory contains multiple .json file'
            )


    # Create a list of tuples containing arguments for each task
    path_intermediate_result = f'{path_intermediate_result}/{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    print(f'Output will be written in {path_intermediate_result}')

    path_intermediate_result = pathlib.Path(path_intermediate_result).resolve()
    path_rewrotejson_dir = path_intermediate_result/'rewrote_json'
    path_completejson_dir = path_intermediate_result/'complete_json'
    path_a3m_dir = path_intermediate_result/'a3m'
    path_slurm_logs = path_intermediate_result/'slurm'
    path_slurm_logsab5000 = path_slurm_logs/'above_5000'
    path_slurm_bash_above5000 = path_intermediate_result/'above_5000_bash'
    path_slurm_logscpu = path_slurm_logs/'below_5000'
    path_slurm_bash_cpu = path_intermediate_result/'cpu'


    path_slurm_logab5000_executed = path_slurm_logsab5000/'executed_jobs.log'
    gpu_restriction = args.gpu_restriction


    os.makedirs(path_rewrotejson_dir, exist_ok=True)
    os.makedirs(path_completejson_dir, exist_ok=True)
    os.makedirs(path_slurm_logs, exist_ok=True)
    os.makedirs(path_a3m_dir, exist_ok=True)
    os.makedirs(path_slurm_logsab5000, exist_ok=True)
    os.makedirs(path_slurm_bash_above5000, exist_ok=True)
    os.makedirs(path_slurm_logscpu, exist_ok=True)
    os.makedirs(path_slurm_bash_cpu, exist_ok=True)

    data_pipeline_config = pipeline.DataPipelineConfig(
        PATH_FAS_MSA= os.path.abspath(args.path_msa_db),
        binary_blastp_script= os.path.abspath(args.binary_blastp_script),
        blastp_query= '',
        blastp_db= os.path.abspath(args.path_db),
        blastp_outfmt = DEFAULT_OMTFMT,

        binary_reformat_script= os.path.abspath(args.binary_reformat_script),
        reformat_target= DEFUALT_REFORMAT_TARGET,
        reformat_fas_path= '',
        reformat_a3m_dir = path_a3m_dir,

        path_output_dir = str(path_outdir),
        path_rewrotejson_dir = str(path_rewrotejson_dir),
        path_completejson_dir = str(path_completejson_dir),
        path_slurm_logsab5000 = str(path_slurm_logsab5000),
        path_slurm_logs = str(path_slurm_logs),
        path_slurm_bash = str(path_slurm_bash_above5000),
        path_slurm_bash_cpu = str(path_slurm_bash_cpu),
    )

    path_array_slurm_dir= os.path.join(path_slurm_logsab5000, 'slurm_array')
    # Create the slurm manager for jobs above 5000 tokens and below 5000 tokens
    configure_slurm_above5000 = pipeline.ConfigureForSlurm(
        path_array_slurm_dir= path_array_slurm_dir,
        path_slurm_array= os.path.join(path_array_slurm_dir, 'array_test.sh'),
        path_exectued_jobs= os.path.join(path_array_slurm_dir, 'executed_jobs.txt'),
        path_output_log= os.path.join(path_array_slurm_dir, 'slurm_output_array.out'),
        path_error_log= os.path.join(path_array_slurm_dir, 'slurm_output_array.err'),
        interval=240  # Interval in seconds for checking the status of the jobs
    )

    path_array_slurm_dir= os.path.join(path_slurm_logscpu, 'slurm_array')
    configure_slurm_below5000_cpu = pipeline.ConfigureForSlurm(
        path_array_slurm_dir= path_array_slurm_dir,
        path_slurm_array= os.path.join(path_array_slurm_dir, 'array_test.sh'),
        path_exectued_jobs= os.path.join(path_array_slurm_dir, 'executed_jobs.txt'),
        path_output_log= os.path.join(path_array_slurm_dir, 'slurm_output_array.out'),
        path_error_log= os.path.join(path_array_slurm_dir, 'slurm_output_array.err'),
        interval=120  # Interval in seconds for checking the status of the jobs
    )


#    create_slurm_manager(configure_slurm_below5000_cpu, path_slurm_bash_cpu, gpu_restriction,slepp_time= '1m')

    for fold_input in fold_inputs:
        process_fold_input(fold_input=fold_input, 
                data_pipeline_config=data_pipeline_config, 
                output_dir=path_outdir)

#    if len(os.listdir(path_slurm_bash_above5000)) > 0:
#        create_slurm_manager(configure_slurm_above5000, path_slurm_bash_above5000, gpu_restriction)

    # result = subprocess.run(["bash", './utils/slurm_manager.sh', path_slurm_bash, path_slurm_logab5000_executed], capture_output=True, text=True, check=True)
    
    # tasks = [(fold_input, data_pipeline_config, path_outdir) for fold_input in fold_inputs]

    # # Use ProcessPoolExecutor to parallelize the task
    # with ProcessPoolExecutor(max_workers=1) as executor:
    #     executor.map(process_fold_input_task, tasks)

if __name__ == "__main__":
    main()
