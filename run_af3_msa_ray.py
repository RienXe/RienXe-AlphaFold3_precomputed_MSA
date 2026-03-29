import ray
import subprocess
import os
import argparse
import pathlib
import datetime
from typing import Mapping


from collections.abc import Collection, Iterator, Mapping, Sequence
from data import folding_input, pipeline
from utils import slurm_handle
import datetime
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Tuple
from utils.slurm_manager import create_slurm_manager


DEFAULT_OMTFMT = "6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen qcovs qseq sseq"
DEFAULT_REFORMAT_TARGET = 'a3m'

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

    job_name = fold_input.sanitised_name()
    path_output = os.path.join(output_dir, job_name)
    # Check if the output directory already exists, if so skip the fold input
    if os.path.exists(path_output):
        return print(f'Output directory {path_output} already exists. Skipping fold input {fold_input.name}.')

    if data_pipeline_config.MSA_mode == 'default':
        print('Use default MSA skipping data pipeline...')
    else:
        print('Running data pipeline...')
        fold_input = pipeline.DataPipeline(data_pipeline_config).process(fold_input)    
        # fold_input = fold_input.fill_missing_fields()
    
    #intermdiate path
    path_rewrotejson_dir = data_pipeline_config.path_rewrotejson_dir

    path_completejson_dir = data_pipeline_config.path_completejson_dir
    path_slurm_logsab5000 = data_pipeline_config.path_slurm_logsab5000
    path_slurm_bash = data_pipeline_config.path_slurm_bash
    path_slurm_bash_cpu = data_pipeline_config.path_slurm_bash_cpu

    path_slurm_logs = data_pipeline_config.path_slurm_logs

    #fill the missing field of json

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
    
    # result_msa = slurm_handle.submit_msajob_toslurm(dict_paths, path_slurm_logs ,job_name,total_lens_noligand)
    return dict_paths, job_name, total_lens_noligand



# ---------- MSA Task ----------
@ray.remote(num_cpus=3, 
            memory=7*1024*1024*1024,
            resources={"CPU_NODE": 0.1}  # Ensure this task runs on the CPU node
            )  # Adjust CPU/memory like in your sbatch
def msa_task(
    dict_paths: Mapping[str, str], 
    job_name: str, 
    total_tokens: int
    ):
    """
    Ray version of the MSA job.
    Runs AlphaFold3 MSA preprocessing inside Apptainer.
    Returns dict with paths for CPU/GPU tasks.
    """

    print(f"running for {job_name}")
    path_rewrotejson = dict_paths["path_rewrotejson"]
    dir_completejson = dict_paths["dir_completejson"]
    output_dir = dict_paths["output_dir"]
    job_name_original = dict_paths["job_name_original"]
    print(f"rewrote json path: {os.path.basename(path_rewrotejson)}")


    os.makedirs(dir_completejson, exist_ok=True)
    cmd = [
        "apptainer", "exec", "--nv", "--no-home",
        f"--bind={os.path.dirname(path_rewrotejson)}:$HOME/af_input",
        f"--bind={dir_completejson}:$HOME/af_output",
        "--bind=/processing/alphafold3:$HOME/models",
        "--bind=/processing/alphafold3:$HOME/public_databases",
        "/processing/alphafold3/alphafold3.sif",
        "python", "/app/alphafold/run_alphafold.py",
        f"--json_path=$HOME/af_input/{os.path.basename(path_rewrotejson)}",
        "--model_dir=$HOME/models",
        "--output_dir=$HOME/af_output",
        "--db_dir=$HOME/public_databases",
        "--run_inference=False",
    ]

    print("Running MSA task:", " ".join(cmd))
    subprocess.run(" ".join(cmd), shell=True, check=True)

    return {
        "job_name": job_name,
        "job_name_original": job_name_original,
        "rewrote_json": path_rewrotejson,
        "dir_completejson": dir_completejson,
        "output_dir": output_dir,
        "total_tokens": total_tokens,
    }



# ---------- CPU Task ----------
@ray.remote(num_cpus=2, 
            # num_gpus=0, 
            memory=7 * 1024 * 1024 * 1024,
            resources={"CPU_NODE": 0.1} 
            )  
def af3_cpu_task(job_name, json_path, complete_json_dir, output_dir):
    """
    Runs AlphaFold3 inference in CPU-only mode inside Apptainer.
    Equivalent to the SLURM CPU sbatch script.
    """
    print("running cpu for", job_name, 'json_path', os.path.basename(json_path) )
    cmd = [
        "apptainer", "exec", "--nv", "--no-home",
        f"--bind={os.path.dirname(json_path)}:$HOME/af_input",
        f"--bind={complete_json_dir}:$HOME/af_output",
        "--bind=/processing/alphafold3:$HOME/models",
        "--bind=/processing/alphafold3:$HOME/public_databases",
        "/processing/alphafold3/alphafold3.sif",
        "python", "/app/alphafold/run_alphafold.py",
        f"--json_path=$HOME/af_input/{os.path.basename(json_path)}",
        "--model_dir=$HOME/models",
        "--output_dir=$HOME/af_output",
        "--db_dir=$HOME/public_databases",
        "--run_inference=False",
    ]
    print("Running CPU task:", " ".join(cmd))
    subprocess.run(" ".join(cmd), shell=True, check=True)
    return {"job_name": job_name, 
            "complete_json_dir": complete_json_dir, 
            "output_dir": output_dir}


# ---------- GPU Task ----------
@ray.remote(num_cpus=15, 
            num_gpus=1, 
            memory=65 * 1024 * 1024 * 1024,
            resources={"GPU_NODE": 0.1}  # Ensure this task runs on the GPU node
            )  
def af3_gpu_task(complete_json_dir, output_dir, job_name):
    """
    Runs AlphaFold3 inference on GPU inside Apptainer.
    Equivalent to the SLURM GPU sbatch script.
    """
    print("running gpu for", job_name)
    print('complete_json_dir', complete_json_dir)
    print('output_dir', output_dir)
    cmd = [
        "apptainer", "exec", "--nv", "--no-home",
        f"--bind={complete_json_dir}:$HOME/af_input",
        f"--bind={output_dir}:$HOME/af_output",
        "--bind=/processing/alphafold3:$HOME/models",
        "--bind=/processing/alphafold3:$HOME/public_databases",
        "/processing/alphafold3/alphafold3.sif",
        "python", "/app/alphafold/run_alphafold.py",
        f"--input_dir=$HOME/af_input/{job_name}",
        "--model_dir=$HOME/models",
        "--output_dir=$HOME/af_output",
        "--run_data_pipeline=False",
    ]
    print("Running GPU task:", " ".join(cmd))
    subprocess.run(" ".join(cmd), shell=True, check=True)
    return {"job_name": job_name, "output_dir": output_dir}


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
    # non-default args
    parser.add_argument('--path_input', type=str, required=True, help='Path to the input json file or directory of multiple json')
    parser.add_argument('--path_outdir', type=str, required=True, help='Path to the output directory')
    parser.add_argument("--ray-address", default="auto")
    # default args
    parser.add_argument('--path_intermediate_result', type=str, required=True, default='/projects/alphafold3_tcr_pmhc/process/', 
                            help='Path to the intermediate result. Default path is /projects/alphafold3_tcr_pmhc/process/')
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

    parser.add_argument('--mode', type=str, required=False, default='pre_msa', choices=['default', 'pre_msa'], 
                        help='Which means it will use pre-computed MSA and skip the MSA step in data pipeline, ' \
                        'which is useful when you want to accelerate the process and your input is human or mammalian, ' \
                        'default means use AF3 default MSA, which means slow but more accurate. Default is pre_msa.')
    
    args = parser.parse_args()
    path_input = pathlib.Path(args.path_input).resolve()
    path_outdir = pathlib.Path(args.path_outdir).resolve()
    path_intermediate_result = os.path.abspath(args.path_intermediate_result)
    MSA_mode = args.mode
    
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
    file_folder_name = path_input.stem if path_input.is_file() else path_input.name
    completed_jobs = os.listdir(path_outdir) if os.path.exists(path_outdir) else []
    print(f'completed jobs found in output dir {len(completed_jobs)}')
    completed_jobs = [job.lower() for job in completed_jobs]
    fold_inputs = list(fold_inputs)  # Ensure fold_inputs is a list for multiple iterations
    print(f'jobs found in input dir/file {len(fold_inputs)}')
    fold_inputs = [fi for fi in fold_inputs if fi.sanitised_name().lower() not in completed_jobs]
    print(f'jobs to be processed after removing completed jobs {len(fold_inputs)}')

    if len(fold_inputs) == 0:
        print('No jobs to be processed. Exiting.')
        return

    # Create a list of tuples containing arguments for each task
    path_intermediate_result = f'{path_intermediate_result}/{file_folder_name}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
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
    print('created intermediate result path and subdirs:', path_intermediate_result)

    data_pipeline_config = pipeline.DataPipelineConfig(
        PATH_FAS_MSA= os.path.abspath(args.path_msa_db),
        MSA_mode= MSA_mode,
        binary_blastp_script= os.path.abspath(args.binary_blastp_script),
        blastp_query= '',
        blastp_db= os.path.abspath(args.path_db),
        blastp_outfmt = DEFAULT_OMTFMT,

        binary_reformat_script= os.path.abspath(args.binary_reformat_script),
        reformat_target= DEFAULT_REFORMAT_TARGET,
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

    print('Data pipeline config:', data_pipeline_config)
    ### new code: ray version
    ray.init(address=args.ray_address)
    print('ray initialized')
    print("Available resources (before msa):", ray.available_resources())

    cpu_futures = []
    gpu_futures = []
    msa_futures = []
    fold_iter = iter(fold_inputs)
    fold_input_remaining = True

    while True:
        # Submit MSA tasks as long as there are fold_inputs left
        try:
            while len(msa_futures) < 10:  # optional: limit concurrent MSA tasks
                fold_input = next(fold_iter)
                dict_paths, job_name, total_lens_noligand = process_fold_input(
                    fold_input=fold_input, 
                    data_pipeline_config=data_pipeline_config, 
                    output_dir=path_outdir
                )
                msa_futures.append(msa_task.remote(dict_paths, job_name, total_lens_noligand))
        except StopIteration:
            fold_input_remaining = False

        # Wait for any MSA or CPU task to finish
        if not msa_futures and not cpu_futures and not fold_input_remaining:
            break  # finished everything

        done, pending = ray.wait(msa_futures + cpu_futures, num_returns=1, timeout=0.1)

        for future in done:
            result = ray.get(future)

            if future in msa_futures:
                msa_futures.remove(future)
                cpu_future = af3_cpu_task.remote(
                    result["job_name"],
                    result["rewrote_json"],
                    result["dir_completejson"],
                    result["output_dir"],
                )
                cpu_futures.append(cpu_future)

            elif future in cpu_futures:
                cpu_futures.remove(future)
                gpu_future = af3_gpu_task.remote(
                    result["complete_json_dir"],
                    result["output_dir"],
                    result["job_name"].replace("_primary", ""),
                )
                gpu_futures.append(gpu_future)
            del result  

        time.sleep(0.1)

    print("All MSA and CPU tasks completed, waiting for GPU tasks to finish...")
    ray.get(gpu_futures)
    print("All GPU tasks completed.")

if __name__ == "__main__":
    main()
