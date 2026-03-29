import subprocess
from collections.abc import Mapping
import pathlib 
import os
import math

def write_script_to_file(script: str, file_path: str) -> None:
    """
    Purpose:
        Writes a given script to a specified file. Each line of the script is written to the file, followed by a newline character.
    Parameters:
        script: The script to be written to the file.
        file_path: The path where the script will be saved.
    """
    # Ensure the directory exists
    with open(file_path, 'w') as file:
        for line in script:
            file.write(line)
            file.write('\n')

def read_txt(file_path):
    """
    Reads a text file and returns its content as a string.
    
    :param file_path: Path to the text file.
    :return: Content of the file as a string.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    content = []
    with open(file_path, 'r') as r_manager:
        lines = r_manager.readlines()
        for line in lines:
            content.append(line.strip())
    
    return content


# Function to create a Slurm script

def run_msa_byaf3(job_name:str,path_rewrotejson:str, dir_completejson:str, path_slurm_batch_gpu:str, total_tokens = 5000) -> str:
    """
    Purpose:
        Generates a command string to perform the MSA step using Apptainer. The command binds directories for input (rewritten JSON), output, models, and public databases, and it calls the run_alphafold.py script with MSA options.
    Parameters:
        job_name: The name of the job, used in the script header.
        path_rewrotejson: Path to the rewritten JSON file.
        dir_completejson: Directory for complete JSON/MSA
        path_slurm_batch_gpu: Path to the Slurm batch script for GPU processing.
        path_slurm_bash: Path to the Slurm bash script.
        total_tokens: The total number of tokens in the input sequence. This value is used to determine the number of GPUs required for the job.
    Returns:
        A command string that includes all required bindings and parameters for the MSA process.
    """

    path_rewrotejson = pathlib.Path(path_rewrotejson)
    path_slurm_log = pathlib.Path(path_slurm_batch_gpu).parent
    dir_rewrotejson = path_rewrotejson.parent
    json_name = path_rewrotejson.name
    # path_slurm_bash_script_ori = os.path.join(path_slurm_log, f'{job_name}.sh')
    # path_slurm_bash_script_des = os.path.join(path_slurm_bash, f'{job_name}.sh')
    cmd = ['time ', 'apptainer ', 'exec ', '--nv ', '--no-home ', '\\\n' ,
            '\t--bind ', f'{dir_rewrotejson}:$HOME/af_input ' , '\\\n', 
            '\t--bind ', f'{dir_completejson}:$HOME/af_output ', '\\\n',
            '\t--bind ', ' /processing/alphafold3:$HOME/models ', '\\\n',
            '\t--bind ', '/processing/alphafold3:$HOME/public_databases ', '\\\n',
            '\t/processing/alphafold3/alphafold3.sif ', '\\\n',
            '\tpython ', '/app/alphafold/run_alphafold.py ', '\\\n',
            f'\t--json_path=$HOME/af_input/{json_name} ','\\\n',
            '\t--model_dir=$HOME/models ','\\\n',
            f'\t--output_dir=$HOME/af_output ', '\\\n',
            f'\t--db_dir=$HOME/public_databases ', '\\\n',
            '\t--run_inference=False ','\\\n',
            '\n']


    if total_tokens <= 5000:
        cmd.append(f'sbatch {path_slurm_batch_gpu}')
        cmd = ''.join(cmd)

        return cmd
    else:
        # write_script_to_file(cmd, path_slurm_bash_script_ori)
        # cmd.append(f'mv {path_slurm_bash_script_ori} {path_slurm_bash_script_des}')
        return cmd


def run_af3_inference(job_name_ori:str,dir_complete_msa:str, output_dir:str,  path_slurm_logs:str, total_tokens = 5000) -> str:

    """
    Purpose:
        Generates a command string to perform the inference step using Apptainer. The command binds directories for input (complete MSA results), output, models, and public databases, and it calls the run_alphafold.py script with inference options.

    Parameters:

        dir_complete_msa: Directory containing the complete MSA JSON results.
        output_dir:Directory where the final inference output will be written.
        job_name_ori:Original job name (sanitized) used to identify the input directory for inference.
        total_tokens: The total number of tokens in the input sequence. This value is used to determine the number of GPUs required for the job.

    Returns:
    A command string that includes all required bindings and parameters for the inference process.
    
    """
    if total_tokens > 5000:
        cmd = ['time ', 'apptainer ', 'exec ', '--nv ', '--no-home ', '\\\n' ,
            '\t--bind ', f'{dir_complete_msa}:$HOME/af_input ' , '\\\n', 
            '\t--bind ', f'{output_dir}:$HOME/af_output ', '\\\n',
            '\t--bind ', ' /processing/alphafold3:$HOME/models ', '\\\n',
            '\t--bind ', '/processing/alphafold3:$HOME/public_databases ', '\\\n',
            '\t--env ', 'XLA_PYTHON_CLIENT_PREALLOCATE=false ', '\\\n',
            '\t--env ', 'TF_FORCE_UNIFIED_MEMORY=true ', '\\\n',
            '\t--env ', 'XLA_CLIENT_MEM_FRACTION=3.2 ', '\\\n',
            '\t/processing/alphafold3/alphafold3.sif ', '\\\n',
            '\tpython ', '/app/alphafold/run_alphafold.py ', '\\\n',
            f'\t--input_dir=$HOME/af_input/{job_name_ori} ','\\\n',
            '\t--model_dir=$HOME/models ','\\\n',
            f'\t--output_dir=$HOME/af_output ', '\\\n',
            f'\t--run_data_pipeline=False ', '\\\n']
        return cmd
    else:
        cmd = ['time ', 'apptainer ', 'exec ', '--nv ', '--no-home ', '\\\n' ,
            '\t--bind ', f'{dir_complete_msa}:$HOME/af_input ' , '\\\n', 
            '\t--bind ', f'{output_dir}:$HOME/af_output ', '\\\n',
            '\t--bind ', ' /processing/alphafold3:$HOME/models ', '\\\n',
            '\t--bind ', '/processing/alphafold3:$HOME/public_databases ', '\\\n',
            '\t/processing/alphafold3/alphafold3.sif ', '\\\n',
            '\tpython ', '/app/alphafold/run_alphafold.py ', '\\\n',
            f'\t--input_dir=$HOME/af_input/{job_name_ori} ','\\\n',
            '\t--model_dir=$HOME/models ','\\\n',
            f'\t--output_dir=$HOME/af_output ', '\\\n',
            f'\t--run_data_pipeline=False ', '\\\n']

    
        cmd = ''.join(cmd)
        return cmd


def create_slurm_script_cpu(job_name:str, 
                            path_output_log:str, 
                            path_error_log:str, 
                            path_slurm_batch:str, 
                            path_slurm_bash_cpu:str,
                            command:Mapping[str]) -> None:
    
    """
    Purpose:
    Creates a Slurm batch script for CPU-based processing. This script sets up job parameters such as job name, CPU allocation, time limit, memory, and log file paths, and embeds the provided command to be executed.

    Parameters:
        job_name:The name of the job; used in both the script header and log file naming.
        path_output_log: Path to the file where standard output logs will be written.
        path_error_log: Path to the file where error logs will be captured.
        path_slurm_batch: The file path where the Slurm batch script will be created.
        command:The command string (or mapping of command details) that will be executed as part of the Slurm job.

    """

    slurm_script = read_txt('./utils/example_slurmtitle_text/slurm_cpu.txt')
    slurm_script.append(f'#SBATCH --job-name={job_name}')
    slurm_script.append(f'#SBATCH --output={path_output_log}')
    slurm_script.append(f'#SBATCH --error={path_error_log}')
    slurm_script.append('#SBATCH --mem=30GB')
    slurm_script.append('\n')
    slurm_script.append(command)

    # slurm_script = ['#!/bin/bash', 
    #                 f'#SBATCH --job-name={job_name}',
    #                 '#SBATCH -p cpu',
    #                 '#SBATCH --ntasks=1',
    #                 '#SBATCH --cpus-per-task=10',
    #                 '#SBATCH --qos=cpu_qos',
    #                 f'#SBATCH --output={path_output_log}',
    #                 f'#SBATCH --error={path_error_log}',
    #                 '#SBATCH --time=10:0:0',
    #                 '#SBATCH --mem=30GB',
    #                 '\n',
    #                 command]

    # slurm_script = ['#!/bin/bash', 
    #                 f'#SBATCH --job-name={job_name}',
    #                 '#SBATCH -p a100',
    #                 '#SBATCH --gres=gpu:1',
    #                 '#SBATCH --qos=a100_qos',
    #                 '#SBATCH --nodelist euctemon',
    #                 f'#SBATCH --output={path_output_log}',
    #                 f'#SBATCH --error={path_error_log}',
    #                 '#SBATCH --time=10:0:0',
    #                 '#SBATCH --mem=100GB',
    #                 '\n',
    #                 command]
    
    write_script_to_file(slurm_script, path_slurm_batch)
    write_script_to_file(slurm_script, path_slurm_bash_cpu)


def create_slurm_script_gpu(job_name: str, 
                            path_output_log:str, 
                            path_error_log:str, 
                            path_slurm_batch: str, 
                            command:Mapping[str],
                            total_noligand_tokens=5000) -> None:
    """
    Purpose:
        Generates a Slurm batch script for GPU-based processing. This script includes directives for GPU allocation along with other job parameters.

    Parameters:
        job_name: The name of the job, used to label the Slurm job.
        path_output_log: File path for capturing standard output.
        path_error_log: File path for capturing error messages.
        path_slurm_batch: The file path where the GPU Slurm batch script will be saved.
        command: The command string (or mapping of command details) that will be executed on the GPU node.
        total_noligand_tokens: The total number of non-ligand tokens in the input sequence. This value is used to determine the number of GPUs required for the job.
    """

    if total_noligand_tokens <= 5000:
        slurm_script = read_txt('./utils/example_slurmtitle_text/slurm_gpu.txt')
        slurm_script.append(f'#SBATCH --job-name={job_name}')
        slurm_script.append(f'#SBATCH --output={path_output_log}')
        slurm_script.append(f'#SBATCH --error={path_error_log}')
        slurm_script.append('#SBATCH --time=5:0:0')
        slurm_script.append('#SBATCH --mem=50GB')
        slurm_script.append('\n')
        slurm_script.append(command)

        # num_gpu = 1
        # slurm_script = ['#!/bin/bash',
        #             f'#SBATCH --job-name={job_name}',
        #             '#SBATCH -p a100',
        #             f'#SBATCH --gres=gpu:{num_gpu}',
        #             '#SBATCH --qos=a100_qos',
        #             '#SBATCH --nodelist euctemon',
        #             f'#SBATCH --output={path_output_log}',
        #             f'#SBATCH --error={path_error_log}',
        #             '#SBATCH --time=5:0:0',
        #             '#SBATCH --mem=50GB',
        #             '\n',
        #             command]

    elif total_noligand_tokens > 5000:
        # num_gpu_extra = math.ceil((total_noligand_tokens - 5000)/1500)
        # num_gpu = num_gpu_extra + 1
        print('WARNING! your token probability higher than 5120, it will use more than 2 GPUS')

        slurm_script = read_txt('./utils/example_slurmtitle_text/slurm_gpu_above5000.txt')
        slurm_script.append(f'#SBATCH --job-name={job_name}')
        slurm_script.append(f'#SBATCH --output={path_output_log}')
        slurm_script.append(f'#SBATCH --error={path_error_log}')
        slurm_script.append('\n')
        slurm_script.append(command)

        # slurm_script = ['#!/bin/bash', 
        #             f'#SBATCH --job-name={job_name}',
        #             '#SBATCH -p h100',
        #             '#SBATCH --nice=200',
        #             f'#SBATCH --gres=gpu:3',
        #             '#SBATCH --qos=eight_h100_qos',
        #             '#SBATCH --nodelist herakles',
        #             f'#SBATCH --output={path_output_log}',
        #             f'#SBATCH --error={path_error_log}',
        #             '#SBATCH --time=3-00:0:0',
        #             '#SBATCH --mem=50GB',
        #             '\n',
        #             command,
        #             ]
    else:
        raise ValueError(f'total non ligand token {total_noligand_tokens} should be integer')

    #slurm_script = ['#!/bin/bash', 
     #               f'#SBATCH --job-name={job_name}',
     #              '#SBATCH -p h100',
     #               f'#SBATCH --gres=gpu:{num_gpu}',
     #              '#SBATCH --qos=eight_h100_qos',
     #                '#SBATCH --nodelist herakles',
     #               f'#SBATCH --output={path_output_log}',
     #               f'#SBATCH --error={path_error_log}',
     #               '#SBATCH --time=5:0:0',
     #               '#SBATCH --mem=50GB',
     #               '\n',
     #               command]
    write_script_to_file(slurm_script, path_slurm_batch)


def create_slurm_bash_forcpu(job_name:str,path_slurm_logs:str,path_slurm_batch_cpu:str) -> str:
    """
    Purpose:
        Creates a bash script to submit the CPU job to Slurm. This script includes the necessary commands to execute the Slurm batch script for CPU processing.

    Parameters:
        job_name: The name of the job, used in the script header.
        path_slurm_logs: Directory where Slurm logs are stored.
        path_slurm_batch_cpu: Path to the Slurm batch script for CPU processing.
    """

    path_slurm_logs = os.path.join(path_slurm_logs, job_name)
    # os.makedirs(path_slurm_logs)

    path_slurm_batch_throughbash = os.path.join(path_slurm_logs, f'{job_name}_bashtoslurm.sh')

    # slurm_bash_script = ['#!/bin/bash',
    #                      f'sbatch {path_slurm_batch_cpu}']
    
    # with open(path_slurm_batch_throughbash, 'w') as file:
    #     for line in slurm_bash_script:
    #         file.write(line)
    #         file.write('\n')
    return path_slurm_batch_throughbash


def submit_msajob_toslurm(dict_paths: Mapping[str,str],
                          path_slurm_logs:str,
                          job_name:str,
                          total_noligand_tokens:int) -> None:

    """
    Purpose:
    Submits a complete MSA job to Slurm by:
        Generating unique log directories for the job.
        Creating both CPU and GPU Slurm batch scripts using the previously defined helper functions.
        Submitting the CPU batch script to Slurm via the sbatch command.

    Parameters:

        dict_paths: A mapping containing critical paths for the job:
        path_rewrotejson: Path to the rewritten JSON file.
        dir_completejson: Directory for complete JSON/MSA results.
        output_dir: Directory for final outputs.
        job_name_original: The original (sanitized) job name.
        path_slurm_logs: The base directory where Slurm logs (output, error, and batch scripts) will be stored.
        job_name: A unique job name identifier used to create subdirectories and file names for logs and scripts.
        total_noligand_tokens: The total number of non-ligand tokens in the input sequence. This value is used to determine the number of GPUs required for the job.

    """
    
    path_rewrotejson = dict_paths['path_rewrotejson']
    dir_completejson = dict_paths['dir_completejson']
    output_dir = dict_paths['output_dir']
    job_name_original = dict_paths['job_name_original']
    path_slurm_logsab5000 = dict_paths['path_slurm_logsab5000']
    path_slurm_bash = dict_paths['path_slurm_bash']
    path_slurm_bash_cpu = dict_paths['path_slurm_bash_cpu']

    print('path_slurm_logs ',path_slurm_logs)
    print('job_name',job_name)
    
    if total_noligand_tokens <= 5000:
        path_slurm_logs = os.path.join(path_slurm_logs, job_name)
    else:
        path_slurm_logs = os.path.join(path_slurm_logsab5000, job_name)

    os.makedirs(path_slurm_logs)

    path_output_log_cpu = os.path.join(path_slurm_logs, f"{job_name}_cpu.out")
    path_error_log_cpu = os.path.join(path_slurm_logs, f'{job_name}_cpu.err')
    path_slurm_batch_cpu = os.path.join(path_slurm_logs, f'{job_name}_cpu.sh')
    path_slurm_gather_cpu = os.path.join(path_slurm_bash_cpu, f'{job_name}_cpu.sh')
    
    path_output_log_gpu = os.path.join(path_slurm_logs, f'{job_name}_gpu.out')
    path_error_log_gpu = os.path.join(path_slurm_logs, f'{job_name}_gpu.err')
    path_slurm_batch_gpu = os.path.join(path_slurm_logs, f'{job_name}_gpu.sh')

    cmd_cpu = run_msa_byaf3(job_name,path_rewrotejson, 
                            dir_completejson,
                            path_slurm_batch_gpu, 
                            total_noligand_tokens)
    
    cmd_gpu = run_af3_inference(job_name_original, 
                                dir_completejson, 
                                output_dir, 
                                path_slurm_logs, 
                                total_noligand_tokens)


    if total_noligand_tokens >5000:
        path_gpu_ori = os.path.join(path_slurm_logs, f'{job_name_original}_primary_gpu.sh')
        path_gpu_des = os.path.join(path_slurm_bash, f'{job_name_original}_primary_gpu.sh')
        cmd_gpu = [''.join(cmd_gpu)]
        write_script_to_file(cmd_gpu, path_gpu_ori)
        cmd_gpu = cmd_gpu[0]
        cmd_mv = ['\n','cp ', path_gpu_ori, ' ' ,path_gpu_des]
        cmd_cpu += cmd_mv
        cmd_cpu = ''.join(cmd_cpu)


    print('dir_completejson', dir_completejson)
    print('output_dir', output_dir)
    print('path_output_log_cpu ', path_output_log_cpu)

    create_slurm_script_gpu(job_name=job_name,
                            path_output_log=path_output_log_gpu,
                            path_error_log=path_error_log_gpu,
                            path_slurm_batch = path_slurm_batch_gpu,
                            command=cmd_gpu,
                            total_noligand_tokens=total_noligand_tokens)

    create_slurm_script_cpu(job_name=job_name,
                            path_output_log=path_output_log_cpu,
                            path_error_log=path_error_log_cpu,
                            path_slurm_batch = path_slurm_batch_cpu,
                            path_slurm_bash_cpu=path_slurm_gather_cpu,
                            command=cmd_cpu)
    
    #create bash script to submit the CPU job
    # path_slurm_batch_throughbash =create_slurm_bash_forcpu(job_name=job_name,
    #                                                        path_slurm_logs=path_slurm_logs,
    #                                                        path_slurm_batch_cpu=path_slurm_batch_cpu)


    # # Submit the CPU job to Slurm through bash
    # result = subprocess.run(["sbatch", path_slurm_batch_cpu], capture_output=True, text=True, check=True)

    # return result
