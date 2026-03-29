import subprocess
from collections.abc import Mapping
from data import pipeline

import pathlib 
import os

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


def create_slurm_manager(configure_slurm:pipeline.ConfigureForSlurm | None,
                              path_slurm_bash:str,
                              gpu_restriction=4,
                              slepp_time = '1m') -> None:
    """
    Purpose:
    Creates a Slurm job manager script that will handle the execution of jobs in a Slurm cluster environment.
    This script is designed to manage the execution of jobs by checking for available GPUs/CPUs and submitting jobs accordingly.
    Parameters:
        configure_slurm (pipeline.ConfigureForSlurm): Configuration object containing paths and settings for Slurm.
        path_slurm_bash (str): Path to the directory where Slurm bash scripts will be stored.
        gpu_restriction (int): Number of GPUs to restrict the job to. Default is 4.
    """
    if len(os.listdir(path_slurm_bash)) == 0:
        sleep_time = slepp_time
    else:
        sleep_time = '0s'

    path_array_slurm_dir = configure_slurm.path_array_slurm_dir
    path_slurm_array = configure_slurm.path_slurm_array
    path_exectued_jobs = configure_slurm.path_exectued_jobs
    path_output_log = configure_slurm.path_output_log
    path_error_log = configure_slurm.path_error_log
    interval = configure_slurm.interval
    os.mkdir(path_array_slurm_dir)
    

    slurm_script = read_txt('./utils/example_slurmtitle_text/slurm_cpu.txt')
    slurm_script.append(f'#SBATCH --output={path_output_log}')
    slurm_script.append(f'#SBATCH --error={path_error_log}')
    slurm_script.append('#SBATCH --time=1-00:0:0')
    slurm_script.append('#SBATCH --mem=5GB')
    slurm_script.append('\n')

    # slurm_script = ['#!/bin/bash', 
    #             f'#SBATCH --job-name=array_test',
    #             '#SBATCH -p cpu',
    #             '#SBATCH --qos=cpu_qos',
    #             f'#SBATCH --output={path_output_log}',
    #             f'#SBATCH --error={path_error_log}',
    #             '#SBATCH --time=3-00:0:0',
    #             '#SBATCH --mem=5GB',
    #             '\n']

    with open(path_slurm_array, 'w') as f:
        for i in slurm_script:
            f.write(i + '\n')
        f.write(f'SCRIPT_DIR="{path_slurm_bash}" \n')
        f.write(f'LOGFILE="{path_exectued_jobs}" \n')
        f.write(f'GPU_LIMIT={gpu_restriction} \n')
        f.write(f'INTERVAL={interval}   # seconds between scans \n')
        f.write(f'sleep {sleep_time} \n')
        with open('./utils/slurm_manager.sh', 'r') as r_manager:
            lines = r_manager.readlines()
            for line in lines:
                f.write(line)
    result = subprocess.run(["sbatch", path_slurm_array], capture_output=True, text=True, check=True)
