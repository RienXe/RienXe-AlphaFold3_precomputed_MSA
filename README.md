# AlphaFold 3 seek pre-computed MSA

This script servers as a broker, it aims to reduce MSA step for human protein or even mamalian protein by using pre-computed MSA database. 
This broker will take json file or directory of json file(s) of AlphaFold3 or AlphaFold server as input, blast the sequence and use pre-computed MSA file 
to facilitate the MSA pipeline of AlphaFold 3. The detail of process can find in the bottom. 

# Optional -- install miniconda to kosmos
Download miniconda3
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```
Set HOME path to ur desired directory
```
export HOME=/path/to/your/choice/
```
install miniconda3
```
bash path/to/downloaded miniconda3/Miniconda3-latest-Linux-x86_64.sh
```
press `ENTER`, go through terms, enter `yes`
press `ENTER` if it ask for path to install, then the miniconda will installed onto your HOME (path/to/your/choice)

you can set miniconda automatically started when you enter the kosmos, by typing `yes` 

activate the conda by 
```
source ~/miniconda3/bin/activate
```
Then, initialize conda on all available shells by running the following command:

```
conda init --all
```
Change the conda default paths for packages folders and envs folders:

```
mkdir -p ~/.conda/pkgs
conda config --add pkgs_dirs ~/.conda/pkgs

mkdir -p ~/.conda/envs
conda config - add envs_dirs ~/.conda/envs
```


# How to install the af3_msa
Clone the repository
```
git clone https://github.com/NKI-AI/alphafold3-nki/edit/main/af3_msa_byren.git
```

Create new virtural environment
```
conda env create --name af3_msa -f af3_msa_requirement.yml
```
# How to use the af3_msa through docker
Build the Docker container. This builds a container with all the right python dependencies:
```
docker build -t af3_msa -f ./docker/Dockerfile .
```
The following command will be use to run af3_msa through docker

```
docker run -it \
    --volume /path/to/json_file_or_directory_of_json:/root/af_input \
    --volume /path/to/af3_prediction:/root/af_output \
    --volume /path/to/save_intermediating_result:/root/process \
    --volume /data/groups/public/archive/humanPPI_MSA/humanPPI_MSA/MSA:/root/MSA \
    af3_msa \
    python run_af3_msa.py \
    --path_input=/root/af_input \
    --path_outdir=/root/af_output \
    --path_intermediate_result=/root/process \
    --path_msa_db=/root/MSA 
```



# How to use the af3_msa
Activate virtual environment

```
conda activate af3_msa
```
Then use the following comment to run the script:
```
python run_af3_msa.py \
  --path_input /path/to/json_file_or_directory_of_json \
  --path_outdir /path/to/af3_prediction \
  --path_db /path/to/af3_msa_byren/review_database/uniprotkb_taxonomy_id_9606_AND_reviewed_2025_02_06.fasta \
  --path_msa_db /path/to/huamnPPI_MSA/ \
  --path_intermediate_result /path/to/save_intermediating_result \
  --binary_blastp_script /path/to/blastp \
  --binary_reformat_script /path/to/af3_msa_byren/reformat.pl \
 
```
* `--path_input`: input path to single AlphaFold3 json file, or folder of multiple AlphaFold3 json file. The format of json file can be both alphafold server and alphafold 3
* `--path_outdir`: output path to the place to store AlphaFold3 prediction
* `--path_db`: the path of Uniprot Human protein database, the default place is within `./review_database/uniprotkb_taxonomy_id_9606_AND_reviewed_2025_02_06.fasta`
* `--path_msa_db`: the path to pre-computed MSA, each file with Uniprot-ID. the default path is `/data/groups/public/archive/humanPPI_MSA/humanPPI_MSA/MSA/`
* `--path_intermediate_result`: the place to store intermediating result. It contains, json file after fill the path of UnpairedMSA with pre-computed msa and json file output of MSA output of AlphaFold 3. The place can be temporary delete after job is submitted. 
* `--binary_blastp_script`: binary path to blastp program from BLAST+ the default path is `/path/to/af3_msa/blast/bin/blastp`
* `--binary_reformat_script`: binary path to reformat.pl, the script to convert fas file to a3m file, which is mandate for AlphaFold 3 as MSA file. the default path is `/path/to/af3_msa/reformat.pl`

Some other aspects:
* The program will skip the already predicted job. This process is done through check whether the folder with the job name is existed in output directory, as well as whether the `{job name}_model.cif` is exist within that folder.
* For total length of proteins higher than 5120 (it will use unified GPU memory), be caution, the prediciton will slow and result may not reliable. ONLY DO IT IN CASES OF REAL NECESSITY !!

# How the broker works

![alt text](https://github.com/NKI-AI/alphafold3-nki/blob/main/af3_msa_byren/docs/process_graph.png)
