# Use a minimal base image
FROM debian:bullseye-slim

# Set environment variables for locale and to include conda in PATH
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies in one layer
RUN apt update --quiet \
    && apt install --yes --quiet software-properties-common \
    && apt install --yes --quiet git wget gcc g++ make zlib1g-dev zstd

# Download and install Miniforge silently (-b flag for batch mode, -p to set install location)
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniforge3-Linux-x86_64.sh

# Copy the repository files into the container (adjust the context as needed)
COPY . /app

# Copy the af3_msa 3 source code from the local machine to the container and
# set the working directory to there.
WORKDIR /app

# Create the conda environment from the YAML file and clean up
# RUN /opt/conda/bin/conda env create -f af3_msa_requirement.yml
COPY af3_msa_requirements.txt .

RUN pip3 install -r af3_msa_requirements.txt

# Use conda run to automatically activate the environment af3_msa
CMD ["python", "run_af3_msa.py"]


