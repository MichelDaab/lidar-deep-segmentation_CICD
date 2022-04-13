FROM nvidia/cuda:10.1-devel-ubuntu18.04 
# An nvidia image seems to be necessary for torch-points-kernel. Also, a "devel" image seems required for the same library


# set the IGN proxy, otherwise apt-get and other applications don't work 
ENV http_proxy 'http://192.168.4.9:3128/'
ENV https_proxy 'http://192.168.4.9:3128/'

# set the timezone, otherwise it asks for it... and freezes
ENV TZ=Europe/Paris
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# all the apt-get installs
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
        software-properties-common  \
        wget                        \
        git                         \
        # postgis                     \
        # pdal                        \
        libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6   # package needed for anaconda

# install anaconda
RUN wget --quiet https://repo.anaconda.com/archive/Anaconda3-2021.11-Linux-x86_64.sh -O ~/anaconda.sh
RUN /bin/bash ~/anaconda.sh -b -p /opt/conda && \
    rm ~/anaconda.sh
ENV PATH /opt/conda/bin:$PATH

WORKDIR /lidar

# copy all the data now (because the requirements files are needed for anaconda)
COPY . .
# RUN git clone https://MichelDaab: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@github.com/IGNF/lidar-prod-quality-control.git

# install the python packages via anaconda
RUN conda env create -f bash/setup_environment/requirements.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "lidar_multiclass", "/bin/bash", "-c"]

# install all the dependencies
RUN conda install -y pytorch=="1.10.1" torchvision=="0.11.2" -c pytorch -c conda-forge \
 && conda install pytorch-lightning==1.5.9 -c conda-forge \
 && pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.1+cpu.html torch-sparse -f https://data.pyg.org/whl/torch-1.10.1+cpu.html torch-geometric \
 && pip install torch-points-kernels --no-cache \
 && pip install torch torchvision \
 && conda install -y pyg==2.0.3 -c pytorch -c pyg -c conda-forge

# the entrypoint garanty that all command will be runned in the conda environment
ENTRYPOINT ["conda",                \   
            "run",                  \
            "-n",                   \
            "lidar_multiclass"]

CMD         ["python", \
            "-m", \
            "lidar_multiclass.predict", \
            "--config-path", \
            "/CICD_github_assets/.hydra", \ 
            "--config-name", \
            "predict_config_V1.6.3.yaml", \
            "predict.src_las=/CICD_github_assets/test/792000_6272000_subset_buildings.las", \
            "predict.output_dir=/output", \
            "predict.resume_from_checkpoint=/CICD_github_assets/checkpoints/epoch_033.ckpt", \
            "predict.gpus=0", \
            "datamodule.batch_size=10", \ 
            "datamodule.subtile_overlap=0", \ 
            "hydra.run.dir=/lidar"]

# just there as a note to mount the necessary store
# sudo mount -v -t cifs -o user=mdaab,domain=IGN,uid=24213,gid=10550 //store.ign.fr/store-lidarhd/projet-LHD/IA/Validation_Module/CICD_github_assets/B2V0.5 /home/MDaab/Data/CICD_github_assets/