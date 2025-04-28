# torch-2.0.1+cu180
conda create -n mega_sam python=3.10
conda activate mega_sam
conda install conda-forge::cudatoolkit=11.8
conda install nvidia/label/cuda-11.8.0::cuda-toolkit
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
pip install opencv-python tqdm imageio einops scipy matplotlib wandb timm ninja numpy huggingface-hub kornia ipython
wget https://anaconda.org/xformers/xformers/0.0.22.post7/download/linux-64/xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
conda install xformers-0.0.22.post7-py310_cu11.8.0_pyt2.0.1.tar.bz2
cd base
python setup.py install


# torch-2.4.0+cu124
conda create -n mega_sam python=3.10
conda activate mega_sam
conda install nvidia/label/cuda-12.4.1::cuda-toolkit
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
pip install opencv-python tqdm imageio einops scipy matplotlib wandb timm ninja numpy huggingface-hub kornia ipython
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers