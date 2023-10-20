usermod -s /bin/bash TA744
sudo apt-get update
sudo apt-get -y install python3.8-venv
sudo mkfs.ext4 /dev/sdb
sudo mkdir -p /mnt/data 
sudo mount /dev/sdb /mnt/data
sudo chmod ugo+rwx /mnt/data
cd /mnt/data

# python3.8 -m venv ./ptorch
# source ptorch/bin/activate

python3.8 -m venv ./ptorch
source ptorch/bin/activate
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
#sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
#wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
#sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
#sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
# some network thing I ended up copying
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
#sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
#sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
#sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
#sudo apt-get update
#sudo apt-get -y install cuda
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu1804-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda
pip install --upgrade pip
pip install boto3 orjson
TMPDIR=/mnt/data/pip_dir/ pip3 install --cache-dir=/mnt/data/pip_dir/ torch torchvision torchaudio


# cuda 10.2
#wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
#sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
#wget https://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
#sudo dpkg -i cuda-repo-ubuntu1804-10-2-local-10.2.89-440.33.01_1.0-1_amd64.deb
#sudo apt-key add /var/cuda-repo-10-2-local-10.2.89-440.33.01/7fa2af80.pub
#sudo apt-get update
#sudo apt-get -y install cuda
mkdir pip_dir
#pip3 install torch==1.10.0+cpu torchvision==0.11.1+cpu torchaudio==0.10.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
#pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
#TMPDIR=/mnt/data/pip_dir/ pip3 install --cache-dir=/mnt/data/pip_dir/ --build /mnt/data/pip_dir/ torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
sudo apt-get -y install python3.8-dev
sudo apt-get -y install libjpeg-dev zlib1g-dev
pip install cython
TMPDIR=/mnt/data/pip_dir/ pip3 install --cache-dir=/mnt/data/pip_dir/ --build /mnt/data/pip_dir/ torch torchvision torchaudio

