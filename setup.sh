sudo apt update
sudo apt install python3-dev python3-pip
sudo apt install libatlas-base-dev
sudo pip3 install -U virtualenv

virtualenv -p python3.7 /home/pi/ece498icc_env
. /home/pi/ece498icc_env/bin/activate
pip install --upgrade pip
pip install --upgrade numpy matplotlib h5py tensorflow==1.14
deactivate
