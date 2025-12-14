# Repo for ECE253

> 2 image preprocessing algoithms, SR_D (Compressive Sensing) and a MRA(Multi-resolutions Analasys)-based method GNet are used for training

- It turns out that GNet has a better result

# How to run the code

- Change the dataset in config map for 8 different datasets (we got 3 in our report)

- If it is full resolution(with no **GT**), please shift to the `trainer.MTF_Full_Train()` in `main.py`

- run `python main.py` to train

- All the other results as comparison are available on `https://drive.google.com/drive/folders/18P7z78KcRFTT_pUELxCcRrrIfA9VFOOq?usp=drive_link`
