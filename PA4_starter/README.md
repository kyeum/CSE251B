# CSE251B PA4
## Image Captioning

## PA4 Organization and Instruction
We used jupyter notebooks for the main body for each part. Due to various experiment, we used differnet json file for the setting.  


* See `default.json` to see the each experiment's structure. 
* Implement factories to return project specific models, datasets based on config. Add more flags as per requirement in the config.
* Implement `experiment_LSTM.py` based on the project requirements. `experiment.py` is default file
* After defining the configuration (say `my_exp.json`) - simply run `python3 main.py my_exp` to start the experiment
* The logs, stats, plots and saved models would be stored in `./experiment_data/my_exp` dir. This can be configured in `constants.py`
* To resume an ongoing experiment, simply run the same command again. It will load the latest stats and models and resume training pr evaluate performance.

### Main Notebooks
1. Baseline LSTM was implemented in `main_LSTM-baseline.ipynb` - The correspond Json is `default_baseline.json`  
2. Best LSTM was implemented in `main_BESTLSTM.ipynb` - The correspond Json is `LSTM_DECAY_0.1.json`  
3. Best RNN was implemented in `main_vRNN.ipynb` - The correspond Json is `LSTM_DECAY_0.1.json`  

To run, simply open the jupyter notebooks with jupyter notebook and press `Run All`.

### Other files
- main.py: Main driver class
- experiment.py: Main experiment class. Initialized based on config - takes care of training, saving stats and plots, logging and resuming experiments.
- dataset_factory: Factory to build datasets based on config
- model_factory.py: Factory to build models based on config
- constants.py: constants used across the project
- file_utils.py: utility functions for handling files 
- caption_utils.py: utility functions to generate bleu scores
- vocab.py: A simple Vocabulary wrapper
- coco_dataset: A simple implementation of `torch.utils.data.Dataset` the Coco Dataset
- get_datasets.ipynb: A helper notebook to set up the dataset in your workspace


