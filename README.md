# Neural Functions for Learning Periodic Signal <br> (ICLR 2025)


## 0. Experimental environment settings.

Run the following code before starting the experiment.

    conda env create -f environment.yaml
    conda activate nert

## 1. Data processing.

         [ Code ]              [ Description of code ]

    python data_processing.py  : Code for data-preprocessing
    

## 2. Train / Test

Run the following code for NeRT training / testing.

         [ Code ]              [ Description of code ]

    sh run_longterm.sh      : Code for NeRT training [longterm time series]
    sh run_periodic.sh      : Code for NeRT training [periodic time series]

Detailed settings can be changed in config.py


You can train/test the NeRT in the setting below.

     [ Experimental default setting (longterm)] 
    
    time series dataset: national illness
    epoch: 10000
    train rate: 0.7
    validation rate: 0.15
    test rate: 0.15
    max scale: 1
    learning rate: 0.001
    optimizer: Adam
    inner frequency (w_inner): 1
    
    
     [ Experimental default setting (periodic)] 
    
    time series dataset: traffic
    epoch : 2000
    number of block: 1
    max scale: 1
    learning rate: 0.001
    optimizer: Adam
    inner frequency (w_inner): 1


## 3. Other code

Brief description of the other code files.

        [ Code ]        [ Description of code ]
        
        model.py   :  NeRT models. (multivariate, univariate)
