# ANN-Encoded-Abstract-Flexibility
The repository for neural models from paper: "tbd"

Please note that you need [Git LFS](https://git-lfs.github.com/) to download the dataset during cloning.

## Contents

* [neuralnets](neuralnets): the neural network definitions for each usage patterns. There is a run script in each directory.
	* Pattern A: [classification](neuralnets/classification)
	* Pattern B: [loadprediction](neuralnets/loadprediction)
	* Pattern C: [representation](neuralnets/representation)
	* Pattern D: [repair](neuralnets/repair)
	* test of Pattern A with individual networks for each season: [classification3nets](neuralnets/classification3nets)
	* test of Pattern A with 5 minute time slots in load profiles: [classification5min](neuralnets/classification5min)
* [data](data)
	* [experiments](data/experiments): trained models, evaluation results and logs of the performed experiments. This was created using the FileObserver of [Sacred](http://sacred.readthedocs.io/).
    	* Pattern A: [classification](data/experiments/classification)
    	* Pattern B: [loadprediction](data/experiments/loadprediction)
    	* Pattern C: [representation](data/experiments/representation)
    	* Pattern D: [repaironlyinfeasible](data/experiments/repaironlyinfeasible)
    	* test of Pattern A with individual networks for each season: [classification3nets](neuralnets/classification3nets)
    	* test of Pattern A with 5 minute time slots in load profiles: [classification5min](neuralnets/classification5min)
    	* test of Pattern D, where the net has to repair both feasible and infeasible load profiles: [repair](data/experiments/repair)
	* [real_eshl_chp](data/real_eshl_chp): load profiles of the CHP from the [KIT Energy Smart Home Lab](http://organicsmarthome.org/)
    * [generated_dataset](data/generated_dataset): generated dataset of CHP and/or BESS load profiles
	* [training](data/training): input/output-vectors converted from [generated_dataset](data/generated_dataset) used for training and evaluating the neural networks
* [Experiment_Results.txt](Experiment_Results.txt): summary of our experiment results
* [Real_ESHL_CHP_Evaluation.ipynb](Real_ESHL_CHP_Evaluation.ipynb): validation of the best classification model using load profiles of a real CHP

## How to run the experiments

1. Unzip the dataset
```
$ cd data/generated_dataset; unzip dataset.zip; cd ../..
```

2. Create a Python 3 VirtualEnv and install the dependencies
```
$ virtualenv --python=python3 venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
```

3. Convert the dataset
```
(venv) $ cd data; ./run.sh; cd ..
```

4. Run the experiments
```
(venv) $ cd neuralnets; bash ./run_batch.sh ; cd ..
```

5. Create a summary of the experiment results. The results should look similar to [Experiment Results.txt](Experiment Results.txt).
```
(venv) $ cd data/experiments; python all_summery.py
```
