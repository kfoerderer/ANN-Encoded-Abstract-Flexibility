#!/bin/env python3
from __future__ import print_function
import os
import jsonpickle
import re

results = {}

for directory in os.listdir('.'):
    if os.path.isdir(directory) and re.match("\d+", os.path.basename(directory)):
        with open(os.path.join(directory, 'run.json'), 'r') as runfile:
            runobj = jsonpickle.decode(runfile.read())
        with open(os.path.join(directory, 'config.json'), 'r') as configfile:
            configobj = jsonpickle.decode(configfile.read())

        id = os.path.basename(directory)
        name = runobj['experiment']['name']
        status = runobj['status']
        dataset = configobj['dataset_filename']

        results_per_dataset = results.setdefault(dataset, {})
        results_per_name = results_per_dataset.setdefault(name, {})

        if status != "COMPLETED":
            results_per_name[id] = status
        else:
            results_per_name[id] = runobj['result']

print("dataset --")
print("    {:36} {:3} {}".format("experiment_name", "id", "result"))
print("---------------------------------------------------------------------------------")

for dataset in sorted(results):
    results_per_dataset = results[dataset]
    print("dataset " + dataset + ": ")
    for experiment_name in sorted(results_per_dataset):
        results_per_name = results_per_dataset[experiment_name]
        for id in sorted(results_per_name):
            result = results_per_name[id]
            print("    {:36} {:3} {}".format(experiment_name, id, result))
    print("")