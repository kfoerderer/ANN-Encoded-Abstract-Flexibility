#!/bin/env python3
from __future__ import print_function
import os
import jsonpickle
import re

for experiment_class in sorted(os.listdir('.')):
    results = {}
    if not os.path.isdir(experiment_class):
        continue
    for directory in os.listdir(experiment_class):
        if os.path.isdir(os.path.join(experiment_class,directory)) and re.match("\d+", os.path.basename(directory)):
            with open(os.path.join(experiment_class, directory, 'run.json'), 'r') as runfile:
                runobj = jsonpickle.decode(runfile.read())
            with open(os.path.join(experiment_class, directory, 'config.json'), 'r') as configfile:
                configobj = jsonpickle.decode(configfile.read())
            try:
                with open(os.path.join(experiment_class, directory, 'info.json'), 'r') as infofile:
                    infoobj = jsonpickle.decode(infofile.read())
            except:
                infoobj = {}
    
            id = int(os.path.basename(directory))
            name = runobj['experiment']['name']
            status = runobj['status']
            dataset = configobj['dataset_filename']
    
            results_per_dataset = results.setdefault(dataset, {})
            results_per_name = results_per_dataset.setdefault(name, {})
    
            if status != "COMPLETED":
                results_per_name[id] = [status]
            else:
                results_per_name[id] = [runobj['result'],"",""]
                if 'mae' in infoobj:
                    results_per_name[id][1] = infoobj['mae']
                if 'keras_history' in infoobj:
                    if type(infoobj['keras_history']) == list:
                        results_per_name[id][2] = len(infoobj['keras_history'][0]['loss'])
                    else:
                        results_per_name[id][2] = len(infoobj['keras_history']['loss'])
    
    print("######################################")
    print("### {:<30} ###".format(experiment_class))
    print("######################################")
    print("    {:36} {:>3} {:<20} {:<20} {}".format("experiment_name", "id", "RMSE", "MAE", "epochs"))
    print("----------------------------------------------------------------------------------------------")
    
    for dataset in sorted(results):
        results_per_dataset = results[dataset]
        print("dataset " + dataset + ": ")
        for experiment_name in sorted(results_per_dataset):
            results_per_name = results_per_dataset[experiment_name]
            for id in sorted(results_per_name):
                result = results_per_name[id]
                if len(result) == 1:
                    print("    {:36} {:3} {:<20}".format(experiment_name, id, result[0]))
                elif len(result) == 3:
                    try:
                        print("    {:36} {:3} {:<20} {:<20} {}".format(experiment_name, id, result[0], result[1], result[2]))
                    except:
                        print(result[0])
                        print(result[1])
                        print(result[2])
                else:
                    print("    {:36} {:3} {:<20}".format(experiment_name, id, result))

        print("")

    print("----------------------------------------------------------------------------------------------")
    print()
    print()
