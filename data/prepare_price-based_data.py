#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from sacred import Experiment
from sacred.observers import FileStorageObserver

import numpy as np

ex = Experiment('sacred_prepare_price-based_data')
ex.observers.append(FileStorageObserver.create('prepare_data'))

@ex.config
def my_config():
    seed = 1924
    
    OPTIMIZED_SCHEDULE_FILES = {
        'BESS': [
            './generated_dataset/18-01-03_cheapest_schedules_battery.npy'
        ],
        'CHP': [
            './generated_dataset/18-01-09_cheapest_schedules_chp.npy'
        ],
        'BESS_CHP': [
            './generated_dataset/18-01-09_cheapest_schedules_chp_battery.npy'   ]
    }

    # Parameter
    configuration = 'BESS_CHP'

def preprocess_data(timestamp, configuration, OPTIMIZED_SCHEDULE_FILES):
    schedule_files = OPTIMIZED_SCHEDULE_FILES[configuration]

    data = []
    for file in schedule_files:
        data_fragment = np.load(file)
        
        for schedule in data_fragment:
            if len(schedule[1]) == 0:
                print('Missing data in file "%s".' % file)
                break;
                
            # fix scaling, transform to kW
            load = schedule[3][0] #/ 1000
            
            state = []
            prices = []

            # aggregate from 5 min to 15 min
            load = load.reshape((96, 3)).mean(axis=1)
            
            state = (schedule[0][1] == 'Winter',
                    schedule[0][1] == 'Transition',
                    schedule[0][1] == 'Summer',
                    schedule[0][5], # chp storage SOC
                    schedule[0][7]) # bess SOC
            
            prices = schedule[1]
            
            data.append(np.concatenate((np.array(state), np.array(prices)/100, load)))

    # shuffle data
    data = np.array(data)
    np.random.shuffle(data)

    # save file
    filename = './training/' + timestamp + '_price-based_15min_' + configuration.lower()

    np.save(filename, np.array(data))

    return filename

@ex.automain
def generate_all_data(OPTIMIZED_SCHEDULE_FILES):
    from datetime import datetime
    timestamp = datetime.now().replace(second=0, microsecond=0).isoformat()

    output_filepaths = []

    for config in ['BESS','CHP','BESS_CHP']:
        output_filepaths.append(
            preprocess_data(timestamp, config, OPTIMIZED_SCHEDULE_FILES)
        )

    ex.info['timestamp'] = timestamp
    ex.info['output_filepaths'] = output_filepaths
    #ex.add_artifact(filename, name='classification_' + output_file)

    return output_filepaths