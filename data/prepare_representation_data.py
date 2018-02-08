#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from sacred import Experiment
from sacred.observers import FileStorageObserver

import numpy as np

ex = Experiment('sacred_prepare_representation_data')
ex.observers.append(FileStorageObserver.create('prepare_data'))

@ex.config
def my_config():
    seed = 1924
 
    # data sources
    REPRESENTATION_SCHEDULE_FILES = {
        # control sequence
        'CS': {
            'BESS': ['./generated_dataset/17-12-30_33000_genome_based_schedules_battery_c.npy'],
            'CHP': ['./generated_dataset/2018-01-09T12:28:00_21000_genome_based_schedules_chp_c.npy']
        },
        # mode + time
        'MT': {
            'BESS': ['./generated_dataset/2018-01-11T19:49:00_33000_genome_based_schedules_battery.npy'],
            'CHP': ['./generated_dataset/2018-01-10T04:38:00_18000_genome_based_schedules_chp.npy']
        }
    }

    # Parameter
    configuration = 'BESS_CHP'
    representation_mode = 'MT'# MT, CS

def preprocess_data(timestamp, configuration, REPRESENTATION_SCHEDULE_FILES, representation_mode):
    # add additional sources here
    schedule_files = REPRESENTATION_SCHEDULE_FILES[representation_mode][configuration]

    data = []
    for file in schedule_files:
        data_fragment = np.load(file)
        
        for schedule in data_fragment:
                
            if len(schedule[2]) == 0:
                print('Missing representation data in file "%s".' % file)
                break;
    
            representation = []
            state = []
            load = []
                
            # aggregate from 5 min to 15 min
            load = np.array(schedule[3]).reshape((96, 3)).mean(axis=1)
            
            state = (schedule[0][1] == 'Winter',
                    schedule[0][1] == 'Transition',
                    schedule[0][1] == 'Summer',
                    schedule[0][5], # chp storage SOC
                    schedule[0][7]) # bess SOC

            genome = np.array(schedule[2])

            if representation_mode == 'MT':
                if configuration == 'BESS':
                    representation = np.concatenate((genome[0]/5, genome[1]/48))
                else:
                    representation = np.concatenate((genome[0], genome[1]/48))
            elif representation_mode == 'CS':
                if configuration == 'BESS':
                    representation = (genome[0].reshape(96,15).mean(axis=1))/5
                else:
                    representation = genome[0]

            data.append(np.concatenate([np.array(state), representation, load]))

    # shuffle data
    data = np.array(data)
    np.random.shuffle(data)

    # save file
    filename = './training/' + timestamp + '_representation_' + representation_mode + '_15min_' + configuration.lower()
    np.save(filename, np.array(data))

    return filename

@ex.automain
def generate_all_data(REPRESENTATION_SCHEDULE_FILES):
    from datetime import datetime
    timestamp = datetime.now().replace(second=0, microsecond=0).isoformat()

    output_filepaths = []

    for config in ['BESS','CHP']:
        for representation_mode in ['CS', 'MT']:
            out_filename = preprocess_data(timestamp, config, REPRESENTATION_SCHEDULE_FILES, representation_mode)
            print("{} \\".format(out_filename))
            output_filepaths.append(out_filename)

    ex.info['timestamp'] = timestamp
    ex.info['output_filepaths'] = output_filepaths
    #ex.add_artifact(filename, name='classification_' + output_file)

    return output_filepaths