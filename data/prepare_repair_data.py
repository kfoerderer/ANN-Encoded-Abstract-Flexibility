#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from sacred import Experiment
from sacred.observers import FileStorageObserver

import numpy as np

ex = Experiment('sacred_prepare_repair_data')
ex.observers.append(FileStorageObserver.create('prepare_data'))

@ex.config
def my_config():
    seed = 1924

    skip_feasible = True
    
    # data sources
    SCHEDULE_FILES = {
        'BESS': [
            './generated_dataset/17-12-29_33000_random_schedules_battery_validated_33000_battery.npy',
            './generated_dataset/2018-01-05T17:49:00_429000_arbitrary_schedules_validated_33000_battery.npy'
        ],
        'CHP': [
            './generated_dataset/2018-01-07T02:06:00_33000_random_schedules_chp_validated_33000_chp.npy',
            './generated_dataset/2018-01-05T17:49:00_429000_arbitrary_schedules_validated_33000_chp.npy'
        ],
        'BESS_CHP': [
            './generated_dataset/2018-01-07T04:29:00_36300_random_schedules_chp_battery_validated_36300_chp_battery.npy',
            './generated_dataset/2018-01-05T17:49:00_429000_arbitrary_schedules_validated_20227_chp_battery.npy'
        ]
    }

    # Parameter
    configuration = 'BESS_CHP'

def preprocess_data(timestamp, configuration, SCHEDULE_FILES, skip_feasible):
        # add additional sources here
    schedule_files = SCHEDULE_FILES[configuration]

    print('Repair: %s' % configuration)
    print('Only infeasible schedules: %s' % skip_feasible)

    data = []
    skipped = 0
    for file in schedule_files:
        warning_displayed = False
        data_fragment = np.load(file)

        for schedule in data_fragment:
                    
            state = []
            load_input = []
            load_output = []
            
            if len(schedule[1]) != 0:
                # price optimized schedules -> load needs to be transformend
                load_input = schedule[3][0] / 1000
            else:
                load_input = np.array(schedule[3])
                
            if len(schedule[5]) == 0:
                # schedule has not been validated -> should be feasible
                # is it really feasible?
                if len(schedule[4]) == 0:
                    print('Missing data in file "%s".' % file)
                    break;
                    
                if schedule[4][287] == False:
                    if warning_displayed == False:
                        warning_displayed = True
                        print('Missing validation data in file "%s".' % file)
                        print('Skipping schedules with missing information.')
                    # skip
                    skipped += 1
                    continue
                else:
                    # it is feasible
                    if skip_feasible == True:
                        continue
                    else:
                        load_output = load_input
                    
                    
            else:
                # schedule has been validated (since it is infeasible)
                load_output = schedule[5][1]
            
            # aggregate from 5 min to 15 min
            load_input = load_input.reshape((96, 3)).mean(axis=1)
            load_output = load_output.reshape((96, 3)).mean(axis=1)
            
            state = (schedule[0][1] == 'Winter',
                    schedule[0][1] == 'Transition',
                    schedule[0][1] == 'Summer',
                    schedule[0][5], # chp storage SOC
                    schedule[0][7]) # bess SOC
            
            data.append(np.concatenate((np.array(state), load_input, load_output)))

    print('Skipped %d schedules.' % skipped)
            
    # shuffle data
    data = np.array(data)
    np.random.shuffle(data)

    # save file
    if skip_feasible:
        filename = './training/' + timestamp + '_repair_only_infeasible_15min_' + configuration.lower()
    else:
        filename = './training/' + timestamp + '_repair_without_price_opt_15min_' + configuration.lower()

    np.save(filename, np.array(data))

    return filename

@ex.automain
def generate_all_data(SCHEDULE_FILES, skip_feasible):
    from datetime import datetime
    timestamp = datetime.now().replace(second=0, microsecond=0).isoformat()

    output_filepaths = []

    for config in ['BESS','CHP','BESS_CHP']:
        output_filepaths.append(
            preprocess_data(timestamp, config, SCHEDULE_FILES, skip_feasible)
        )

    ex.info['timestamp'] = timestamp
    ex.info['output_filepaths'] = output_filepaths
    #ex.add_artifact(filename, name='classification_' + output_file)

    return output_filepaths