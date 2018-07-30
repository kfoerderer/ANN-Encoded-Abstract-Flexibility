#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import division, print_function, unicode_literals
from sacred import Experiment
from sacred.observers import FileStorageObserver

import numpy as np

ex = Experiment('sacred_prepare_classification_data')
ex.observers.append(FileStorageObserver.create('prepare_data'))

@ex.config
def my_config():
    seed = 1924
    aggregate_to_15min = True

    BOUNDARIES_FEASIBILITY = {
        'BESS': {
            'max': (89 / 1000, 89 / 1000),
            'mean': (89 / 1000, 89 / 1000)
        },
        'CHP': {
            'max': (1050 / 1000, 1050 / 1000),
            'mean': (44 / 1000, 350 / 1000)
        },
        'BESS_CHP': {
            'max': (1139 / 1000, 1139 / 1000),
            'mean': (133 / 1000, 439 / 1000)
        }
    }

    # data sources
    SCHEDULE_FILES = {
        'BESS': [
            './generated_dataset/17-12-29_33000_random_schedules_battery.npy',
            './generated_dataset/2018-01-05T17:49:00_429000_arbitrary_schedules_validated_33000_battery.npy'
        ],
        'CHP': [
            './generated_dataset/2018-01-07T02:06:00_33000_random_schedules_chp.npy',
            './generated_dataset/2018-01-05T17:49:00_429000_arbitrary_schedules_validated_33000_chp.npy'
        ],
        'BESS_CHP': [
            './generated_dataset/2018-01-07T04:29:00_36300_random_schedules_chp_battery_validated_36300_chp_battery.npy',
            './generated_dataset/2018-01-05T17:49:00_429000_arbitrary_schedules_validated_20227_chp_battery.npy'
        ]
    }

    # Parameter
    configuration = 'BESS_CHP'

def preprocess_data(timestamp, configuration, aggregate_to_15min, SCHEDULE_FILES, BOUNDARIES_FEASIBILITY):
    schedule_files = SCHEDULE_FILES[configuration]

    data = []
    i = 0
    for file in schedule_files:
        data_fragment = np.load(file)

        for schedule in data_fragment:
            feasible = False
            load = []
            state = []

            # schedules need to be validated if there is no feasibility info
            # or if they are infeasible to assure infeasibility
            if len(schedule[4]) == 0 or schedule[4][287] == False:

                # use validation rules
                if len(schedule[4]) == 0 and len(schedule[5]) == 0:
                    data = []
                    print('Missing data in file "%s".' % file)
                    break
                elif len(schedule[5]) == 0:
                    # this dataset has not been validated which is fine for the 1 DER configurations
                    if configuration == 'BESS_CHP':
                        print('Warning: Using unvalidated input from file "%s" for a system with multiple DER.' % file)

                    feasible = False
                else:
                    # apply validation rules
                    if (schedule[5][0] < BOUNDARIES_FEASIBILITY[configuration]['max'][0]
                            and np.mean(np.abs(schedule[5][1] - schedule[3])) <
                            BOUNDARIES_FEASIBILITY[configuration]['mean'][0]):

                        feasible = True
                    elif (schedule[5][0] >= BOUNDARIES_FEASIBILITY[configuration]['max'][1]
                          or np.mean(np.abs(schedule[5][1] - schedule[3])) >=
                          BOUNDARIES_FEASIBILITY[configuration]['mean'][1]):

                        feasible = False
                    else:
                        # this schedule is filtered
                        print('Filtering schedule')
                        continue

            else:
                feasible = schedule[4][287]  # = True

            # aggregate from 5 min to 15 min
            if aggregate_to_15min:
                load = np.array(schedule[3]).reshape((96, 3)).mean(axis=1)
            else:
                load = np.array(schedule[3])

            state = (schedule[0][1] == 'Winter',
                     schedule[0][1] == 'Transition',
                     schedule[0][1] == 'Summer',
                     schedule[0][5],  # chp storage SOC
                     schedule[0][7])  # bess SOC

            data.append(np.concatenate((np.array(state), load, np.array([feasible, i]))))

        i += 1

    # shuffle data
    data = np.array(data)
    np.random.shuffle(data)

    # save file
    if aggregate_to_15min:
        filename = './training/' + timestamp + '_classification_15min_' + configuration.lower()
    else:
        filename = './training/' + timestamp + '_classification_05min_' + configuration.lower()
    np.save(filename, np.array(data))

    return filename

@ex.automain
def generate_all_data(SCHEDULE_FILES, BOUNDARIES_FEASIBILITY):
    from datetime import datetime
    timestamp = datetime.now().replace(second=0, microsecond=0).isoformat()

    output_filepaths = []

    for config in ['BESS','CHP','BESS_CHP']:
        for aggregate in [True, False]:
            output_filepaths.append(
                preprocess_data(timestamp, config, aggregate, SCHEDULE_FILES, BOUNDARIES_FEASIBILITY)
            )

    ex.info['timestamp'] = timestamp
    ex.info['output_filepaths'] = output_filepaths
    #ex.add_artifact(filename, name='classification_' + output_file)

    return output_filepaths
