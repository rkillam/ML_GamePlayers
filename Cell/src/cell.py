#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""Main execution file that starts the Cell game

Usage:
    cell.py [--conf=<CONF_FILE>] [(-d | --debug) | (-v | --verbose)]
    cell.py -h | --help

Options:
    --conf=<CONF_FILE>  The config file that specifies game settings [default: cell.conf]
    -h --help           Show this help message
    -d --debug          Show all logs
    -v --verbose        Show warning logs
"""

import controller
import model
import utils
import view

import configparser
import logging
import numpy as np


def main(conf_file, debug, verbose):
    if debug:
        utils.set_log_level(logging.DEBUG)
    elif verbose:
        utils.set_log_level(logging.WARNING)

    config = configparser.ConfigParser()
    config.read(conf_file)

    processing_config = config['Processing']
    utils.NUM_WORKERS = int(processing_config['num_workers'])

    world_config = config['World']

    num_rows = int(world_config['rows'])
    num_cols = int(world_config['cols'])

    num_pellets = int(world_config['food_pellets'])
    num_herbivors = int(world_config['herbivors'])

    world_model = model.World((num_rows, num_cols))
    world_view = view.PygameWorld(world_model, 'Cell')

    for entity_type, entity_count in ((model.FoodPellet, num_pellets), (model.Herbivor, num_herbivors)):
        for _ in range(entity_count):
            y = np.random.randint(world_model.dimensions[0])
            x = np.random.randint(world_model.dimensions[1])

            entity = entity_type.from_coordinates((y, x))
            world_model.add_entity(entity)

    controller.PygameEventController()
    controller.ClockTickController().run()


if __name__ == '__main__':
    import docopt
    args = docopt.docopt(__doc__)

    main(args['--conf'], args['--debug'], args['--verbose'])
