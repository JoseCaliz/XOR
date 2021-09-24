# -*- coding: utf-8 -*-
import click
from time import time
import numpy as np
from datetime import timedelta
import logging
import csv
import random
from pathlib import Path


@click.command()
@click.option('-s', '--size', required=True, default=100_000)
@click.option('--variable_length/--no_variable_length', default=False)
@click.argument('output_filepath', type=click.Path())
def main(size, variable_length, output_filepath):
    """ Randomly generates a sample of `size` binary strings than are
        cleaned data ready to be analyzed (saved in ../processed).
    """

    logger = logging.getLogger(__name__)
    start_time = time()
    logger.info('Starting Generation')
    end_time = time()

    generated_strings = list()
    possible_length_values = [50] if not(variable_length) else range(1, 50)

    np.random.seed(0)
    for _ in range(size):
        length = np.random.choice(possible_length_values)
        generated_strings.append(
            format(random.getrandbits(length), 'b')
        )

    with open(output_filepath, 'w') as file:
        writer = csv.writer(file)
        for row in generated_strings:
            if not(variable_length):
                writer.writerow([row, row.count('1') % 2])
            else:
                writer.writerow([row.rjust(50, "0"), row.count('1') % 2])

    execution_time = str(timedelta(seconds=end_time - start_time))
    logger.info(f'Ending Generation. Time taken: {execution_time}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
