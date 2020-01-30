import os
import json
import argparse
import subprocess
from pathlib import Path
from time import gmtime, strftime
from ISR.utils.logger import get_logger
from multiprocessing import Process

parser = argparse.ArgumentParser(description='')
parser.add_argument('-s', '--source_directory', type=str, default='./pdfs', help="super resolution upscale factor")
parser.add_argument('-d', '--destination_directory', type=str, default=None, help='testing batch size')
parser.add_argument('-r', '--resolution', type=int, default=300, help='dpi image')
parser.add_argument('-t', '--threads', type=int, default=1, help='number of threads for Samples synthetic to use')
parser.add_argument('-l', '--logging', type=str, default=None, help='directory for log-file')
parser.add_argument('-n', '--name', type=str, default='dataset', help='name dataset')

opt = parser.parse_args()
logger = get_logger(opt.name, opt.source_directory) if opt.logging is None else get_logger(opt.file, opt.logging)
time = strftime("%Y-%m-%d_%H_%M_%S", gmtime())
config_name = r'congif_{}.json'
default_name_dir = 'data_{}'
cmd = lambda pdf, source, destination: ['pdftoppm',
                                        str(Path(source + '/' + pdf)),
                                        '-r', str(opt.resolution),
                                        str(Path(destination + '/' + pdf)),
                                        '-jpeg']


def info():
    return 'Parent process: {}, process id:{}'.format(os.getppid(), os.getpid())


def worker(pdfs, source, destination, logger):
    logger.info(info())
    for pdf in pdfs:
        process = subprocess.Popen(cmd(pdf, source, destination))
        std_out, std_error = process.communicate()

        if std_out is not None or std_out is not None:
            logger.debug(std_out)
            logger.error(std_error)


def run():
    # Check exist source path with pdfs
    if opt.source_directory is None or not Path(opt.source_directory).exists():
        logger.error('Not source path.')
        return 1
    else:
        logger.info('Path exist: ' + str(Path(opt.source_directory)))

    # Create destination directory
    if opt.destination_directory is None:
        opt.destination_directory = default_name_dir.format(time)

    # Check exists path or make it
    if not Path(opt.destination_directory).exists():
        Path(opt.destination_directory).mkdir(parents=True, exist_ok=True)
        logger.info('Make directory: ' + str(Path(opt.destination_directory)))

    # Synthetic
    if opt.threads < 1:
        opt.threads = 1

    list_pdfs = [str(pdf.name) for pdf in Path(opt.source_directory).glob('*.pdf')]

    proccesses = []
    if len(list_pdfs) / opt.threads < 1:
        step = 1
    else:
        step = len(list_pdfs) // opt.threads

    for i in range(0, len(list_pdfs), step):
        if i + step > len(list_pdfs):
            end = len(list_pdfs)
        else:
            end = i + step

        # Run process
        p = Process(target=worker, args=(list_pdfs[i:end], opt.source_directory, opt.destination_directory, logger, ))
        proccesses.append(p)
        p.start()

    for p in proccesses:
        p.join()

    # Save params arguments
    with open(str(Path(opt.destination_directory + '/' + config_name.format(time))), 'w') as f:
        config = opt.__dict__
        config['size_list_pdfs'] = len(list_pdfs)
        json.dump(opt.__dict__, f, indent=2)


if __name__ == '__main__':
    run()
