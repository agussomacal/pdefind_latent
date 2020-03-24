import os

project_dir = os.path.abspath(__file__).split('ret-ode')[0]+'ret-ode/'
plots_dir = project_dir + "results/"
# plots_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/#plots/"


def get_filename(filename, experiment, subfolders=[]):
    directory = plots_dir + experiment + '/'

    # check create directories
    for subfolder in subfolders:
        directory = directory + subfolder + '/'
        if not os.path.exists(directory):
            os.mkdir(directory)

    return directory + filename
