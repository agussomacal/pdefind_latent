import os

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
plots_dir = project_dir + "/results/"
# plots_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + "/#plots/"
if not os.path.exists(plots_dir):
    os.mkdir(plots_dir)


def get_filename(filename, experiment, subfolders=[]):
    directory = plots_dir + experiment + '/'

    # check create directories
    for subfolder in subfolders:
        directory = directory + subfolder + '/'
        if not os.path.exists(directory):
            os.mkdir(directory)

    return directory + filename
