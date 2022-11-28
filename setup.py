from setuptools import setup, find_packages

setup_args = dict(
    name = 'pymls',
    version = '0.0.0',
	description = 'Python implementation of the dislocation contrast factor calculation described by Martinez-Garcia, Leoni, and Scardi.',
    author = 'Peter Metz',
    author_email = 'pmetz1@utk.edu',
    package_dir = {'': 'src'},
    packages = ['pymls']
)

if __name__ == '__main__':
    print(find_packages())
    setup(**setup_args)

