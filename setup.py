from setuptools import setup, find_packages

def get_requirements(file_path):
    '''
    Reads a requirements file and returns a list of dependencies.
    '''
    requirements = []
    with open(file_path, 'r') as file:
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='mlproject',
    version='0.1.0',
    author='Reza Kalhor',
    author_email='r3za.kalhor@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)