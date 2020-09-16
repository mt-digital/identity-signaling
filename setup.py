from setuptools import setup, find_packages

setup(
    name='id_signaling',
    version='0.1',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Click',
        'matplotlib',
        'numpy',
        'pandas',
        'scipy'
    ],
    entry_points={
        'console_scripts': [
            'runexp=id_signaling.scripts.runner:run',
            'subexp=id_signaling.scripts.runner:sub',
            'run_analysis=id_signaling.scripts.runner:run_analysis'
        ]
    }
)
