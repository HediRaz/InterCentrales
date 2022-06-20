"""Setup of gan-face-editing."""

from setuptools import find_packages, setup

# Installation
config = {
        'name': 'gan-face-editing-2', 'version': '1.0.1',
        'description': ('Inter Centrales Ceteris Paribus face challenge, '
                        '2nd team.'),
        'author': 'Hédi Razgallah', 'packages': find_packages(),
        'zip_safe': True
        }

setup(**config)
