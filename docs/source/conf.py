import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

project = 'JMU Smart Supermarket'
copyright = '2026, Marius Dausacker, Verena Fanous'
author = 'Marius Dausacker, Verena Fanous'
release = '1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme'
]

templates_path = ['_templates']
exclude_patterns = []
language = 'de'

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']