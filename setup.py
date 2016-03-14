from distutils.core import setup

setup(
    name='pixlc',
    version='1.0',
    packages=['pixlc',],
    scripts=['bin/pixLC-cat','pixlc/pixLC.py'],
    long_description=open('README.md').read(),
    )
