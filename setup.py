from distutils.core import setup

setup(
    name='pixlc',
    version='1.0',
    packages=['pixlc',],
    scripts=['bin/pixLC-cat','bin/pixLC-socts','pixlc/pixLC.py'],
    long_description=open('README.md').read(),
    )
