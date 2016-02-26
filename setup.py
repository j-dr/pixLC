from distutils.core import setup

setup(
    name='pixlc',
    version='1.0',
    packages=['pixlc',],
    scripts=['pixlc/pixLC.py'],
    long_description=open('README.md').read(),
)
