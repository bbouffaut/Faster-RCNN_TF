from setuptools import setup
from faster_rcnn_tf_module import faster_rcnn_tf

def read_readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='faster_rcnn_tf',
    version=faster_rcnn_tf.__version__,
    license='MIT',
    description='Module to load Faster RCNN TF capabilities',
    long_description=read_readme(),
    url='https://github.com/bbouffaut/Faster-RCNN_TF',
    author='Fork from bigsnarfdude',
    author_email='baptiste.bouffaut@gmail.com',
    keywords='tensorflow Convutional Neural-Network ',
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: System :: Monitoring',
    ],
    #packages=['gpustat'],
    py_modules=['faster_rcnn_tf'],
    install_requires=[
    ],
    test_suite='',
    tests_require=[],
    entry_points={
        'console_scripts': ['faster_rcnn_tf=faster_rcnn_tf:main'],
    },
    include_package_data=True,
    zip_safe=False,
)