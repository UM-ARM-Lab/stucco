from setuptools import setup, find_packages

setup(
    name='stucco',
    version='1.1.0',
    packages=find_packages(),
    url='',
    license='',
    author='zhsh',
    author_email='zhsh@umich.edu',
    description='',
    test_suite='pytest',
    tests_require=[
        'pytest', 'pybullet', 'pynput', 'trimesh', 'pymeshlab', 'mesh-to-sdf'
        'window_recorder @ git+ssh://git@github.com/LemonPi/window_recorder.git'
    ], install_requires=[
        'matplotlib', 'numpy', 'scipy', 'scikit-learn', 'torch', 'gpytorch',
    ]
)
