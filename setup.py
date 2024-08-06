from setuptools import setup, find_packages

setup(
    name='test_mlops_project',
    version='0.1.0',
    description='A sample machine learning project for ci/cd',
    author='Prakash Prasad',
    author_email='your.email@example.com',
    url='https://github.com/prakash-prasad/test-repo-v2',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.9',
)
