from setuptools import setup, find_packages

setup(
    name='points2regions',
    version='0.1.0',
    packages=find_packages(exclude=["tests*", "example*", "tissuumaps"]),
    description='A simple and efficient clustering tool for spatial 2D points with categorical labels.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Axel Andersson',
    author_email='axel.andersson@it.uu.se',
    url='https://github.com/wahlby-lab/Points2Regions',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn',
        'scikit-image'
    ],
    classifiers=[
        # Classifiers help users find your project,
        # see: https://pypi.org/classifiers/
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
