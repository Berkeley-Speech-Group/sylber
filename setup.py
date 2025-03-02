from setuptools import setup, find_packages

setup(
    name='sylber',  
    version='0.1.0',
    author='Cheol Jun Cho, Nicholas Lee, Akshat Gupta, Dhruv Agarwal, Ethan Chen, Alan Black, Gopala Anumanchipalli',
    author_email='cheoljun@berkeley.edu',
    description='Python code for "Sylber: Syllabic Embedding Representation of Speech from Raw Audio"',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Berkeley-Speech-Group/sylber',  
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',  
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'soundfile',
        'librosa',
        'torch',
        'torchaudio',
        'transformers',
        'huggingface-hub',
        'tqdm',
        'speech-articulatory-coding',
        'typeguard>=4.0.1',
        'torchode==1.0.0',
        'torchdiffeq==0.2.4',
        'beartype==0.19.0',
        'gateloop-transformer==0.2.5',
        'vector-quantize-pytorch==1.18.5',
    ],
    include_package_data=False,  
    license='Apache-2.0',  
)
