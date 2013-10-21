# -*- coding: utf-8 -*-

__author__ = 'Luciana Dubiau'

if __name__ == "__main__":
    from distutils.core import setup

    setup(
        name='sentiment_analyzer',
        version='0.1',
        author='Luciana Dubiau',
        author_email='lu.dubiau@gmail.com',        
        install_requires=[
            'httplib2==0.7.6',
            'beautifulsoup4==4.1.3',
            'nltk==2.0.3',
            'numpy==1.6.2',
            'PrettyTable==0.6.1',
            'scipy==0.10.0',
            'scikit-learn==0.12.1',
            'svmlight==0.4'
        ]
    )
