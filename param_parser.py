#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse

def parameter_parser():
    parser = argparse.ArgumentParser(description="Run OpcodeExtraction.")
    
    parser.add_argument('--input-path','-i',
                        nargs='?',
                        default='./TestingBin/malware/00a2bd396600e892da75c686be60d5d13e5a34824afd76c02e8cf7257d2cf5c5',
                        help='Please input the binary file path.')
    parser.add_argument('--output-path','-o',
                        nargs='?',
                        default='Output')

    return parser.parse_args()

