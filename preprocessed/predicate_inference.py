# implementation of an information extraction system based on predicate calculus
import re
import sys
import os
import json
import numpy as np

premises = []


class inference:
    def __init__(self, data):
        self.data = data
        self.fact = dict()
        self.rules = dict(dict())
        self.statements = dict()
        self.result = str()
        self.fact_extraction()
        self.rules_extraction()
        self.predicate_inference()
        self.split_string()
        self.in_database()


    def fact_extraction(self):
        '''
        Extract facts from data
        UNFINISHED
        :return: facts
        '''
        return self.fact

    def rules_extraction(self):
        '''
        Extract rules from data
        UNFINISHED
        :return:
        '''
        return self.rules

    # core algorithm: reasoning
    def predicate_inference(self):
        '''
        Inference
        UNFINISHED
        :return:
        '''
        res = ""
        change = True

        while change:
            change = False

        return res

