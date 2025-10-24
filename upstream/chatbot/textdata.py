# Copyright 2015 Conchylicultor. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Loads the dialogue corpus, builds the vocabulary
"""

import numpy as np
import nltk  # For tokenize
from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string
import collections

from chatbot.corpus.cornelldata import CornellData
from chatbot.corpus.opensubsdata import OpensubsData
from chatbot.corpus.scotusdata import ScotusData
from chatbot.corpus.ubuntudata import UbuntuData
from chatbot.corpus.lightweightdata import LightweightData


class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.encoderSeqs = []
        self.decoderSeqs = []
        self.targetSeqs = []
        self.weights = []


class TextData:
    """Dataset class
    Warning: No vocabulary limit
    """

    availableCorpus = collections.OrderedDict([  # OrderedDict because the first element is the default choice
        ('cornell', CornellData),
        ('opensubs', OpensubsData),
        ('scotus', ScotusData),
        ('ubuntu', UbuntuData),
        ('lightweight', LightweightData),
    ])

    @staticmethod
    def corpusChoices():
        """Return the dataset availables
        Return:
            list<string>: the supported corpus
        """
        return list(TextData.availableCorpus.keys())

    def __init__(self, args):
        """Load all conversations
        Args:
            args: parameters of the model
        """
        # Model parameters
        self.args = args

        # Path variables
        self.corpusDir = os.path.join(self.args.rootDir, 'data', self.args.corpus)
        basePath = self._constructBasePath()
        self.fullSamplesPath = basePath + '.pkl'  # Full sentences length/vocab
        self.filteredSamplesPath = basePath + '-length{}-filter{}-vocabSize{}.pkl'.format(
