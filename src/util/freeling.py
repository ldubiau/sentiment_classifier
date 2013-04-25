import os
import tempfile
import shlex, subprocess
from classifiers import logger

FREELING_DOC_SPLIT_PATTERN = '| | Fz 1'

class FreelingWord(object):
    def __init__(self, word, lemma, tag, prob):
        self.word = word
        self.lemma = lemma
        self.tag = tag
        self.prob = prob


class FreelingProcessor(object):
    def __init__(self, cmd=None, config_file=None):
        if cmd:
            self.cmd = cmd
        else:
            self.cmd = 'analyze'
        if config_file:
            self.config_file = config_file
        else:
            self.config_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'externaltools', 'myconfig.cfg')

    def process_text(self, text):
        out = self._execute_freeling(text)
        docs = out.split(FREELING_DOC_SPLIT_PATTERN)
        freeling_docs = []
        for doc in docs:
            lines = doc.split('\n')
            freeling_words = []
            for line in lines:
                line_data = line.split(' ')
                if len(line_data) == 4:
                    word = line_data[0].decode('utf-8');
                    lemma = line_data[1].decode('utf-8');
                    tag = line_data[2]
                    prob = float(line_data[3])

                    freeling_words.append(FreelingWord(word, lemma, tag, prob))
            freeling_docs.append(freeling_words)
        return freeling_docs

    def _execute_freeling(self, document):
        args = [self.cmd, '-f', self.config_file]

        print args
        p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout, stderr = p.communicate(input=document.encode('utf-8'))
        return stdout

    def _read_freeling_output(self):
        pass
