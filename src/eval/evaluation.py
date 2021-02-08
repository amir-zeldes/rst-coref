import os
import torch
from eval.metrics import Metrics
from models.tree import RstTree
from features.rst_dataset import *
from utils.document import Doc
from utils.constants import *
from ubc_coref.loader import Document
import nltk
from utils.other import action_map, relation_map


class Evaluator(object):
    def __init__(self, parser, data_helper, config, model_dir='../data/model'):
        print('Load parsing models ...')
        #clf.eval()
        self.parser = parser
        self.data_helper = data_helper
        self.config = config

    def parse(self, doc):
        """ Parse one document using the given parsing models"""
        pred_rst = self.parser.sr_parse(doc)
        return pred_rst

    @staticmethod
    def writebrackets(fname, brackets):
        """ Write the bracketing results into file"""
        with open(fname, 'w') as fout:
            print("Writing to ", fname)
            for item in brackets:
                fout.write(str(item) + '\n')
                
    def eval_parser(self, dev_data=None, path='./examples', save_preds=True, use_parseval=False):
        """ Test the parsing performance"""
        # Evaluation
        met = Metrics(levels=['span', 'nuclearity', 'relation'], use_parseval=use_parseval)
        # ----------------------------------------
        # Read all files from the given path
        if dev_data is None:
            dev_data = [os.path.join(path, fname) for fname in os.listdir(path) if fname.endswith('.dis')]
        total_cost = 0
        for eval_instance in dev_data:
            # ----------------------------------------
            fmerge = eval_instance.replace('.dis', '.merge')
            doc = Doc()
            doc.read_from_fmerge(fmerge)
            gold_rst = RstTree(eval_instance, fmerge)
            gold_rst.build()
            
            # tok_edus = [nltk.word_tokenize(edu) for edu in doc.doc_edus]
            tok_edus = [edu.split(" ") for edu in doc.doc_edus]
            tokens = flatten(tok_edus)

            coref_document = Document(raw_text=None, tokens=tokens, sents=tok_edus, 
                                      corefs=[], speakers=["0"] * len(tokens), genre="nw", 
                                      filename=None)

            coref_document.token_dict = doc.token_dict
            coref_document.edu_dict = doc.edu_dict
            doc = coref_document
                
            gold_action_seq, gold_rel_seq = gold_rst.decode_rst_tree()
            
            gold_action_seq = [action_map[x] for x in gold_action_seq]
            gold_relation_seq = [relation_map[x.lower()] for x in gold_rel_seq if x is not None]
            pred_rst, cost = self.parser.sr_parse(doc, 
                                                 torch.cuda.LongTensor(gold_action_seq),
                                                 torch.cuda.LongTensor(gold_relation_seq))
            total_cost += cost
            
            if save_preds:
                if not os.path.isdir('../data/predicted_trees'):
                    os.mkdir('../data/predicted_trees')

                filename = eval_instance.split(os.sep)[-1]
                filepath = f'../data/predicted_trees/{self.config[MODEL_NAME]}_{filename}'

                pred_brackets = pred_rst.bracketing()
                # Write brackets into file
                Evaluator.writebrackets(filepath, pred_brackets)
            # ----------------------------------------
            # Evaluate with gold RST tree
            met.eval(gold_rst, pred_rst)
            
        print("Total cost: ", total_cost)
        if use_parseval:
            print("Reporting original Parseval metric.")
        else:
            print("Reporting RST Parseval metric.")
        met.report()

def flatten(alist):
    """ Flatten a list of lists into one list """
    return [item for sublist in alist for item in sublist]
        
