# based on https://www.youtube.com/watch?v=8u57WSXVpmw
from collections import Counter
import json
import pickle
import random

from numpy.core.arrayprint import _void_scalar_repr
import pandas as pd
import spacy
from spacy.kb import KnowledgeBase
from spacy.training import Example
from spacy.util import minibatch, compounding


def load_entities(file_path):
    df = pd.read_csv(file_path)
    names = dict(zip(df['QID'], df['Name']))
    descriptions = dict(zip(df['QID'], df['Description']))
    return names, descriptions


def create_kb(kb_dir='sample_kb', nlp_dir='sample_nlp'):
    nlp = spacy.load('en_core_web_lg')
    text = 'Tennis champion Emerson was expected to win Wimbledon.'
    doc = nlp(text)
    file_path = 'entities.csv'
    name_dict, desc_dict = load_entities(file_path)

    sample_qid, sample_desc = list(desc_dict.items())[0]
    sample_doc = nlp(sample_desc)
    entity_vector_length = len(sample_doc.vector)  # should be 300
    kb = KnowledgeBase(vocab=nlp.vocab,
                       entity_vector_length=entity_vector_length)

    for qid, desc in desc_dict.items():
        desc_doc = nlp(desc)
        desc_enc = desc_doc.vector
        # NB: entity_vector could be any encoding
        # freq is the count of times the word appears in the corpus
        # not used in this tutorial
        kb.add_entity(entity=qid, entity_vector=desc_enc, freq=42)

    # add provided alias
    for qid, name in name_dict.items():
        # probabilities is P(entity|alias) = 1.0
        # we assume that each alias only maps to one entity
        kb.add_alias(alias=name, entities=[qid], probabilities=[1])

    # add additional alias with equal probability
    # this could be learned from data
    qids = name_dict.keys()
    probs = [0.3 for qid in qids]
    kb.add_alias(alias='Emerson', entities=qids, probabilities=probs)

    print(f'Entities in the KB: {kb.get_entity_strings()}')
    print(f'Aliases in the KB: {kb.get_alias_strings()}')
    print()
    # questions here are:
    # 1) what matching function is being used? - is this deterministic?
    # 2) what threshold is being used to determine how many candidates are presented?
    entities = [
        c.entity_ for c in kb.get_alias_candidates('Roy Stanley Emerson')
    ]
    print(f'Candidates for \'Roy Stanley Emerson\': {entities}')
    entities = [c.entity_ for c in kb.get_alias_candidates('Emerson')]
    print(f'Candidates for \'Emerson\': {entities}')
    entities = [c.entity_ for c in kb.get_alias_candidates('Todd')]
    print(f'Candidates for \'Todd\': {entities}')

    kb.to_disk(kb_dir)
    nlp.to_disk(nlp_dir)


def train_el(kb_dir='sample_kb',
             nlp_dir='sample_nlp',
             data_path='emerson_annotated_text.jsonl',
             out_dir='nlp_el'):
    nlp = spacy.load(nlp_dir)

    # needed when creating entity_linker
    def create_kb(vocab):
        kb = KnowledgeBase(vocab, entity_vector_length=1)
        kb.from_disk(kb_dir)
        return kb

    kb = create_kb(nlp.vocab)
    dataset = []
    with open(data_path, "r", encoding="utf8") as jsonfile:
        for line in jsonfile:
            example = json.loads(line)
            text = example["text"]
            if example["answer"] == "accept":
                QID = example["accept"][0]
                offset = (example["spans"][0]["start"],
                          example["spans"][0]["end"])
                links_dict = {QID: 1.0}
            dataset.append((text, {"links": {offset: links_dict}}))

    # inspect dataset
    print(dataset[0])
    gold_ids = []
    for text, annot in dataset:
        for span, links_dict in annot["links"].items():
            for link, value in links_dict.items():
                if value:
                    gold_ids.append(link)
    print(Counter(gold_ids))

    # grab 80% of labeled data from each entity_id for training set
    train_dataset = []
    test_dataset = []
    for QID in kb.get_entity_strings():
        indices = [i for i, j in enumerate(gold_ids) if j == QID]
        train_dataset.extend(dataset[index]
                             for index in indices[0:8])  # first 8 in training
        test_dataset.extend(dataset[index]
                            for index in indices[8:10])  # last 2 in test

    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    # pass the documents through the NER tagger
    TRAIN_DOCS = []
    for text, annotation in train_dataset:
        doc = nlp(text)
        TRAIN_DOCS.append((doc, annotation))

    nlp.add_pipe('entity_linker', last=True, config={'incl_prior': False})
    nlp.get_pipe('entity_linker').set_kb(create_kb)

    # disable other components
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'entity_linker']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.create_optimizer()
        for itn in range(500):
            random.shuffle(TRAIN_DOCS)
            batches = minibatch(TRAIN_DOCS, size=compounding(4.0, 32.0, 1.001))
            losses = {}
            for batch in batches:
                # this is no longer working in spacy 3.0
                texts, annotations = zip(*batch)
                # breakpoint()
                # texts are Docs and annotations are dicts of the form {'links': {(65, 72): {'Q312545': 1.0}}}
                # expecting batch of Examples, unclear what format that ought to be in
                train_batch = []
                for i in range(len(texts)):
                    train_batch.append(
                        Example.from_dict(texts[i], annotations[i]))
                nlp.update(train_batch, drop=0.2)
            if itn % 50 == 0:
                print(itn, 'losses', losses)
    print(itn, 'losses', losses)

    nlp.to_disk(out_dir)

    with open('test_set_pkl', 'wb') as f:
        pickle.dump(test_dataset)


if __name__ == '__main__':
    kb_dir = 'sample_kb'
    nlp_dir = 'sample_nlp'
    data_path = 'emerson_annotated_text.jsonl'
    out_dir = 'nlp_el'
    # create_kb(kb_dir='sample_kb', nlp_dir='sample_nlp')
    train_el(kb_dir='sample_kb',
             nlp_dir='sample_nlp',
             data_path='emerson_annotated_text.jsonl',
             out_dir='nlp_el')
