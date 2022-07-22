import csv
import json
import os
import time
import pickle as pkl
import random
import re
import torch
from collections import defaultdict
import numpy as np

import parlai.core.build_data as build_data
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
import nltk


def _split_data(redial_path):
    # Copied from https://github.com/RaymondLi0/conversational-recommendations/blob/master/scripts/split-redial.py
    data = []
    for line in open(os.path.join(redial_path, "train_data.jsonl")):
        data.append(json.loads(line))
    random.shuffle(data)
    n_data = len(data)
    split_data = [data[: int(0.9 * n_data)], data[int(0.9 * n_data) :]]

    with open(os.path.join(redial_path, "train_data.jsonl"), "w") as outfile:
        for example in split_data[0]:
            json.dump(example, outfile)
            outfile.write("\n")
    with open(os.path.join(redial_path, "valid_data.jsonl"), "w") as outfile:
        for example in split_data[1]:
            json.dump(example, outfile)
            outfile.write("\n")


def _entity2movie(entity, abstract=""):
    # strip url
    x = entity[::-1].find("/")
    movie = entity[-x:-1]
    movie = movie.replace("_", " ")

    # extract year
    pattern = re.compile(r"\d{4}")
    match = re.findall(pattern, movie)
    year = match[0] if match else None
    # if not find in entity title, find in abstract
    if year is None:
        pattern = re.compile(r"\d{4}")
        match = re.findall(pattern, abstract)
        if match and 1900 < int(match[0]) < 2020:
            year = match[0]

    # recognize (20xx film) or (film) to help disambiguation
    pattern = re.compile(r"\(.*film.*\)")
    match = re.findall(pattern, movie)
    definitely_is_a_film = match != []

    # remove parentheses
    while True:
        pattern = re.compile(r"(.+)( \(.*\))")
        match = re.search(pattern, movie)
        if match:
            movie = match.group(1)
        else:
            break
    movie = movie.strip()

    return movie, year, definitely_is_a_film


DBPEDIA_ABSTRACT_PATH = "../../../dbpedia/short_abstracts_en.ttl"
DBPEDIA_PATH = "../../../dbpedia/mappingbased_objects_en.ttl"


def _build_dbpedia(dbpedia_path):
    movie2entity = {}
    movie2years = defaultdict(set)
    with open(dbpedia_path) as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            entity, line = line[: line.index(" ")], line[line.index(" ") + 1 :]
            _, line = line[: line.index(" ")], line[line.index(" ") + 1 :]
            abstract = line[:-4]
            movie, year, definitely_is_a_film = _entity2movie(entity, abstract)
            if (movie, year) not in movie2entity or definitely_is_a_film:
                movie2years[movie].add(year)
                movie2entity[(movie, year)] = entity
    return {"movie2years": movie2years, "movie2entity": movie2entity}


def _load_kg(path):
    kg = defaultdict(list)
    with open(path) as f:
        for line in f.readlines():
            tuples = line.split()
            if tuples and len(tuples) == 4 and tuples[-1] == ".":
                h, r, t = tuples[:3]
                # TODO: include property/publisher and subject/year, etc
                if "ontology" in r:
                    kg[h].append((r, t))
    return kg


def _extract_subkg(kg, seed_set, n_hop):
    subkg = defaultdict(list)
    subkg_hrt = set()

    ripple_set = []
    for h in range(n_hop):
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = seed_set
        else:
            tails_of_last_hop = ripple_set[-1][2]

        for entity in tails_of_last_hop:
            for tail_and_relation in kg[entity]:
                h, r, t = entity, tail_and_relation[0], tail_and_relation[1]
                if (h, r, t) not in subkg_hrt:
                    subkg[h].append((r, t))
                    subkg_hrt.add((h, r, t))
                memories_h.append(h)
                memories_r.append(r)
                memories_t.append(t)

        ripple_set.append((memories_h, memories_r, memories_t))

    return subkg


def load_text_embeddings(entity2entityId, dim, abstract_path):
    entities = []
    texts = []
    sent_tok = nltk.data.load('tokenizers/punkt/english.pickle')
    word_tok = nltk.tokenize.treebank.TreebankWordTokenizer()

    def nltk_tokenize(text):
        return [token for sent in sent_tok.tokenize(text)
                for token in word_tok.tokenize(sent)]

    with open(abstract_path, 'r') as f:
        for line in f.readlines():
            try:
                entity = line[:line.index('>')+1]
                if entity not in entity2entityId:
                    continue
                line = line[line.index('> "')+2:len(line)-line[::-1].index('@')-1]
                entities.append(entity)
                texts.append(line.replace('\\', ''))
            except Exception:
                pass
    vec_dim = dim
    try:
        model = Doc2Vec.load('doc2vec')
    except Exception:
        corpus = [nltk_tokenize(text) for text in texts]
        corpus = [
            TaggedDocument(words, ['d{}'.format(idx)])
            for idx, words in enumerate(corpus)
        ]
        model = Doc2Vec(corpus, vector_size=vec_dim, min_count=5, workers=28)
        model.save('doc2vec')

    full_text_embeddings = torch.zeros(len(entity2entityId), vec_dim)
    for i, entity in enumerate(entities):
        full_text_embeddings[entity2entityId[entity]] = torch.from_numpy(model.docvecs[i])

    return full_text_embeddings


def get_init_embedding(entity2entityId):
    init_embedding = torch.diag(torch.ones(len(entity2entityId)))
    return init_embedding


def build_vocab(data_path, embedding_path, embedding_dim, id2movie):
    word2vec_map = {}
    embedding_vocab = set()
    with open(embedding_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line_data = line.strip().split(' ')
            assert len(line_data) == embedding_dim + 1
            vec = [float(data) for data in line_data[1:]]
            word2vec_map[line_data[0]] = vec
            embedding_vocab.add(line_data[0])
    for idx in id2movie:
        movie_name = re.sub("[^\w\s@]", "", id2movie[idx]).strip().split(' ')
        movie_embeddings = []
        for word in movie_name:
            try:
                word_embed = word2vec_map[word]
            except KeyError:
                # continue
                word_embed = [0.0 for i in range(embedding_dim)]
            movie_embeddings.append(word_embed)
        movie_embeddings = np.array(movie_embeddings)
        movie_embedding = np.mean(movie_embeddings, axis=0).tolist()
        word2vec_map['@'+str(idx)] = movie_embedding
        embedding_vocab.add('@'+str(idx))
    print("Embedding Size: ", len(word2vec_map))
    instances = []
    with open(data_path) as json_file:
        for line in json_file.readlines():
            instances.append(json.loads(line))
    vocab = set()
    for instance in instances:
        messages = instance["messages"]
        for message in messages:
            text = message['text']
            # text = text.strip().split(' ')
            text = re.sub("[^\w\s@]", " ", text).lower().strip().split(' ')
            for word in text:
                vocab.add(word)
    print("Vocabulary size: ", len(vocab))
    intersec = embedding_vocab & vocab
    Complement = vocab - embedding_vocab
    print("Covered vocabulary: ", len(intersec), "Covered vocabulary rate: ", len(intersec)/len(vocab))
    word2idx = {}
    idx2word = {}
    word2idx['__unk__'] = 1
    idx2word[1] = '__unk__'
    word2idx['__pad__'] = 0
    idx2word[0] = '__pad__'
    for word in embedding_vocab:
        word2idx[word] = len(word2idx)
        idx2word[word2idx[word]] = word
    assert len(word2idx) == len(idx2word)
    embedding = [[0.0 for j in range(embedding_dim)] for i in range(len(word2idx))]
    for idx in range(2, len(word2idx)):
        embedding[idx] = word2vec_map[idx2word[idx]]
    return embedding_vocab, word2idx, embedding


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "crs")
    # define version if any
    version = None

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print("[building data: " + dpath + "]")

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        if not os.path.exists(dpath):
            build_data.make_dir(dpath)
        if os.path.exists(dpath + "/id2entity.pkl") and os.path.exists(dpath + "/dbpedia.pkl"):
            id2entity = pkl.load(open(dpath + "/id2entity.pkl", 'rb'))
            dbpedia = pkl.load(open(dpath + "/dbpedia.pkl", 'rb'))
        else:
            # download the data.
            fname = "redial_dataset.zip"
            url = "https://github.com/ReDialData/website/raw/data/" + fname  # dataset URL
            build_data.download(url, dpath, fname)

            # uncompress it
            build_data.untar(dpath, fname)

            _split_data(dpath)

            dbpedia = _build_dbpedia(DBPEDIA_ABSTRACT_PATH)
            movie2entity = dbpedia["movie2entity"]
            movie2years = dbpedia["movie2years"]

            # Match REDIAL movies to dbpedia entities
            movies_with_mentions_path = os.path.join(dpath, "movies_with_mentions.csv")
            with open(movies_with_mentions_path, "r") as f:
                reader = csv.reader(f)
                id2movie = {int(row[0]): row[1] for row in reader if row[0] != "movieId"}
            id2entity = {}
            for movie_id in id2movie:
                movie = id2movie[movie_id]
                pattern = re.compile(r"(.+)\((\d+)\)")
                match = re.search(pattern, movie)
                if match is not None:
                    name, year = match.group(1).strip(), match.group(2)
                else:
                    name, year = movie.strip(), None
                if year is not None:
                    if (name, year) in movie2entity:
                        id2entity[movie_id] = movie2entity[(name, year)]
                    else:
                        if len(movie2years) == 1:
                            id2entity[movie_id] = movie2entity[(name, movie2years[name][0])]
                        else:
                            id2entity[movie_id] = None

                else:
                    id2entity[movie_id] = (
                        movie2entity[(name, year)] if (name, year) in movie2entity else None
                    )
            # HACK: make sure movies are matched to different entities
            matched_entities = set()
            for movie_id in id2entity:
                if id2entity[movie_id] is not None:
                    if id2entity[movie_id] not in matched_entities:
                        matched_entities.add(id2entity[movie_id])
                    else:
                        id2entity[movie_id] = None
        if os.path.exists(dpath + "/vocab.pkl") and \
            os.path.exists(dpath + "/word2idx.pkl") and os.path.exists(dpath + "/embeddings.pkl"):
            vocab = pkl.load(open(dpath + "/vocab.pkl", 'rb'))
            word2idx = pkl.load(open(dpath + "/word2idx.pkl", 'rb'))
            embeddings = pkl.load(open(dpath + "/embeddings.pkl", 'rb'))
        else:
            movies_with_mentions_path = os.path.join(dpath, "movies_with_mentions.csv")
            with open(movies_with_mentions_path, "r") as f:
                reader = csv.reader(f)
                id2movie = {int(row[0]): row[1] for row in reader if row[0] != "movieId"}
            vocab, word2idx, embeddings = \
                build_vocab(dpath+'/train_data.jsonl', dpath + "/../../glove/glove.6B.100d.txt",
                            opt["vocab_dim"], id2movie)

        # Extract sub-kg related to movies
        if os.path.exists(dpath + "/kg.pkl") and \
                os.path.exists(dpath + "/entity2entityId.pkl") and os.path.exists(dpath + "/relation2relationId.pkl"):
            entity2entityId = pkl.load(open(dpath + "/entity2entityId.pkl", 'rb'))
            relation2relationId = pkl.load(open(dpath + "/relation2relationId.pkl", 'rb'))
            kg_idx = pkl.load(open(dpath + "/kg.pkl", 'rb'))
        else:
            kg = _load_kg(DBPEDIA_PATH)
            kg = _extract_subkg(
                kg,
                [
                    id2entity[k]
                    for k in id2entity
                    if id2entity[k] is not None and kg[id2entity[k]] != []
                ],
                opt["hop"],
            )
            for movie_id in id2entity:
                if id2entity[movie_id] is not None:
                    kg[id2entity[movie_id]].append(('self_loop', id2entity[movie_id]))
                else:
                    kg[movie_id].append(('self_loop', movie_id))
            entities = set([k for k in kg]) | set([x[1] for k in kg for x in kg[k]])
            entity2entityId = dict([(k, i) for i, k in enumerate(entities)])
            relations = set([x[0] for k in kg for x in kg[k]])
            relation2relationId = dict([(k, i) for i, k in enumerate(relations)])
            kg_idx = defaultdict(list)
            for h in kg:
                for r, t in kg[h]:
                    kg_idx[entity2entityId[h]].append((relation2relationId[r], entity2entityId[t]))
        if os.path.exists(dpath + "./movie_ids.pkl"):
            movie_ids = pkl.load(open(dpath + "./movie_ids.pkl", 'rb'))
        else:
            movie_ids = []
            for k in id2entity:
                movie_ids.append(entity2entityId[id2entity[k]] if id2entity[k] is not None else entity2entityId[k])

        opt["n_entity"] = len(entity2entityId)
        # start_time = time.time()
        # init_entity_embedding = load_text_embeddings(entity2entityId, 128, DBPEDIA_ABSTRACT_PATH)
        # print("Get doc2vec time: ", time.time() - start_time)
        # init_entity_embedding = get_init_embedding(entity2entityId)
        pkl.dump(id2entity, open(os.path.join(dpath, "id2entity.pkl"), "wb"))
        pkl.dump(dbpedia, open(os.path.join(dpath, "dbpedia.pkl"), "wb"))
        pkl.dump(kg_idx, open(os.path.join(dpath, "kg.pkl"), "wb"))
        pkl.dump(entity2entityId, open(os.path.join(dpath, "entity2entityId.pkl"), "wb"))
        pkl.dump(relation2relationId, open(os.path.join(dpath, "relation2relationId.pkl"), "wb"))
        pkl.dump(movie_ids, open(os.path.join(dpath, "movie_ids.pkl"), "wb"))
        pkl.dump(vocab, open(os.path.join(dpath, "vocab.pkl"), "wb"))
        pkl.dump(word2idx, open(os.path.join(dpath, "word2idx.pkl"), "wb"))
        pkl.dump(embeddings, open(os.path.join(dpath, "embeddings.pkl"), "wb"))
        # pkl.dump(init_entity_embedding, open(os.path.join(dpath, "init_entity_embedding.pkl"), "wb"))

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
