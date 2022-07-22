import copy
import re
import csv
import json
import os
import pickle as pkl

import requests

import parlai.core.agents as core_agents
from parlai.core.teachers import DialogTeacher
from parlai.core.dict import DictionaryAgent

from .build import build


def _path(opt):
    # ensure data is built
    build(opt)

    # set up paths to data (specific to each dataset)
    dt = opt["datatype"].split(":")[0]
    return (
        os.path.join(opt["datapath"], "crs", f"{dt}_data.jsonl"),
        os.path.join(opt["datapath"], "crs", "movies_with_mentions.csv"),
        os.path.join(opt["datapath"], "crs", "id2entity.pkl"),
        os.path.join(opt["datapath"], "crs", "entity_dict.pkl"),
        os.path.join(opt["datapath"], "crs", "text_dict_tmdb4.pkl"),
        os.path.join(opt["datapath"], "crs", "entity2entity_id4.pkl"),
        # os.path.join(opt["datapath"], "crs", "entity2entityId.pkl"),
        os.path.join(opt["datapath"], "crs", "relation2relationId.pkl"),
    )


def _id2dbpedia(movie_id):
    pass


def _text2entities(text, text_dict):
    return list(set(text_dict[text]))


class RedialTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        # store datatype
        self.datatype = opt["datatype"].split(":")[0]
        self.match = opt.get("match", False)
        if self.datatype != 'train':
            a = 1

        # store identifier for the teacher in the dialog
        self.id = "crs"
        self.dt = opt["datatype"].split(":")[0]

        # store paths to images and labels
        opt[
            "datafile"
        ], movies_with_mentions_path, id2entity_path, entity_dict_path, text_dict_path, entity2entityId_path, relation2relationId_path = _path(
            opt
        )

        if not shared:
            self.entity2entityId = pkl.load(open(entity2entityId_path, "rb"))
            self.id2entity = {}
            for e in self.entity2entityId:
                self.id2entity[self.entity2entityId[e]] = e
            self.text_dict = pkl.load(open(text_dict_path, "rb"))
        else:
            self.entity2entityId = shared["entity2entityId"]
            self.id2entity = shared["id2entity"]
            self.text_dict = shared["text_dict"]

        super().__init__(opt, shared)

    def share(self):
        """Share internal states."""
        shared = super().share()
        shared["entity2entityId"] = self.entity2entityId
        shared["id2entity"] = self.id2entity
        shared["text_dict"] = self.text_dict
        return shared

    def _convert_ids_to_indices(self, text, questions):
        """@movieID -> @movieIdx"""
        pattern = re.compile("@\d+")
        movieId_list = []

        def convert(match):
            movieId = match.group(0)
            try:
                entity = self.id2entity[int(movieId[1:])]
                if entity is not None:
                    movieId_list.append(str(self.entity2entityId[entity]))
                else:
                    movieId_list.append(str(self.entity2entityId[int(movieId[1:])]))
                return DictionaryAgent.default_unk
            except Exception:
                return ""

        return re.sub(pattern, convert, text), movieId_list

    def movie2idx(self, text):
        patten = re.compile("@\d+")
        movie_entity_idx = []
        movie_id_list = re.findall(patten, text)
        for movieId in movie_id_list:
            movie_entity_idx.append(str(self.entity2entityId[movieId[1:]]))
        return text, movie_entity_idx

    def movie2idx_replace(self, text):
        pattern = re.compile("@\d+")
        movie_entity_idx = []

        def convert(match):
            movie_id = match.group(0)
            try:
                movie_entity_idx.append(str(self.entity2entityId[movie_id[1:]]))
                # return '__unk__'
                return str(self.entity2entityId[movie_id[1:]])
            except KeyError:
                return str(movie_id[1:])

        return re.sub(pattern, convert, text), movie_entity_idx

    def movie2idx_replace_(self, text, previous_mentioned=None, source=True):
        pattern = re.compile("@\d+")
        movie_entity_idx = []

        def convert(match):
            movie_id = match.group(0)
            try:
                movie_entity_idx.append(str(self.entity2entityId[movie_id[1:]]))
                if source:
                    return str(self.entity2entityId[movie_id[1:]])
                elif not previous_mentioned or str(self.entity2entityId[movie_id[1:]]) not in previous_mentioned:
                    return '__unk__'
                else:
                    return str(self.entity2entityId[movie_id[1:]])
            except KeyError:
                return str(movie_id[1:])

        return re.sub(pattern, convert, text), movie_entity_idx

    def _get_entities(self, text):
        """text -> [#entity1, #entity2]"""
        entities = _text2entities(text, self.text_dict)
        # entities = [str(self.entity2entityId[x]) for x in entities if x in self.entity2entityId]
        entities = [str(x) for x in entities]
        return entities

    def _get_entities_tmdb(self, text):
        entities = self.text_dict[text]
        return entities

    def setup_data(self, path):
        self.instances = []
        with open(path) as json_file:
            for line in json_file.readlines():
                self.instances.append(json.loads(line))

        # define iterator over all queries
        for instance in self.instances:
            initiator_id = instance["initiatorWorkerId"]
            respondent_id = instance["respondentWorkerId"]
            initiator_ques = instance["initiatorQuestions"]
            liked_movie = []
            disliked_movie = []
            for movie_id in initiator_ques:
                if initiator_ques[movie_id]["liked"] == 1:
                    liked_movie.append(self.entity2entityId[movie_id])
                elif initiator_ques[movie_id]["liked"] == 0:
                    disliked_movie.append(self.entity2entityId[movie_id])
            messages = instance["messages"]
            message_idx = 0
            new_episode = True

            previously_mentioned_movies_list = []
            previously_utterance = []
            mentioned_entities = []
            turn = 0
            while message_idx < len(messages):
                source_text = []
                target_text = []
                while (
                    message_idx < len(messages)
                    and messages[message_idx]["senderWorkerId"] == initiator_id
                ):
                    source_text.append(messages[message_idx]["text"])
                    message_idx += 1
                while (
                    message_idx < len(messages)
                    and messages[message_idx]["senderWorkerId"] == respondent_id
                ):
                    target_text.append(messages[message_idx]["text"])
                    message_idx += 1
                source_text = [text for text in source_text if text != ""]
                target_text = [text for text in target_text if text != ""]
                if source_text != [] or target_text != []:
                    for i, src in enumerate(source_text):
                        mentioned_entities += self._get_entities(src)
                        source_entity = self.text_dict[src]
                        source_text[i] = source_text[i].lower()
                        if source_entity and self.match:
                            for e_id in source_entity:
                                source_text[i] = \
                                    re.sub("\\b" + self.id2entity[e_id].lower() + "\\b", str(e_id), source_text[i].lower())
                                if self.id2entity[e_id] == "comedy":
                                    source_text[i] = re.sub("\\bcomedies\\b", str(e_id), source_text[i].lower())
                                if self. id2entity[e_id] == "William Shakespeare":
                                    source_text[i] = re.sub("\\bshakespeare\\b", str(e_id), source_text[i].lower())
                    target_mentioned_entities = []
                    for i, tgt in enumerate(target_text):
                        target_mentioned_entities += self._get_entities(tgt)
                        target_entity = self.text_dict[tgt]
                        target_text[i] = target_text[i].lower()
                        if target_entity and self.match:
                            for e_id in target_entity:
                                target_text[i] = \
                                    re.sub("\\b" + self.id2entity[e_id].lower() + "\\b", str(e_id),
                                           target_text[i].lower())
                                if self.id2entity[e_id] == "comedy":
                                    target_text[i] = re.sub("\\bcomedies\\b", str(e_id), target_text[i].lower())
                                if self.id2entity[e_id] == "William Shakespeare":
                                    target_text[i] = re.sub("\\bshakespeare\\b", str(e_id), target_text[i].lower())
                    source_text = '\n'.join(source_text)
                    target_text = '\n'.join(target_text)
                    if self.match:
                        source_text, source_movie_list = self.movie2idx_replace(source_text)
                        target_text, target_movie_list = self.movie2idx_replace(target_text)
                    else:
                        source_text, source_movie_list = self.movie2idx(source_text)
                        target_text, target_movie_list = self.movie2idx(target_text)
                    turn += 1
                    previously_utterance.append(source_text)
                    # previously_utterance.append(target_text)
                    # remove movies mentioned before from target movie list
                    # remove category 4
                    target_movie_list = list(set(target_movie_list))
                    for target_movie in target_movie_list:
                        if target_movie in source_movie_list + previously_mentioned_movies_list:
                            target_movie_list.remove(target_movie)
                    # remove disliked movies or unknown movies from target movie list
                    # remove category 2, 3
                    for target_movie in target_movie_list:
                        if int(target_movie) not in liked_movie:
                            target_movie_list.remove(target_movie)
                    if len(source_text) == 0 and len(previously_utterance) != 0:
                        continue
                    if len(target_text) == 0:
                        continue
                    yield (source_text, [target_text], None,
                           [str(turn), ' '.join(previously_mentioned_movies_list + source_movie_list),
                            ' '.join(target_movie_list), ' '.join(mentioned_entities), target_text,
                            ' '.join(source_movie_list)],
                           None), new_episode
                    new_episode = False
                    previously_mentioned_movies_list += source_movie_list + target_movie_list
                    mentioned_entities += target_mentioned_entities
                    previously_utterance.append(target_text)


class DefaultTeacher(RedialTeacher):
    pass
