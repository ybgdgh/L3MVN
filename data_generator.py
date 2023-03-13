from cProfile import label
import os
from tqdm import tqdm
import glob

import numpy as np
from sympy.utilities.iterables import multiset_permutations
import pickle

import habitat
from habitat.sims import make_sim
from habitat_sim import Simulator
import habitat_sim

import torch
import torch.nn.functional as F
from transformers import (
    BertModel,
    BertTokenizer,
    RobertaModel,
    RobertaTokenizer,
    GPT2Model,
    GPT2Tokenizer,
    GPTNeoModel,
    AutoTokenizer,
    AutoModelForCausalLM,
    GPTJModel,
)

import sys 
sys.path.append("..") 

from constants import mp3d_category_id, category_to_id, hm3d_category

fileName = 'data/matterport_category_mappings.tsv'

text = ''
lines = []
items = []
hm3d_semantic_mapping={}

with open(fileName, 'r') as f:
    text = f.read()
lines = text.split('\n')

for l in lines:
    items.append(l.split('    '))

for i in items:
    if len(i) > 3:
        hm3d_semantic_mapping[i[2]] = i[-1]


class DataGenerator:

    def __init__(
        self,
        default_lm=None,
        device=None,
        verbose=False,
        label_set="mpcat40",
        use_gt_cooccurrencies=True,
    ):

        self.verbose = verbose
        self.device = (
            torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if device is None else device)

        self.lm = None
        self.lm_model = None
        self.tokenizer = None
        self.embedder = None

        if default_lm is not None:
            self.configure_lm(default_lm)

        self.max_num_obj = None
        self.objects = None
        self.labels = None
        # self.object_counts = None

    def configure_lm(self, lm):
        """
        Configure the language model, tokenizer, and embedding generator function.

        Sets self.lm, self.lm_model, self.tokenizer, and self.embedder based on the
        selected language model inputted to this function.

        Args:
            lm: str representing name of LM to use

        Returns:
            None
        """
        if self.lm is not None and self.lm == lm:
            print("LM already set to", lm)
            return

        self.lm = lm

        if self.verbose:
            print("Setting up LM:", self.lm)

        if lm == "BERT":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            lm_model = BertModel.from_pretrained("bert-base-uncased")
            start = "[CLS]"
            end = "[SEP]"
        elif lm == "BERT-large":
            tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
            lm_model = BertModel.from_pretrained("bert-large-uncased")
            start = "[CLS]"
            end = "[SEP]"
        elif lm == "RoBERTa":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            lm_model = RobertaModel.from_pretrained("roberta-base")
            start = "<s>"
            end = "</s>"
        elif lm == "RoBERTa-large":
            tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
            lm_model = RobertaModel.from_pretrained("roberta-large")
            start = "<s>"
            end = "</s>"
        elif lm == "GPT2-large":
            lm_model = GPT2Model.from_pretrained("gpt2-large")
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2-large")
        elif lm == "GPT-Neo":
            lm_model = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B")
            tokenizer = GPT2Tokenizer.from_pretrained(
                "EleutherAI/gpt-neo-1.3B")
        elif lm == "GPT-J":
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
            lm_model = GPTJModel.from_pretrained(
                "EleutherAI/gpt-j-6B",
                revision="float16",
                torch_dtype=torch.float16,  # low_cpu_mem_usage=True
            )
        else:
            print("Model option " + lm + " not implemented yet")
            raise

        self.lm_model = lm_model
        self.lm_model.eval()
        self.lm_model = self.lm_model.to(self.device)

        self.tokenizer = tokenizer

        if self.verbose:
            print("Loaded LM:", self.lm)
        # self.tokenizer = self.tokenizer.to(self.device)

        if lm in ["BERT", "BERT-large", "RoBERTa", "RoBERTa-large"]:
            self.embedder = self._initialize_embedder(True,
                                                      start=start,
                                                      end=end)
        else:
            self.embedder = self._initialize_embedder(False)

        if self.verbose:
            print("Created corresponding embedder.")

        return

    def _initialize_embedder(self, is_mlm, start=None, end=None):
        """
        Returns a function that embeds sentences with the selected
        language model.

        Args:
            is_mlm: bool (optional) indicating if self.lm_model is an mlm.
                Default
            start: str representing start token for MLMs.
                Must be set if is_mlm == True.
            end: str representing end token for MLMs.
                Must be set if is_mlm == True.

        Returns:
            function that takes in a query string and outputs a
                [batch size=1, hidden state size] summary embedding
                using self.lm_model
        """
        if not is_mlm:

            def embedder(query_str):
                tokens_tensor = torch.tensor(
                    self.tokenizer.encode(query_str,
                                          add_special_tokens=False,
                                          return_tensors="pt").to(self.device))

                outputs = self.lm_model(tokens_tensor)
                print(outputs)
                print(outputs.last_hidden_state.shape)
                # Shape (batch size=1, hidden state size)
                return outputs.last_hidden_state[:, -1]

        else:

            def embedder(query_str):
                query_str = start + " " + query_str + " " + end
                tokenized_text = self.tokenizer.tokenize(query_str)
                tokens_tensor = torch.tensor(
                    [self.tokenizer.convert_tokens_to_ids(tokenized_text)])
                """ tokens_tensor = torch.tensor([indexed_tokens.to(self.device)])
                 """
                tokens_tensor = tokens_tensor.to(
                    self.device)  # if you have gpu

                with torch.no_grad():
                    outputs = self.lm_model(tokens_tensor)
                    # hidden state is a tuple
                    hidden_state = outputs.last_hidden_state

                # Shape (batch size=1, num_tokens, hidden state size)
                # Return just the start token's embeddinge
                return hidden_state[:, -1]

        return embedder

    def reset_data(self):
        self.objects = []
        self.labels = []
        self.all_objs = []


    def extract_data(self, max_num_obj, SPLIT=""):
        """
        Extracts and saves the most interesting objects from each room.

        TODO: Finish docstring
        """
        self.max_num_obj = max_num_obj
        self.reset_data()

        scenes = glob.glob("data/scene_datasets/hm3d/"+SPLIT+"/*/*.basis.glb")
        dataset_path = glob.glob("data/datasets/objectgoal_hm3d/"+SPLIT+"/content/*.json.gz")
        split_dataset_path = []
        for datasets in dataset_path:
            split_dataset_path.append(os.path.split(datasets)[1].split('.')[0])
        print(scenes)
        print(len(scenes))
        count = 0
        for sce in scenes:
            split_sce = os.path.split(sce)[1].split('.')[0]

            if split_sce in split_dataset_path:
                config=habitat.get_config("envs/habitat/configs/tasks/objectnav_hm3d.yaml")

                config.defrost()
                # config.DATASET.SPLIT = SPLIT
                # config.SIMULATOR.SCENE_DATASET = "./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
                config.SIMULATOR.SCENE = sce
                # config.SIMULATOR.AGENT_0.SENSORS = []
                config.freeze()


                sim = habitat.sims.make_sim("Sim-v0", config=config.SIMULATOR)

                print(len(scenes))
                count+=1
                print("current count: ", count)
                scene = sim.semantic_scene
    
                for region in scene.regions:
                    # print(
                    #     f"Region id:{region.id},"
                    #     f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
                    # )
                    objs = []
                    all_obj = []
                    for obj in region.objects:
                        # print(
                        #     f"Object id:{obj.id}, category:{obj.category.name()},"
                        #     f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                        # )
                        if obj.category.name() in hm3d_semantic_mapping:
                            hm3d_category_name = hm3d_semantic_mapping[obj.category.name()]
                        else:
                            hm3d_category_name = obj.category.name()

                        if hm3d_category_name in hm3d_category and hm3d_category_name not in all_obj: 
                            all_obj.append(hm3d_category_name)

                    goal_list = list(set(all_obj) & set(category_to_id))

                    if len(all_obj) > 1:
                        for goal in goal_list:
                            temp_obj = []
                            for obj in all_obj:
                                if obj != goal:
                                    temp_obj.append(obj)
                            self.objects.append(temp_obj)
                            self.labels.append(goal)
                    # self.all_objs.append(all_obj_names)
                sim.close()
                
    def generate_data(self, k, num_objs, all_permutations=True, skip_rms=True):
        """
        Constructs query string using selected number of objects

        Args:
            k: int <= num_objs number of objects to include in
                query string
            num_objs: int <= self.max_num_objs number of objects
                to choose k out of when generating query strings. Prioritizes
                most semantically interesting objects.

        Returns:
            Tuple of (list of strs, torch.tensor, torch.tensor, torch.tensor).
                Respectively:
                1) list of query sentences of length
                    (# rooms) * (num_obs P k)
                2) tensor of int room labels corresponding to above list
                3) tensor of sentence embeddings corresponding to above list
                4) tensor of sentence embeddings corresponding to room label string
        """
        query_sentence_list = []
        label_list = []
        query_embedding_list = []
        goal_embedding_list = []
        all_objs_list = []

        for objs, label in tqdm(
                zip(self.objects, self.labels)):
            if skip_rms:
                if len(objs) < num_objs:
                    continue
            else:
                if len(objs) == 0:
                    continue

            k_room = min(len(objs), k)
            n = min(len(objs), num_objs)

            np_objs = objs[:n]
            np_label = label
            if all_permutations:
                for objs_p in multiset_permutations(np_objs, k_room):
                    # objs_p = torch.tensor(np_objs_p)
                    query_str = self._object_query_constructor(objs_p)
                    goal_str = np_label

                    query_embedding = self.embedder(query_str)
                    goal_embedding = self.embedder(goal_str)

                    query_sentence_list.append(query_str)
                    label_list.append(category_to_id.index(goal_str))
                    query_embedding_list.append(query_embedding)
                    goal_embedding_list.append(goal_embedding)
            else:
                # objs_p = torch.tensor(np_objs)
                query_str = self._object_query_constructor(objs_p)
                goal_str = np_label

                query_embedding = self.embedder(query_str)
                goal_embedding = self.embedder(goal_str)

                query_sentence_list.append(query_str)
                label_list.append(label)
                query_embedding_list.append(query_embedding)
                goal_embedding_list.append(goal_embedding)
        return (
            query_sentence_list,
            torch.tensor(label_list),
            torch.cat(query_embedding_list),
            torch.cat(goal_embedding_list),
        )

    def _object_query_constructor(self, objects):
        """
        Construct a query string based on a list of objects

        Args:
            objects: torch.tensor of object indices contained in a room

        Returns:
            str query describing the room, eg "This is a room containing
                toilets and sinks."
        """
        assert len(objects) > 0
        query_str = "This room contains "
        names = []
        for ob in objects:
            names.append(ob)
        if len(names) == 1:
            query_str += names[0]
        elif len(names) == 2:
            query_str += names[0] + " and " + names[1]
        else:
            for name in names[:-1]:
                query_str += name + ", "
            query_str += "and " + names[-1]
        query_str += "."
        return query_str

    def _room_str_constructor(self, room):
        room_str = self.room_list[room]
        if room_str != "utility room" and room_str[0] in "aeiou":
            return "An " + room_str + "."
        else:
            return "A " + room_str + "."

    def data_split_generator(self, data_generation_params, k_test):
        max_n = np.max([i[1] for i in data_generation_params])

        split_dict = {}

        # Train
        dg.extract_data(max_n, SPLIT="train")
        TEMP = {}
        count = 0
        for k, total in data_generation_params:
            suffix = "train_k" + str(k) + "_total" + str(total)
            sentences, labels, query_embeddings, goal_embedding = dg.generate_data(
                k, total)
            count += len(sentences)
            TEMP[suffix] = [
                sentences, labels, query_embeddings,
                goal_embedding
            ]
        split_dict["train"] = TEMP
        print(count, "train sentences")

        # Val
        dg.extract_data(max_n, SPLIT="val")
        TEMP = {}
        count = 0
        for k, total in data_generation_params:
            suffix = "val_k" + str(k) + "_total" + str(total)
            sentences, labels, query_embeddings, goal_embedding = dg.generate_data(
                k, total)
            count += len(sentences)
            TEMP[suffix] = [
                sentences, labels, query_embeddings,
                goal_embedding
            ]
        split_dict["val"] = TEMP
        print(count, "val sentences")

        # Test
        if k_test > 0:
            dg.extract_data(max_n, SPLIT="test")
            TEMP = {}
            count = 0
            suffix = "test_k" + str(k_test)
            sentences, all_objs_list, labels, query_embeddings, room_embeddings = dg.generate_data(
                k_test, k_test, all_permutations=False, skip_rms=False)
            count += len(sentences)
            TEMP[suffix] = [
                sentences, all_objs_list, labels, query_embeddings,
                room_embeddings
            ]
            split_dict["test"] = TEMP
            print(count, "test sentences")
        return split_dict


if __name__ == "__main__":
    for lm in ["RoBERTa-large", "BERT-large"]:
        for label_set in ["mpcat40"]:
            for use_gt in [True, False]:
                data_folder = os.path.join(
                    "./llm_priors/data/",
                    lm + "_" + label_set + "_useGT_" + str(use_gt) + "_502030")
                if not os.path.exists(data_folder):
                    os.makedirs(data_folder)
                for split in ["train", "val"]:
                    if not os.path.exists(os.path.join(data_folder, split)):
                        os.makedirs(os.path.join(data_folder, split))

                dg = DataGenerator(verbose=True,
                                   label_set=label_set,
                                   use_gt_cooccurrencies=use_gt)
                dg.configure_lm(lm)

                data_generation_params = [(1, 1), (2, 2), (3, 3), (1, 2),
                                          (2, 3), (3, 4)]
                k_test = 0

                split_dict = dg.data_split_generator(data_generation_params,
                                                     k_test)
                # Save
                splits = ["train", "val", "test"
                          ] if k_test > 0 else ["train", "val"]
                for split in splits:
                    for suffix in split_dict[split]:
                        sentences, labels, query_embeddings, goal_embedding = split_dict[
                            split][suffix]

                        # Save query sentences
                        with open(
                                os.path.join(
                                    data_folder, split,
                                    "query_sentences_" + suffix + ".pkl"),
                                "wb",
                        ) as fp:
                            pickle.dump(sentences, fp)

                        # Save labels
                        torch.save(
                            labels,
                            os.path.join(data_folder, split,
                                         "labels_" + suffix + ".pt"))

                        # Save query embeddings
                        torch.save(
                            query_embeddings,
                            os.path.join(data_folder, split,
                                         "query_embeddings_" + suffix + ".pt"),
                        )

                        # Save room embeddings
                        torch.save(
                            goal_embedding,
                            os.path.join(data_folder, split,
                                         "goal_embeddings_" + suffix + ".pt"),
                        )