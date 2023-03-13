from cgi import test
import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor


class FinetuningDataset(Dataset):

    def __init__(self, lm=None, label_set=None, co_suffix=""):
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.sentences = []

        if lm is not None and label_set is not None:
            self.lm = lm
            data_path = os.path.join(
                "./data",
                lm + "_" + label_set + co_suffix,
            )

            print(data_path)

            suffixes = []
            for file_name in os.listdir(data_path):
                if "labels_" in file_name:
                    suffixes.append(file_name[len("labels_"):-len(".pt")])

            sent_files = [
                "query_sentences_" + suffix + ".pkl" for suffix in suffixes
            ]
            label_files = ["labels_" + suffix + ".pt" for suffix in suffixes]
            q_emb_files = [
                "query_embeddings_" + suffix + ".pt" for suffix in suffixes
            ]
            r_emb_files = [
                "room_embeddings_" + suffix + ".pt" for suffix in suffixes
            ]

            # Check for correspondencies between inputs and labels
            for file_name in sent_files + label_files + q_emb_files + r_emb_files:
                assert file_name in os.listdir(data_path)

            for file_name in sent_files:
                with open(os.path.join(data_path, file_name), "rb") as fp:
                    self.sentences += pickle.load(fp)

            self.labels = torch.cat(
                [
                    torch.load(os.path.join(data_path, file_name))
                    for file_name in label_files
                ],
                dim=0,
            ).to(self.device)

            self.query_embeddings = torch.vstack([
                torch.load(os.path.join(data_path, file_name))
                for file_name in q_emb_files
            ]).to(self.device)

            self.room_embeddings = torch.vstack([
                torch.load(os.path.join(data_path, file_name))
                for file_name in r_emb_files
            ]).to(self.device)

            # print(len(self.sentences))
            # print(self.labels.shape)
            # print(self.query_embeddings.shape)
            # print(self.room_embeddings.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.query_embeddings[idx], self.room_embeddings[idx],
                self.labels[idx])

    def _populate_split_ds(self, ds, inds):
        for i in inds:
            ds.sentences.append(self.sentences[i])
        inds = torch.tensor(inds)
        ds.labels = self.labels[inds].to(self.device)
        ds.query_embeddings = self.query_embeddings[inds].to(self.device)
        ds.room_embeddings = self.room_embeddings[inds].to(self.device)
        ds.building_list, ds.room_list, ds.object_list = (
            self.building_list,
            self.room_list,
            self.object_list,
        )
        ds.building_list_pl, ds.room_list_pl, ds.object_list_pl = (
            self.building_list_pl,
            self.room_list_pl,
            self.object_list_pl,
        )
        return ds

    def create_split(self, train_frac, val_frac, seed=0):
        train_ds, val_ds, test_ds = (
            FinetuningDataset(),
            FinetuningDataset(),
            FinetuningDataset(),
        )
        torch.manual_seed(seed)
        shuffle = torch.randperm(len(self.labels)).to(self.device)

        train_inds = shuffle[:int(train_frac * len(self.labels))]
        val_inds = shuffle[int(train_frac *
                               len(self.labels)):int((train_frac + val_frac) *
                                                     len(self.labels))]
        test_inds = shuffle[int((train_frac + val_frac) * len(self.labels)):]

        train_ds = self._populate_split_ds(train_ds, train_inds)
        val_ds = self._populate_split_ds(val_ds, val_inds)
        test_ds = self._populate_split_ds(test_ds, test_inds)

        return train_ds, val_ds, test_ds

    def create_holdout_split(self,
                             train_frac,
                             val_frac,
                             holdout_objs,
                             holdout_rooms,
                             seed=0):
        train_ds, val_ds, test_ds, holdout_ds = (
            FinetuningDataset(),
            FinetuningDataset(),
            FinetuningDataset(),
            HoldoutDataset(),
        )
        torch.manual_seed(seed)
        shuffle = torch.randperm(len(self.labels))
        holdout_inds = []
        holdout_terms = []
        non_holdout_inds = []

        for i in shuffle:
            query_sentence = self.sentences[i]
            label_str = self.room_list[self.labels[i]]
            held_out = False
            for ho_room in holdout_rooms:
                if ho_room in label_str:
                    holdout_inds.append(int(i))
                    held_out = True
                    holdout_terms.append(ho_room)
                    break
            if not held_out:
                for ho_obj in holdout_objs:
                    if ho_obj in query_sentence:
                        holdout_inds.append(int(i))
                        held_out = True
                        holdout_terms.append(ho_obj)
                        break
            if not held_out:
                non_holdout_inds.append(int(i))

        train_inds = non_holdout_inds[:int(train_frac * len(non_holdout_inds))]
        val_inds = non_holdout_inds[int(train_frac *
                                        len(non_holdout_inds)):int(
                                            (train_frac + val_frac) *
                                            len(non_holdout_inds))]
        test_inds = non_holdout_inds[int((train_frac + val_frac) *
                                         len(non_holdout_inds)):]

        train_ds = self._populate_split_ds(train_ds, train_inds)
        val_ds = self._populate_split_ds(val_ds, val_inds)
        test_ds = self._populate_split_ds(test_ds, test_inds)
        holdout_ds = self._populate_split_ds(holdout_ds, holdout_inds)
        holdout_ds.set_holdout_terms(holdout_terms)

        return train_ds, val_ds, test_ds, holdout_ds


class HoldoutDataset(FinetuningDataset):

    def set_holdout_terms(self, holdout_terms):
        self.holdout_terms = holdout_terms

    def __getitem__(self, idx):
        return (
            self.query_embeddings[idx],
            self.room_embeddings[idx],
            self.labels[idx],
            self.holdout_terms[idx],
        )


class BuildingDataset(Dataset):

    def __init__(self, embeddings, label_tensor):
        # Extract object, room, and bldg labels
        dataset = Matterport3dDataset(
            "./mp_data/nyuClass_matterport3d_w_edge.pkl")
        labels, pl_labels = create_label_lists(dataset)
        self.building_list, self.room_list, self.object_list = labels
        self.building_list_pl, self.room_list_pl, self.object_list_pl = pl_labels

        del dataset

        self.query_embeddings = embeddings
        self.labels = label_tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.query_embeddings[idx], self.labels[idx])


class RoomDataset(Dataset):

    def __init__(self,
                 path_to_data,
                 device="cuda",
                 return_sentences=False,
                 return_all_objs=True):

        self.device = device
        self.return_sentences = return_sentences
        self.return_all_objs = return_all_objs

        # Initialize data attrs
        self.query_embeddings = []
        self.goal_embeddings = []
        self.labels = []
        if self.return_sentences:
            self.sentences = []
        if self.return_all_objs:
            self.all_objs = []

        # Extract all suffixes
        suffixes = []
        for file in os.listdir(path_to_data):
            if "labels_" in file:
                suffixes.append(file[len("labels"):-len(".pt")])

        for s in suffixes:
            query_embeddings = torch.load(
                os.path.join(path_to_data, "query_embeddings" + s + ".pt"))
            goal_embeddings = torch.load(
                os.path.join(path_to_data, "goal_embeddings" + s + ".pt"))
            labels = torch.load(
                os.path.join(path_to_data, "labels" + s + ".pt"))
            if self.return_sentences:
                with open(
                        os.path.join(path_to_data,
                                     "query_sentences" + s + ".pkl"),
                        "rb") as fp:
                    self.sentences += pickle.load(fp)
            if self.return_all_objs:
                with open(os.path.join(path_to_data, "all_objs" + s + ".pkl"),
                          "rb") as fp:
                    self.all_objs += pickle.load(fp)
            self.query_embeddings.append(query_embeddings)
            self.goal_embeddings.append(goal_embeddings)
            self.labels.append(labels)

        self.query_embeddings = torch.cat(self.query_embeddings).to(
            self.device)
        self.goal_embeddings = torch.cat(self.goal_embeddings).to(self.device)
        self.labels = torch.cat(self.labels).to(self.device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        res = [
            self.query_embeddings[idx], self.goal_embeddings[idx],
            self.labels[idx]
        ]
        if self.return_sentences:
            res += [self.sentences[idx]]
        if self.return_all_objs:
            res += [self.all_objs[idx]]
        return res


def create_building_splits(path_to_data_dir, split_ratio, device, seed=0):
    path_to_embeddings = os.path.join(path_to_data_dir, "query_embeddings.pt")
    path_to_labels = os.path.join(path_to_data_dir, "labels.pt")

    embeddings = torch.load(path_to_embeddings).to(device)
    labels = torch.load(path_to_labels).to(device)

    total_num = len(embeddings)
    train_ind = int(total_num * split_ratio[0])
    val_ind = int(total_num * (split_ratio[0] + split_ratio[1]))

    torch.manual_seed(0)
    shuffle_inds = torch.randperm(total_num).to(device)
    embeddings = embeddings[shuffle_inds]
    labels = torch.tensor(labels[shuffle_inds])

    train_ds = BuildingDataset(embeddings[:train_ind], labels[:train_ind])
    val_ds = BuildingDataset(embeddings[train_ind:val_ind],
                             labels[train_ind:val_ind])
    test_ds = BuildingDataset(embeddings[val_ind:], labels[val_ind:])

    return train_ds, val_ds, test_ds


def create_comparison_building_splits(path_to_data_dir, device, seed=0):
    ds_list = []
    for split in ["train", "val", "test"]:
        path_to_embeddings = os.path.join(path_to_data_dir,
                                          "query_embeddings_" + split + ".pt")
        path_to_labels = os.path.join(path_to_data_dir,
                                      "labels_" + split + ".pt")

        embeddings = torch.load(path_to_embeddings).to(device)
        labels = torch.load(path_to_labels).to(device)

        total_num = len(embeddings)
        torch.manual_seed(0)
        shuffle_inds = torch.randperm(total_num).to(device)
        embeddings = embeddings[shuffle_inds]
        labels = torch.tensor(labels[shuffle_inds])

        ds = BuildingDataset(embeddings, labels)
        ds_list.append(ds)

    train_ds, val_ds, test_ds = ds_list
    return train_ds, val_ds, test_ds


def create_room_splits(path_to_data,
                       device="cuda",
                       return_sentences=False,
                       return_all_objs=False):

    train_ds = RoomDataset(os.path.join(path_to_data, "train"),
                           device=device,
                           return_sentences=return_sentences,
                           return_all_objs=return_all_objs)
    val_ds = RoomDataset(os.path.join(path_to_data, "val"),
                         device=device,
                         return_sentences=return_sentences,
                         return_all_objs=return_all_objs)
    # test_ds = RoomDataset(os.path.join(path_to_data, "test"),
    #                       device=device,
    #                       return_sentences=return_sentences,
    #                       return_all_objs=return_all_objs)
    test_ds = None
    return train_ds, val_ds, test_ds