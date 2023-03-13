from collections import deque, defaultdict
from itertools import count
import os
import logging
import time
import json
import gym
import torch.nn as nn
import torch
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
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


from skimage import measure
import skimage.morphology

import cv2


from model import Semantic_Mapping, FeedforwardNet
from envs.utils.fmm_planner import FMMPlanner
from envs import make_vec_envs
from arguments import get_args
import algo

from constants import category_to_id, hm3d_category, category_to_id_gibson

import envs.utils.pose as pu

os.environ["OMP_NUM_THREADS"] = "1"


fileName = 'data/matterport_category_mappings.tsv'

text = ''
lines = []
items = []
hm3d_semantic_mapping={}
hm3d_semantic_index={}
hm3d_semantic_index_inv={}

with open(fileName, 'r') as f:
    text = f.read()
lines = text.split('\n')[1:]

for l in lines:
    items.append(l.split('    '))

for i in items:
    if len(i) > 3:
        hm3d_semantic_mapping[i[2]] = i[-1]
        hm3d_semantic_index[i[-1]] = int(i[-2])
        hm3d_semantic_index_inv[int(i[-2])] = i[-1]


def find_big_connect(image):
    img_label, num = measure.label(image, connectivity=2, return_num=True)#输出二值图像中所有的连通域
    props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
    # print("img_label.shape: ", img_label.shape) # 480*480
    resMatrix = np.zeros(img_label.shape)
    tmp_area = 0
    for i in range(0, len(props)):
        if props[i].area > tmp_area:
            tmp = (img_label == i + 1).astype(np.uint8)
            resMatrix = tmp
            tmp_area = props[i].area 
    
    return resMatrix
    

def main():
    args = get_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Setup Logging
    log_dir = "{}/models/{}/".format(args.dump_location, args.exp_name)
    dump_dir = "{}/dump/{}/".format(args.dump_location, args.exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    logging.basicConfig(
        filename=log_dir + 'train.log',
        level=logging.INFO)
    print("Dumping at {}".format(log_dir))
    print(args)
    logging.info(args)

    # Logging and loss variables
    num_scenes = args.num_processes
    num_episodes = int(args.num_eval_episodes)
    device = args.device = torch.device("cuda:0" if args.cuda else "cpu")

    g_masks = torch.ones(num_scenes).float().to(device)
    step_masks = torch.zeros(num_scenes).float().to(device)

    if args.eval:
        episode_success = []
        episode_spl = []
        episode_dist = []
        for _ in range(args.num_processes):
            episode_success.append(deque(maxlen=num_episodes))
            episode_spl.append(deque(maxlen=num_episodes))
            episode_dist.append(deque(maxlen=num_episodes))

    episode_sem_frontier = []
    episode_sem_goal = []
    episode_loc_frontier = []
    for _ in range(args.num_processes):
        episode_sem_frontier.append([])
        episode_sem_goal.append([])
        episode_loc_frontier.append([])

    finished = np.zeros((args.num_processes))
    wait_env = np.zeros((args.num_processes))

    g_process_rewards = 0
    g_total_rewards = np.ones((num_scenes))
    g_sum_rewards = 1
    g_sum_global = 1

    stair_flag = np.zeros((num_scenes))
    clear_flag = np.zeros((num_scenes))

    # Starting environments
    torch.set_num_threads(1)
    envs = make_vec_envs(args)
    obs, infos = envs.reset()

    torch.set_grad_enabled(False)

    # Initialize map variables:
    # Full map consists of multiple channels containing the following:
    # 1. Obstacle Map
    # 2. Exploread Area
    # 3. Current Agent Location
    # 4. Past Agent Locations
    # 5,6,7,.. : Semantic Categories
    nc = args.num_sem_categories + 4  # num channels

    # Calculating full and local map sizes
    map_size = args.map_size_cm // args.map_resolution
    full_w, full_h = map_size, map_size # 2400/5=480
    local_w = int(full_w / args.global_downscaling)
    local_h = int(full_h / args.global_downscaling)

    # Initializing full and local map
    full_map = torch.zeros(num_scenes, nc, full_w, full_h).float().to(device)
    local_map = torch.zeros(num_scenes, nc, local_w,
                            local_h).float().to(device)

    local_ob_map = np.zeros((num_scenes, local_w,
                            local_h))

    local_ex_map = np.zeros((num_scenes, local_w,
                            local_h))

    target_edge_map = np.zeros((num_scenes, local_w,
                            local_h))
    target_point_map = np.zeros((num_scenes, local_w,
                            local_h))


    # dialate for target map
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3, 3))
    tv_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7, 7))

    # Initial full and local pose
    full_pose = torch.zeros(num_scenes, 3).float().to(device)
    local_pose = torch.zeros(num_scenes, 3).float().to(device)

    # Origin of local map
    origins = np.zeros((num_scenes, 3))

    # Local Map Boundaries
    lmb = np.zeros((num_scenes, 4)).astype(int)

    # Planner pose inputs has 7 dimensions
    # 1-3 store continuous global agent location
    # 4-7 store local map boundaries
    planner_pose_inputs = np.zeros((num_scenes, 7))

    frontier_score_list = []
    for _ in range(args.num_processes):
        frontier_score_list.append(deque(maxlen=10))

    
    object_norm_inv_perplexity = torch.tensor(np.load('data/object_norm_inv_perplexity.npy')).to(device)

    def get_local_map_boundaries(agent_loc, local_sizes, full_sizes):
        loc_r, loc_c = agent_loc
        local_w, local_h = local_sizes
        full_w, full_h = full_sizes

        if args.global_downscaling > 1:
            gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
            gx2, gy2 = gx1 + local_w, gy1 + local_h
            if gx1 < 0:
                gx1, gx2 = 0, local_w
            if gx2 > full_w:
                gx1, gx2 = full_w - local_w, full_w

            if gy1 < 0:
                gy1, gy2 = 0, local_h
            if gy2 > full_h:
                gy1, gy2 = full_h - local_h, full_h
        else:
            gx1, gx2, gy1, gy2 = 0, full_w, 0, full_h

        return [gx1, gx2, gy1, gy2]

    def get_frontier_boundaries(frontier_loc, frontier_sizes, map_sizes):
        loc_r, loc_c = frontier_loc
        local_w, local_h = frontier_sizes
        full_w, full_h = map_sizes

        gx1, gy1 = loc_r - local_w // 2, loc_c - local_h // 2
        gx2, gy2 = gx1 + local_w, gy1 + local_h
        if gx1 < 0:
            gx1, gx2 = 0, local_w
        if gx2 > full_w:
            gx1, gx2 = full_w - local_w, full_w

        if gy1 < 0:
            gy1, gy2 = 0, local_h
        if gy2 > full_h:
            gy1, gy2 = full_h - local_h, full_h
 
        return [int(gx1), int(gx2), int(gy1), int(gy2)]

    def init_map_and_pose():
        full_map.fill_(0.)
        full_pose.fill_(0.)
        full_pose[:, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]

            full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

            lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                              (local_w, local_h),
                                              (full_w, full_h))

            planner_pose_inputs[e, 3:] = lmb[e]
            origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                          lmb[e][0] * args.map_resolution / 100.0, 0.]

        for e in range(num_scenes):
            local_map[e] = full_map[e, :,
                                    lmb[e, 0]:lmb[e, 1],
                                    lmb[e, 2]:lmb[e, 3]]
            local_pose[e] = full_pose[e] - \
                torch.from_numpy(origins[e]).to(device).float()

    def init_map_and_pose_for_env(e):
        full_map[e].fill_(0.)
        full_pose[e].fill_(0.)
        local_ob_map[e]=np.zeros((local_w,
                            local_h))
        local_ex_map[e]=np.zeros((local_w,
                            local_h))
        target_edge_map[e]=np.zeros((local_w,
                            local_h))
        target_point_map[e]=np.zeros((local_w,
                            local_h))

        step_masks[e]=0
        stair_flag[e] = 0
        clear_flag[e] = 0


        full_pose[e, :2] = args.map_size_cm / 100.0 / 2.0

        locs = full_pose[e].cpu().numpy()
        planner_pose_inputs[e, :3] = locs
        r, c = locs[1], locs[0]
        loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                        int(c * 100.0 / args.map_resolution)]

        full_map[e, 2:4, loc_r - 1:loc_r + 2, loc_c - 1:loc_c + 2] = 1.0

        lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                          (local_w, local_h),
                                          (full_w, full_h))

        planner_pose_inputs[e, 3:] = lmb[e]
        origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                      lmb[e][0] * args.map_resolution / 100.0, 0.]

        local_map[e] = full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]]
        local_pose[e] = full_pose[e] - \
            torch.from_numpy(origins[e]).to(device).float()

    init_map_and_pose()


    def remove_small_points(local_ob_map, image, threshold_point, pose):
        # print("goal_cat_id: ", goal_cat_id)
        # print("sem: ", sem.shape)
        selem = skimage.morphology.disk(1)
        traversible = skimage.morphology.binary_dilation(
            local_ob_map, selem) != True
        # traversible = 1 - traversible
        planner = FMMPlanner(traversible)
        goal_pose_map = np.zeros((local_ob_map.shape))
        pose_x = int(pose[0].cpu()) if int(pose[0].cpu()) < local_w-1 else local_w-1
        pose_y = int(pose[1].cpu()) if int(pose[1].cpu()) < local_w-1 else local_w-1
        goal_pose_map[pose_x, pose_y] = 1
        # goal_map = skimage.morphology.binary_dilation(
        #     goal_pose_map, selem) != True
        # goal_map = 1 - goal_map
        planner.set_multi_goal(goal_pose_map)

        img_label, num = measure.label(image, connectivity=2, return_num=True)#输出二值图像中所有的连通域
        props = measure.regionprops(img_label)#输出连通域的属性，包括面积等
        # print("img_label.shape: ", img_label.shape) # 480*480
        # print("img_label.dtype: ", img_label.dtype) # 480*480
        Goal_edge = np.zeros((img_label.shape[0], img_label.shape[1]))
        Goal_point = np.zeros(img_label.shape)
        Goal_score = []

        dict_cost = {}
        for i in range(1, len(props)):
            # print("area: ", props[i].area)
            # dist = pu.get_l2_distance(props[i].centroid[0], pose[0], props[i].centroid[1], pose[1])
            dist = planner.fmm_dist[int(props[i].centroid[0]), int(props[i].centroid[1])] * 5
            dist_s = 8 if dist < 300 else 0
            
            cost = props[i].area + dist_s

            if props[i].area > threshold_point and dist > 50 and dist < 500:
                dict_cost[i] = cost
        
        if dict_cost:
            dict_cost = sorted(dict_cost.items(), key=lambda x: x[1], reverse=True)
            
            # print(dict_cost)
            for i, (key, value) in enumerate(dict_cost):
                # print(i, key)
                Goal_edge[img_label == key + 1] = 1
                Goal_point[int(props[key].centroid[0]), int(props[key].centroid[1])] = i+1 #
                Goal_score.append(value)
                if i == 3:
                    break

        return Goal_edge, Goal_point, Goal_score

  

    def configure_lm(lm):
        """
        Configure the language model, tokenizer, and embedding generator function.

        Sets self.lm, self.lm_model, self.tokenizer, and self.embedder based on the
        selected language model inputted to this function.

        Args:
            lm: str representing name of LM to use

        Returns:
            None
        """

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

        lm_model.eval()
        lm_model = lm_model.to(device)


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

        def embedder(query_str):
            query_str = start + " " + query_str + " " + end
            tokenized_text = tokenizer.tokenize(query_str)
            tokens_tensor = torch.tensor(
                [tokenizer.convert_tokens_to_ids(tokenized_text)])
            """ tokens_tensor = torch.tensor([indexed_tokens.to(self.device)])
                """
            tokens_tensor = tokens_tensor.to(
                device)  # if you have gpu

            with torch.no_grad():
                outputs = lm_model(tokens_tensor)
                # hidden state is a tuple
                hidden_state = outputs.last_hidden_state

            # Shape (batch size=1, num_tokens, hidden state size)
            # Return just the start token's embeddinge
            return hidden_state[:, -1]

        return embedder

    def _object_query_constructor(objects):
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



    # Semantic Mapping
    sem_map_module = Semantic_Mapping(args).to(device)
    sem_map_module.eval()

    ### LLM 
    embedder = configure_lm("RoBERTa-large")

    output_size = len(category_to_id)

    ff_net = FeedforwardNet(1024, output_size)
    ff_net.to(device)

    if args.load != "0":
        print("Loading LLM model {}".format(args.load))
        state_dict = torch.load(args.load,
                                map_location=lambda storage, loc: storage)
        ff_net.load_state_dict(state_dict)
    ff_net.eval()
    

    # Predict semantic map from frame 1
    poses = torch.from_numpy(np.asarray(
        [infos[env_idx]['sensor_pose'] for env_idx in range(num_scenes)])
    ).float().to(device)

    eve_angle = np.asarray(
        [infos[env_idx]['eve_angle'] for env_idx
            in range(num_scenes)])
    

    increase_local_map, local_map, local_map_stair, local_pose = \
        sem_map_module(obs, poses, local_map, local_pose, eve_angle)

    local_map[:, 0, :, :][local_map[:, 13, :, :] > 0] = 0


    actions = torch.randn(num_scenes, 2)*6
    # print("actions: ", actions.shape)
    cpu_actions = nn.Sigmoid()(actions).cpu().numpy()
    global_goals = [[int(action[0] * local_w), int(action[1] * local_h)]
                    for action in cpu_actions]
    global_goals = [[min(x, int(local_w - 1)), min(y, int(local_h - 1))]
                    for x, y in global_goals]

    goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]

    for e in range(num_scenes):
        goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

    planner_inputs = [{} for e in range(num_scenes)]
    for e, p_input in enumerate(planner_inputs):
        p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
        p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
        p_input['pose_pred'] = planner_pose_inputs[e]
        p_input['goal'] = goal_maps[e]  # global_goals[e]
        p_input['map_target'] = target_point_map[e]  # global_goals[e]
        p_input['new_goal'] = 1
        p_input['found_goal'] = 0
        p_input['wait'] = wait_env[e] or finished[e]
        if args.visualize or args.print_images:
            p_input['map_edge'] = target_edge_map[e]
            local_map[e, -1, :, :] = 1e-5
            p_input['sem_map_pred'] = local_map[e, 4:, :, :
                                                ].argmax(0).cpu().numpy()

    obs, _, done, infos = envs.plan_act_and_preprocess(planner_inputs)

    start = time.time()
    g_reward = 0

    torch.set_grad_enabled(False)
    spl_per_category = defaultdict(list)
    success_per_category = defaultdict(list)

    for step in range(args.num_training_frames // args.num_processes + 1):
        if finished.sum() == args.num_processes:
            break

        g_step = (step // args.num_local_steps) % args.num_global_steps
        l_step = step % args.num_local_steps

        # ------------------------------------------------------------------
        # Reinitialize variables when episode ends
        l_masks = torch.FloatTensor([0 if x else 1
                                     for x in done]).to(device)
        g_masks *= l_masks

        for e, x in enumerate(done):
            if x:
                spl = infos[e]['spl']
                success = infos[e]['success']
                dist = infos[e]['distance_to_goal']
                spl_per_category[infos[e]['goal_name']].append(spl)
                success_per_category[infos[e]['goal_name']].append(success)
                if args.eval:
                    episode_success[e].append(success)
                    episode_spl[e].append(spl)
                    episode_dist[e].append(dist)
                    if len(episode_success[e]) == num_episodes:
                        finished[e] = 1
   
                wait_env[e] = 1.
                init_map_and_pose_for_env(e)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Semantic Mapping Module
        poses = torch.from_numpy(np.asarray(
            [infos[env_idx]['sensor_pose'] for env_idx
             in range(num_scenes)])
        ).float().to(device)

        eve_angle = np.asarray(
            [infos[env_idx]['eve_angle'] for env_idx
             in range(num_scenes)])
        

        increase_local_map, local_map, local_map_stair, local_pose = \
            sem_map_module(obs, poses, local_map, local_pose, eve_angle)


        locs = local_pose.cpu().numpy()
        planner_pose_inputs[:, :3] = locs + origins
        local_map[:, 2, :, :].fill_(0.)  # Resetting current location channel
        for e in range(num_scenes):
            r, c = locs[e, 1], locs[e, 0]
            loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                            int(c * 100.0 / args.map_resolution)]
            local_map[e, 2:4, loc_r - 2:loc_r + 3, loc_c - 2:loc_c + 3] = 1.

            # work for stairs in val
            # ------------------------------------------------------------------
            if args.eval:
            # # clear the obstacle during the stairs
                if loc_r > local_w: loc_r = local_w-1
                if loc_c > local_h: loc_c = local_h-1
                if infos[e]['clear_flag'] or local_map[e, 18, loc_r, loc_c] > 0.5:
                    stair_flag[e] = 1

                if stair_flag[e]:
                    # must > 0
                    if torch.any(local_map[e, 18, :, :] > 0.5):
                        local_map[e, 0, :, :] = local_map_stair[e, 0, :, :]
                    local_map[e, 0, :, :] = local_map_stair[e, 0, :, :]
            # ------------------------------------------------------------------


        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Global Policy
        if l_step == args.num_local_steps - 1:
            # For every global step, update the full and local maps
            for e in range(num_scenes):

                step_masks[e]+=1

                if wait_env[e] == 1:  # New episode
                    wait_env[e] = 0.

                
                full_map[e, :, lmb[e, 0]:lmb[e, 1], lmb[e, 2]:lmb[e, 3]] = \
                    local_map[e]
                full_pose[e] = local_pose[e] + \
                    torch.from_numpy(origins[e]).to(device).float()

                locs = full_pose[e].cpu().numpy()
                r, c = locs[1], locs[0]
                loc_r, loc_c = [int(r * 100.0 / args.map_resolution),
                                int(c * 100.0 / args.map_resolution)]

                lmb[e] = get_local_map_boundaries((loc_r, loc_c),
                                                (local_w, local_h),
                                                (full_w, full_h))

                planner_pose_inputs[e, 3:] = lmb[e]
                origins[e] = [lmb[e][2] * args.map_resolution / 100.0,
                            lmb[e][0] * args.map_resolution / 100.0, 0.]

                local_map[e] = full_map[e, :,
                                        lmb[e, 0]:lmb[e, 1],
                                        lmb[e, 2]:lmb[e, 3]]
                local_pose[e] = full_pose[e] - \
                    torch.from_numpy(origins[e]).to(device).float()

                if infos[e]['clear_flag']:
                    clear_flag[e] = 1

                if clear_flag[e]:
                    local_map[e].fill_(0.)
                    clear_flag[e] = 0

            # ------------------------------------------------------------------
          
            ### select the frontier edge            
            # ------------------------------------------------------------------
            # Edge Update
            for e in range(num_scenes):

                ############################ choose global goal map #############################
                # choose global goal map
                _local_ob_map = local_map[e][0].cpu().numpy()
                local_ob_map[e] = cv2.dilate(_local_ob_map, kernel)

                show_ex = cv2.inRange(local_map[e][1].cpu().numpy(),0.1,1)
                
                kernel = np.ones((5, 5), dtype=np.uint8)
                free_map = cv2.morphologyEx(show_ex, cv2.MORPH_CLOSE, kernel)

                contours,_=cv2.findContours(free_map, cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
                if len(contours)>0:
                    contour = max(contours, key = cv2.contourArea)
                    cv2.drawContours(local_ex_map[e],contour,-1,1,1)

                # clear the boundary
                local_ex_map[e, 0:2, 0:local_w]=0.0
                local_ex_map[e, local_w-2:local_w, 0:local_w-1]=0.0
                local_ex_map[e, 0:local_w, 0:2]=0.0
                local_ex_map[e, 0:local_w, local_w-2:local_w]=0.0
                
                target_edge = np.zeros((local_w, local_h))
                target_edge = local_ex_map[e]-local_ob_map[e]

                target_edge[target_edge>0.8]=1.0
                target_edge[target_edge!=1.0]=0.0

                local_pose_map = [local_pose[e][1]*100/args.map_resolution, local_pose[e][0]*100/args.map_resolution]
                target_edge_map[e], target_point_map[e], Goal_score = remove_small_points(_local_ob_map, target_edge, 4, local_pose_map) 
  


                local_ob_map[e]=np.zeros((local_w,
                        local_h))
                local_ex_map[e]=np.zeros((local_w,
                        local_h))

                # ------------------------------------------------------------------

                ##### LLM frontier score
                # ------------------------------------------------------------------

                cn = infos[e]['goal_cat_id'] + 4
                cname = infos[e]['goal_name'] 
                frontier_score_list[e] = []
                tpm = len(list(set(target_point_map[e].ravel()))) -1
                
                for lay in range(tpm):
                    f_pos = np.argwhere(target_point_map[e] == lay+1)
                    fmb = get_frontier_boundaries((f_pos[0][0], f_pos[0][1]),
                                                    (local_w/4, local_h/4),
                                                    (local_w, local_h))
                    objs_list = []
                    for se_cn in range(args.num_sem_categories-1):
                        if local_map[e][se_cn+4, fmb[0]:fmb[1], fmb[2]:fmb[3]].sum() != 0.:
                            objs_list.append(hm3d_category[se_cn])

                    if len(objs_list)>0:

                        objs_p = [hm3d_semantic_index[obj] for obj in objs_list]
                        objs_p = torch.tensor(objs_p)
                        y_object = F.one_hot(objs_p, 42).type(torch.LongTensor)
                        # np_objs = objs
                        y_object = y_object.to(device)

                        scores = y_object * object_norm_inv_perplexity.reshape(
                                [1, -1])

                        maxes = torch.max(scores, dim=1).values
                        top_max_inds = torch.topk(maxes, max(min((maxes > 0).sum(), 3),
                                                                1)).indices
                        objs = torch.argmax(scores[top_max_inds], dim=1)
                        objs = torch.where(
                            torch.bincount(objs, minlength=len(objs)) > 0)[0]
                        # for objs_p in multiset_permutations(np_objs, k_room):
                        objs = objs.cpu().numpy()
                        objs_n = [hm3d_semantic_index_inv[obj] for obj in objs]
                       
                        query_str = _object_query_constructor(objs_n)
                        # query_str = torch.tensor(query_str)
                        query_embedding = embedder(query_str)
                        pred = ff_net(query_embedding)
                        pred = nn.Softmax(dim=1)(pred)
               
                        frontier_score_list[e].append(pred[0][hm3d_category.index(cname)].cpu().numpy()) 
                      
                    else:
                        frontier_score_list[e].append(Goal_score[lay]/max(Goal_score) * 0.1 + 0.1) 


            # ------------------------------------------------------------------

            ##### select randomly point
            # ------------------------------------------------------------------
            actions = torch.randn(num_scenes, 2)*6
            cpu_actions = nn.Sigmoid()(actions).numpy()
            global_goals = [[int(action[0] * local_w),
                                int(action[1] * local_h)]
                            for action in cpu_actions]
            global_goals = [[min(x, int(local_w - 1)),
                                min(y, int(local_h - 1))]
                            for x, y in global_goals]

            g_masks = torch.ones(num_scenes).float().to(device)

            # --------------------------------------------------------------------


        # ------------------------------------------------------------------
        # Update long-term goal if target object is found
        found_goal = [0 for _ in range(num_scenes)]
    
        local_goal_maps = [np.zeros((local_w, local_h)) for _ in range(num_scenes)]
        


        for e in range(num_scenes):

            # ------------------------------------------------------------------
            ##### select frontier point
            # ------------------------------------------------------------------
            global_item = 0
            if len(frontier_score_list[e]) > 0:
                if max(frontier_score_list[e]) > 0.2:
                    global_item = frontier_score_list[e].index(max(frontier_score_list[e]))
                
                # elif max(frontier_score_list[e]) > 0.1:
                #     for f_score in frontier_score_list[e]:
                #         if f_score > 0.1:
                #             break
                #         else:
                #             global_item += 1
                # else:
                #     global_item = 0
                #------------------------------------------------------------------

                ###### Get llm frontier reward
                
                # ------------------------------------------------------------------
                if max(frontier_score_list[e]) > 0.1:
                    if args.task_config == "tasks/objectnav_gibson.yaml":
                        g_reward = infos[e]['g_reward']
                        g_process_rewards += g_reward
                    g_sum_rewards += 1
                    # print("get llm result!")


            if np.any(target_point_map[e] == global_item+1):
                local_goal_maps[e][target_point_map[e] == global_item+1] = 1
                # print("Find the edge")
                g_sum_global += 1
            else:
                local_goal_maps[e][global_goals[e][0], global_goals[e][1]] = 1

                # print("Don't Find the edge")

            cn = infos[e]['goal_cat_id'] + 4
            if local_map[e, cn, :, :].sum() != 0.:
                # print("Find the target")
                cat_semantic_map = local_map[e, cn, :, :].cpu().numpy()
                cat_semantic_scores = cat_semantic_map
                cat_semantic_scores[cat_semantic_scores > 0] = 1.
                if cn == 9:
                    cat_semantic_scores = cv2.dilate(cat_semantic_scores, tv_kernel)
                local_goal_maps[e] = find_big_connect(cat_semantic_scores)
                found_goal[e] = 1

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------
        # Take action and get next observation
        planner_inputs = [{} for e in range(num_scenes)]
        for e, p_input in enumerate(planner_inputs):
            # planner_pose_inputs[e, 3:] = [0, local_w, 0, local_h]
            p_input['map_pred'] = local_map[e, 0, :, :].cpu().numpy()
            p_input['exp_pred'] = local_map[e, 1, :, :].cpu().numpy()
            p_input['pose_pred'] = planner_pose_inputs[e]
            p_input['goal'] = local_goal_maps[e]  # global_goals[e]
            p_input['map_target'] = target_point_map[e]  # global_goals[e]
            p_input['new_goal'] = l_step == args.num_local_steps - 1
            p_input['found_goal'] = found_goal[e]
            p_input['wait'] = wait_env[e] or finished[e]
            if args.visualize or args.print_images:
                p_input['map_edge'] = target_edge_map[e]
                local_map[e, -1, :, :] = 1e-5
                p_input['sem_map_pred'] = local_map[e, 4:, :,
                                                :].argmax(0).cpu().numpy()
   

        obs, fail_case, done, infos = envs.plan_act_and_preprocess(planner_inputs)
        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

        # ------------------------------------------------------------------

        if step % args.log_interval == 0:
            end = time.time()
            time_elapsed = time.gmtime(end - start)
            log = " ".join([
                "Time: {0:0=2d}d".format(time_elapsed.tm_mday - 1),
                "{},".format(time.strftime("%Hh %Mm %Ss", time_elapsed)),
                "num timesteps {},".format(step * num_scenes),
                "FPS {},".format(int(step * num_scenes / (end - start)))
            ])

            log += "\n\tLLM Rewards: " + str(g_process_rewards /g_sum_rewards)
            log += "\n\tLLM use rate: " + str(g_sum_rewards /g_sum_global)

            if args.eval:
                total_success = []
                total_spl = []
                total_dist = []
                for e in range(args.num_processes):
                    for acc in episode_success[e]:
                        total_success.append(acc)
                    for dist in episode_dist[e]:
                        total_dist.append(dist)
                    for spl in episode_spl[e]:
                        total_spl.append(spl)

                if len(total_spl) > 0:
                    log += " ObjectNav succ/spl/dtg:"
                    log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                        np.mean(total_success),
                        np.mean(total_spl),
                        np.mean(total_dist),
                        len(total_spl))

                total_collision = []
                total_exploration = []
                total_detection = []
                total_success = []
                for e in range(args.num_processes):
                    total_collision.append(fail_case[e]['collision'])
                    total_exploration.append(fail_case[e]['exploration'])
                    total_detection.append(fail_case[e]['detection'])
                    total_success.append(fail_case[e]['success'])

                if len(total_spl) > 0:
                    log += " Fail Case: collision/exploration/detection/success:"
                    log += " {:.0f}/{:.0f}/{:.0f}/{:.0f}({:.0f}),".format(
                        np.sum(total_collision),
                        np.sum(total_exploration),
                        np.sum(total_detection),
                        np.sum(total_success),
                        len(total_spl))


            print(log)
            logging.info(log)
        # ------------------------------------------------------------------


    # Print and save model performance numbers during evaluation
    if args.eval:
        print("Dumping eval details...")
        
        log += "\n\tLLM Rewards: " + str(g_process_rewards /g_sum_rewards)
        log += "\n\tLLM use rate: " + str(g_sum_rewards /g_sum_global)


        total_success = []
        total_spl = []
        total_dist = []
        for e in range(args.num_processes):
            for acc in episode_success[e]:
                total_success.append(acc)
            for dist in episode_dist[e]:
                total_dist.append(dist)
            for spl in episode_spl[e]:
                total_spl.append(spl)

        if len(total_spl) > 0:
            log = "Final ObjectNav succ/spl/dtg:"
            log += " {:.3f}/{:.3f}/{:.3f}({:.0f}),".format(
                np.mean(total_success),
                np.mean(total_spl),
                np.mean(total_dist),
                len(total_spl))

        print(log)
        logging.info(log)
            
        # Save the spl per category
        log = "Success | SPL per category\n"
        for key in success_per_category:
            log += "{}: {} | {}\n".format(key,
                                          sum(success_per_category[key]) /
                                          len(success_per_category[key]),
                                          sum(spl_per_category[key]) /
                                          len(spl_per_category[key]))

        print(log)
        logging.info(log)

        with open('{}/{}_spl_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(spl_per_category, f)

        with open('{}/{}_success_per_cat_pred_thr.json'.format(
                dump_dir, args.split), 'w') as f:
            json.dump(success_per_category, f)



if __name__ == "__main__":
    main()
