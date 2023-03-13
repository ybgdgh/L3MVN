import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
import cv2
from collections import defaultdict
import random
import argparse
import numpy as np
import torchvision
import skimage.transform
import torch

from PIL import Image
import matplotlib.pyplot as plt
from habitat_sim.utils.common import d3_40_colors_rgb
from habitat.utils.visualizations import maps

from arguments import get_args

FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
UP_KEY="q"
DOWN_KEY="e"
FINISH="f"

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def print_scene_recur(scene, limit_output=10):
    count = 0
    # for level in scene.levels:
    #     print(
    #         f"Level id:{level.id}, center:{level.aabb.center},"
    #         f" dims:{level.aabb.sizes}"
    #     )
    #     for region in level.regions:
    #         print(
    #             f"Region id:{region.id}, category:{region.category.name()},"
    #             f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            # )
    for obj in scene.objects:
        print(
            f"Object id:{obj.id}, category:{obj.category.name()},"
            f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
        )
        print(obj.id.split("_")[-1])
        # print(dir(obj))
        # print(dir(obj.category))
        # print(dir(obj.id))
        count += 1
        if count >= limit_output:
            return None

def transform_rgb_bgr(image):
    return image[:, :, [2, 1, 0]]

def display_sample(rgb_obs, semantic_obs, depth_obs, count_steps):
    rgb_img = Image.fromarray(rgb_obs, mode="RGB")

    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")

    # depth_img = Image.fromarray((depth_obs * 255).astype(np.uint8), mode="L")

    # arr = [rgb_img, semantic_img, depth_img]

    # titles = ['rgb', 'semantic', 'depth']
    # plt.figure(figsize=(12 ,8))
    # for i, data in enumerate(arr):
    #     ax = plt.subplot(1, 3, i+1)
    #     ax.axis('off')
    #     ax.set_title(titles[i])
    #     plt.imshow(data)
    # plt.show()

    rgb_img = cv2.cvtColor(np.asarray(rgb_img),cv2.COLOR_RGB2BGR)
    sem_img = cv2.cvtColor(np.asarray(semantic_img),cv2.COLOR_RGB2BGR)

    # fn = 'result_target/Vis-rgb-{}.png'.format(count_steps)
    # cv2.imwrite(fn, rgb_img)
    # fn = 'result_target/Vis-sem-{}.png'.format(count_steps)
    # cv2.imwrite(fn, sem_img)

    cv2.imshow("RGB", rgb_img)
    # cv2.imshow("depth_img", depth_obs)
    cv2.imshow("Sematic", sem_img)

    
def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def example(run_type: str):

    # config=habitat.get_config("envs/habitat/configs/tasks/objectnav_gibson.yaml")
    config=habitat.get_config("envs/habitat/configs/tasks/objectnav_hm3d.yaml")

    config.defrost()
    config.DATASET.SPLIT = run_type
    # config.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
    config.freeze()

    env = habitat.Env(
        config=config
    )
    env.seed(10000)

    agg_metrics: Dict = defaultdict(float)

    observations = env.reset()
    
    num_episodes = 20
    count_episodes = 0
    scene_id = env.current_episode.scene_id
    old_scene_id = ''
    category = env.current_episode.goals[0].object_category
    old_category = ''
    start_height=0
    while count_episodes < num_episodes:

        observations = env.reset()
        scene_id = env.current_episode.scene_id
        category = env.current_episode.goals[0].object_category


        while scene_id == old_scene_id or category == old_category:
            observations = env.reset()
            scene_id = env.current_episode.scene_id
            category = env.current_episode.goals[0].object_category


        print("Environment ", count_episodes+1, " creation successful! Total: ", num_episodes)

        print("scene_id", scene_id)
        old_scene_id = scene_id

        print("Destination goal: ", category)
        old_category = category

        cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

        print("Agent stepping around inside environment.")

        # print("Agent Position: ", env.get_sim_location())


        count_steps = 0
        while not env.episode_over:
            keystroke = cv2.waitKey(0)

            if keystroke == ord(FORWARD_KEY):
                action = HabitatSimActions.MOVE_FORWARD
                print("action: FORWARD")
            elif keystroke == ord(LEFT_KEY):
                action = HabitatSimActions.TURN_LEFT
                print("action: LEFT")
            elif keystroke == ord(RIGHT_KEY):
                action = HabitatSimActions.TURN_RIGHT
                print("action: RIGHT")
            elif keystroke == ord(FINISH):
                action = HabitatSimActions.STOP
                print("action: LOOK_UP")
            elif keystroke == ord(UP_KEY):
                action = HabitatSimActions.LOOK_UP
                print("action: LOOK_DOWN")
            elif keystroke == ord(DOWN_KEY):
                action = HabitatSimActions.LOOK_DOWN
                print("action: FINISH")
            else:
                print("INVALID KEY")
                continue

            observations = env.step(action)
            count_steps += 1
            
            cv2.imshow("RGB", transform_rgb_bgr(observations["rgb"]))

        print("Episode finished after {} steps.".format(count_steps))

        if (
            action == HabitatSimActions.STOP and 
            env.get_metrics()["spl"]
        ):
            print("you successfully navigated to destination point")
        else:
            print("your navigation was not successful")

        metrics = env.get_metrics()


        for m, v in metrics.items():
            agg_metrics[m] += v
        count_episodes += 1

    avg_metrics = {k: v / count_episodes for k, v in agg_metrics.items()}

    for k, v in avg_metrics.items():
        print("{}: {:.3f}".format(k, v))

def inference(image, depth):

    image = torch.from_numpy(image).to(device).unsqueeze_(0).float()
    depth = torch.from_numpy(depth).to(device).unsqueeze_(0).float()

    # print(depth.shape) # torch.Size([1, 480, 640, 1])
            
    # print(image.shape) # torch.Size([1, 480, 640, 3])

    pred = model(image, depth).squeeze().cpu().detach().numpy()

    return pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "val"],
        required=False,
        default="val",
        help="run type of the experiment (train or eval)",
    )

    args = parser.parse_args()

    example(**vars(args))