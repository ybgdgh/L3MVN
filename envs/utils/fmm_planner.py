import cv2
import numpy as np
import skfmm
import skimage
from numpy import around, array, ma


def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2 \
               and ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 > \
                    (step_size - 1) ** 2:
                mask[i, j] = 1

    mask[size // 2, size // 2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2:
                mask[i, j] = max(5,
                                 (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                  ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner():
    def __init__(self, traversible, scale=1, step_size=5):
        self.scale = scale
        self.step_size = step_size
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.))
        self.fmm_dist = None
        self.around = np.zeros(traversible.shape)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9, 9))
        self.traversible_around = cv2.dilate((1-self.traversible*1).astype('uint8'), kernel)
        traversible_around_ma = ma.masked_values(self.traversible_around*1, 0)
        traversible_around_ma[self.traversible==0] = 0
        if np.any(traversible_around_ma == 0):
            dd = skfmm.distance(traversible_around_ma, dx=1)
            dd = (np.max(dd) - dd)
            dd = ma.filled(dd, 0)
            self.around = dd


    def set_goal(self, goal, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.)), \
            int(goal[1] / (self.scale * 1.))

        if self.traversible[goal_x, goal_y] == 0. and auto_improve:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        return

    def set_multi_goal(self, goal_map):
        traversible_ma = ma.masked_values(self.traversible * 1, 0) #mask掉障碍物层(0),只剩空白区域
        traversible_ma[goal_map == 1] = 0 # 除去障碍物层的地图，目标为0，其他为1
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1) # 将mask的值替换为max(dd)+1, 即障碍物为最大距离
        self.fmm_dist = dd
        self.around[goal_map == 1] = 0
        return

    def get_short_term_goal(self, state):
        scale = self.scale * 1.
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = get_mask(dx, dy, scale, self.step_size)
        dist_mask = get_dist(dx, dy, scale, self.step_size)

        state = [int(x) for x in state]
        self.fmm_dist += self.around
        dist = np.pad(self.fmm_dist, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset = dist[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]
        
        dist_around = np.pad(self.around, self.du,
                      'constant', constant_values=self.fmm_dist.shape[0] ** 2)
        subset_around = dist_around[state[0]:state[0] + 2 * self.du + 1,
                      state[1]:state[1] + 2 * self.du + 1]

        assert subset.shape[0] == 2 * self.du + 1 and \
            subset.shape[1] == 2 * self.du + 1, \
            "Planning error: unexpected subset shape {}".format(subset.shape)

        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2

        if subset[self.du, self.du] < 0.25 * 100 / 5.:  # 25cm
            stop = True
        else:
            stop = False

        subset -= subset[self.du, self.du]
        ratio1 = subset / dist_mask
        subset[ratio1 < -1.5] = 1

        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)

        # remove the imfluence of the around map
        if subset[stg_x, stg_y]+subset_around[self.du, self.du] > -0.0001:
            replan = True
        else:
            replan = False

        return (stg_x + state[0] - self.du) * scale, \
               (stg_y + state[1] - self.du) * scale, replan, stop

    def _find_nearest_goal(self, goal):
        traversible = skimage.morphology.binary_dilation(
            np.zeros(self.traversible.shape),
            skimage.morphology.disk(2)) != True
        traversible = traversible * 1.
        planner = FMMPlanner(traversible)
        planner.set_goal(goal)

        mask = self.traversible

        dist_map = planner.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(dist_map.argmin(), dist_map.shape)

        return goal
