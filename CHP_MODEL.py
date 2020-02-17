"""
Environment for Intergrated Energy System.
You can customize this script in a way you want.

Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""
import numpy as np


class CHPEnv(object):
    GB_bound = [0.2, 1]
    GT_bound = [0, 1]
    TST_bound = [- 0.1, 0.2]
    Grid_bound = [-1, 1]
    action_bound = [-0.2, 0.2]
    action_dim = 4
    state_dim = 14
    dt = .1  # refresh rate
    get_point = False
    grab_counter = 0

    def __init__(self):
        self.device_info = np.zeros((4, 4))
        self.device_info[0, 0] = 5000
        self.device_info[1, 0] = 5000
        self.device_info[2, 0] = 5000
        self.device_info[3, 0] = 2000
        self.he_q = 0
        self.cost = 0
        self.realcost = 0
        self.total_p = 0
        self.wind = 0
        self.rtp = 0
        self.point_info = np.squeeze(np.array([3500, 8500]))
        self.point_l = 0.075
        self.center = np.squeeze(np.array([2500, 8500]))
        self.point_info_init = self.point_info.copy()
        self.tank_info = 2000

    def step(self, action):
        # action = (node1 angular v, node2 angular v)
        gt_action = np.clip(action[0], *self.action_bound)
        gb_action = np.clip(action[1], *self.action_bound)
        tst_action = np.clip(action[2], *self.action_bound)
        grid_action = np.clip(action[3], *self.action_bound)

        self.device_info[0, 1] += gt_action * self.dt
        self.device_info[0, 1] = np.clip(self.device_info[0, 1], 0.2, 1)
        self.device_info[1, 1] += gb_action * self.dt
        self.device_info[1, 1] = np.clip(self.device_info[1, 1], 0, 1)
        self.device_info[2, 1] += tst_action * self.dt
        self.device_info[2, 1] = np.clip(self.device_info[2, 1], (self.tank_info - 500)/5000, (self.tank_info + 1000)/5000)
        self.device_info[2, 1] = np.clip(self.device_info[2, 1], 0, 1)
        self.device_info[3, 1] += grid_action * self.dt
        self.device_info[3, 1] = np.clip(self.device_info[3, 1], -1, 1)

        gt_p = self.device_info[0, 1]*self.device_info[0, 0]
        self.device_info[0, 2] = gt_p
        gt_q = gt_p*2.3*0.75
        self.device_info[0, 3] = gt_q
        gb_q = self.device_info[1, 1]*self.device_info[1, 0]
        self.device_info[1, 3] = gb_q
        tst_q = np.clip((self.device_info[2, 1]*self.device_info[2, 0] - self.tank_info), -500, 1000)
        self.device_info[2, 3] = tst_q

        # gb_q = self.point_info[1] - gt_q + tst_q
        # self.device_info[1, 1] = np.clip(gb_q / 5000, 0, 1)
        # gb_q = self.device_info[1, 1] * self.device_info[3, 0]
        # buy_p = self.point_info[0] - gt_p - self.wind
        # self.device_info[3, 1] = np.clip(buy_p/2000, -1, 1)
        buy_p = self.device_info[3, 1]*self.device_info[3, 0]
        self.device_info[3, 2] = buy_p

        self.he_q = (gt_q + gb_q - tst_q)  # 换热站总热量
        self.total_p = gt_p + buy_p + self.wind  # 总产生电量

        self.cost = (0.345 * gt_q * 2.3*0.75/(1+2.3*0.75) + 0.345 * gb_q/0.9)/self.point_info[1] +\
                    (buy_p * self.rtp + 0.345 * gt_q * 1/(1+2.3*0.75))/self.point_info[0]

        self.realcost = 0.345 * gt_q * 2.3*0.75 + 0.345 * gb_q/0.9 + buy_p * self.rtp

        s, p_distance, q_distance = self._get_state()
        r = self._r_func(p_distance, q_distance, tst_q)

        return s, r, self.get_point

    def reset(self):
        self.get_point = False
        self.grab_counter = 0
        # price = np.array([0.427, 0.427, 0.427, 0.427, 0.427, 0.427, 0.527, 0.527, 0.627, 0.627, 0.627,
        #                 0.527, 0.527, 0.527, 0.527, 0.527, 0.527, 0.627, 0.627, 0.627, 0.627, 0.627,
        #                 0.427, 0.427])
        # wind_power = np.array([875, 1234, 1390, 1392, 1336, 1223, 1173, 1136, 1158, 1312, 1369,
        #                       1376, 1315, 1301, 1343, 1310, 1208, 1055, 896, 773, 672, 626,
        #                       624, 703])

        # temperature = 2*np.array([4800, 4896, 4953.6, 4992, 4896, 4800, 4560, 4320, 4128,
        #                           3984, 3888, 3801.6, 3758.4, 3744, 3748.8, 3772.8, 3820.8,
        #                           3888, 4032, 4224, 4416, 4608, 4752, 4800])
#
        # e_load = np.array([2178.0, 2009, 1873, 1755, 1704, 1839, 2517, 4211, 5397, 5375,
        #                    5651, 5481, 5227, 5176, 5143, 5227, 5909, 6417, 6545, 6206,
        #                   5698, 4510, 4025, 2093])

        p = np.random.uniform(1500, 7000)
        q = np.random.uniform(5000, 11000)
        t = np.random.uniform(200, 2000)
        rtp = np.random.uniform(0.4, 0.7)
        tank = np.random.uniform(1000, 3000)
        self.point_info[0] = p
        self.point_info[1] = q
        self.wind = t
        self.rtp = rtp
        self.tank_info = tank
        return self._get_state()[0]

    def set(self):
        self.get_point = False
        self.grab_counter = 0
        self.point_info[0] = 2000
        self.point_info[1] = 9000
        self.wind = 700
        self.rtp = 0.627
        self.tank_info = 1000
        return self._get_state()[0]

    def render(self):
        s = self._get_state()
        print(s, self.realcost)

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def _get_state(self):
        # return the distance (dx, dy) between arm finger point with blue point
        he_q = self.he_q
        total_p = self.total_p
        t_p = total_p - self.point_info[0]
        t_q = he_q - self.point_info[1]
        # dis = self.point_info[1]/self.point_info[0]
        # ratio = self.device_info[0, 1] - self.device_info[1, 1]
        p_gt = self.device_info[0, 2]/self.point_info[0]
        q_gt = self.device_info[0, 3]/self.point_info[1]
        q_gb = self.device_info[1, 3]/self.point_info[1]
        q_tst = self.device_info[2, 3]/self.point_info[1]
        p_grid = self.device_info[3, 2]/self.point_info[0]
        p_wind = self.wind/self.point_info[0]
        cen_dis_p = (self.center[0] - self.point_info[0])/5000
        cen_dis_q = (self.center[1] - self.point_info[1])/5000
        in_point = 1 if self.grab_counter > 0 else 0
        return np.hstack([in_point, t_p/5000,  t_q/5000, cen_dis_p, cen_dis_q, p_gt, q_gt,
                          q_gb, q_tst, p_grid, p_wind, self.device_info[2, 1], self.tank_info/5000, self.rtp
                          # arm1_distance_p, arm1_distance_b,
                          ]), t_p/5000, t_q/5000

    def _r_func(self, p_distance, q_distance, tst_q):
        t = 30
        abs_distance = np.sqrt(np.square(p_distance)+np.square(q_distance))
        r = - abs_distance - self.cost - 0.1*np.abs(self.device_info[2, 1] - 0.4)
        # print(0.1*np.abs(self.device_info[2, 1] - 0.4))
        # print(-self.cost)
        # print(tst_q*(self.rtp-self.rtp_m)/1000)
        if abs_distance < self.point_l and (not self.get_point):
            r += 1
            self.grab_counter += 1
            if self.grab_counter > t:
                r += 10.
                self.get_point = True
        elif abs_distance > self.point_l:
            self.grab_counter = 0
            self.get_point = False
        return r


if __name__ == '__main__':
    env = CHPEnv()
    action = env.sample_action()
    print(action)
    env.reset()
    print(env.tank_info)
    s, r, env.get_point = env.step(action)
    print(env.device_info)
