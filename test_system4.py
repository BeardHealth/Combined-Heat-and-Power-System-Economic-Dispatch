import numpy as np
import torch


class PowerOnlyUnits(object):
    def __init__(self, alpha, beta, gamma, lamda, rho, pmin, pmax, num):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lamda = lamda
        self.rho = rho
        self.min = pmin
        self.max = pmax
        self.num = num
        self.output_p = 0
        self.cost = 0

    def reset(self, p=None):
        if p:
            self.output_p = p
        else:
            self.output_p = self.min + np.random.uniform(low=0, high=1) * (self.max - self.min)
        self.cost = self.alpha * self.output_p * self.output_p + self.beta * self.output_p + self.gamma

    def _step(self, action):
        self.output_p = self.output_p + action * (self.max - self.min)
        if self.output_p > self.max:
            self.output_p = self.max
        if self.output_p < self.min:
            self.output_p = self.min
        self.cost = self.alpha * self.output_p * self.output_p + self.beta * self.output_p + self.gamma

    def set(self, delta_p):
        self.output_p += delta_p / self.num
        if self.output_p > self.max:
            self.output_p = self.max
        if self.output_p < self.min:
            self.output_p = self.min
        self.cost = self.alpha * self.output_p * self.output_p + self.beta * self.output_p + self.gamma

    def get(self):
        p = self.output_p * self.num
        cost = self.cost * self.num
        out = (p, cost)
        return out


class CHPUnits(object):
    def __init__(self, a, b, c, d, e, f, feasible_region, num):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.e = e
        self.f = f
        self.feasible_region = feasible_region
        self.num = num
        self.output_p = 0
        self.output_h = 0
        self.cost = 0
        self.max = 0
        self.min = 0

    def reset(self, P=None):
        # feasible region
        if len(self.feasible_region) == 4:
            p = []
            h = []
            for i in range(4):
                p.append(self.feasible_region[i][0])
                h.append(self.feasible_region[i][1])
            self.max = p[3]
            self.min = p[1]
            self.output_p = p[1] + np.random.uniform(low=0, high=1) * (p[0] - p[1])
            if self.output_p < p[1]:
                self.output_p = p[1]
                self.output_h = h[1]
            if p[1] <= self.output_p <= p[0]:
                k = (h[1] - h[2]) / (p[1] - p[2])  # bias
                b = h[1] - k * p[1]  # intercep
                self.output_h = self.output_p * k + b  # maximum in feasible range
            if p[0] < self.output_p <= p[2]:
                k = (h[1] - h[2]) / (p[1] - p[2])  # bias
                b = h[1] - k * p[1]  # intercept
                self.output_h = (k * self.output_p + b)
            if p[2] < self.output_p <= p[3]:
                k = (h[2] - h[3]) / (p[2] - p[3])
                b = h[2] - k * p[2]  # intercept
                self.output_h = (k * self.output_p + b)
            if p[3] < self.output_p:
                self.output_p = p[3]
                self.output_h = 0

        if len(self.feasible_region) == 5:
            p = []
            h = []
            for i in range(5):
                p.append(self.feasible_region[i][0])
                h.append(self.feasible_region[i][1])
            self.max = p[4]
            self.min = p[0]
            self.output_p = p[1] + np.random.uniform(low=0, high=1) * (p[2] - p[1])
            if self.output_p < p[0]:
                self.output_p = p[0]
                self.output_h = h[1]
            if p[1] <= self.output_p <= p[2]:
                k = (h[1] - h[2]) / (p[1] - p[2])  # bias
                b = h[1] - k * p[1]  # intercept
                self.output_h = self.output_p * k + b
            if p[3] < self.output_p <= p[4]:
                k = (h[3] - h[4]) / (p[3] - p[4])  # bias
                b = h[3] - k * p[3]  # intercept
                self.output_h = self.output_p * k + b
            if p[4] < self.output_p:
                self.output_p = p[4]
                self.output_h = h[4]

        if len(self.feasible_region) == 6:
            p = []
            h = []
            for i in range(6):
                p.append(self.feasible_region[i][0])
                h.append(self.feasible_region[i][1])
            self.max = p[5]
            self.min = p[2]
            self.output_p = p[2] + np.random.uniform(low=0, high=1) * (p[1] - p[2])
            if self.output_p < p[2]:
                self.output_p = p[2]
                self.output_h = h[2]
            if p[2] < self.output_p <= p[3]:
                k = (h[2] - h[3]) / (p[2] - p[3])  # bias
                b = h[3] - k * p[3]  # intercept
                self.output_h = self.output_p * k + b
            if p[3] < self.output_p <= p[4]:
                k = (h[3] - h[4]) / (p[3] - p[4])  # bias
                b = h[3] - k * p[3]  # intercept
                self.output_h = self.output_p * k + b
            if p[4] < self.output_p:
                self.output_p = p[4]
                self.output_h = h[4]

        self.cost = self.a * self.output_p * self.output_p + self.b * self.output_p + self.c \
                    + self.d * self.output_h * self.output_h + self.e * self.output_h + self.f * self.output_p * self.output_h

    def _step(self, action_p):
        if len(self.feasible_region) == 4:
            p = []
            h = []
            for i in range(4):
                p.append(self.feasible_region[i][0])
                h.append(self.feasible_region[i][1])
            self.output_p = self.output_p + action_p * (p[3] - p[1])
            if self.output_p < p[1]:
                self.output_p = p[1]
                self.output_h = h[1]
            if p[1] <= self.output_p <= p[0]:
                k = (h[1] - h[2]) / (p[1] - p[2])  # bias
                b = h[1] - k * p[1]  # intercep
                self.output_h = self.output_p * k + b  # maximum in feasible range
            if p[0] < self.output_p <= p[2]:
                k = (h[1] - h[2]) / (p[1] - p[2])  # bias
                b = h[1] - k * p[1]  # intercept
                self.output_h = (k * self.output_p + b)
            if p[2] < self.output_p <= p[3]:
                k = (h[2] - h[3]) / (p[2] - p[3])
                b = h[2] - k * p[2]  # intercept
                self.output_h = (k * self.output_p + b)
            if p[3] < self.output_p:
                self.output_p = p[3]
                self.output_h = 0

        if len(self.feasible_region) == 5:
            p = []
            h = []
            for i in range(5):
                p.append(self.feasible_region[i][0])
                h.append(self.feasible_region[i][1])
            self.output_p = self.output_p + action_p * (p[4] - p[1])
            if self.output_p < p[0]:
                self.output_p = p[0]
                self.output_h = h[1]
            if p[1] <= self.output_p <= p[2]:
                k = (h[1] - h[2]) / (p[1] - p[2])  # bias
                b = h[1] - k * p[1]  # intercept
                self.output_h = self.output_p * k + b
            if p[3] < self.output_p <= p[4]:
                k = (h[3] - h[4]) / (p[3] - p[4])  # bias
                b = h[3] - k * p[3]  # intercept
                self.output_h = self.output_p * k + b
            if p[4] < self.output_p:
                self.output_p = p[4]
                self.output_h = h[4]

        if len(self.feasible_region) == 6:
            p = []
            h = []
            for i in range(6):
                p.append(self.feasible_region[i][0])
                h.append(self.feasible_region[i][1])
            self.output_p = p[1] + action_p * (p[5] - p[2])
            if self.output_p < p[2]:
                self.output_p = p[2]
                self.output_h = h[2]
            if p[2] < self.output_p <= p[3]:
                k = (h[2] - h[3]) / (p[2] - p[3])  # bias
                b = h[3] - k * p[3]  # intercept
                self.output_h = self.output_p * k + b
            if p[3] < self.output_p <= p[4]:
                k = (h[3] - h[4]) / (p[3] - p[4])  # bias
                b = h[3] - k * p[3]  # intercept
                self.output_h = self.output_p * k + b
            if p[4] < self.output_p:
                self.output_p = p[4]
                self.output_h = h[4]

        self.cost = self.a * self.output_p * self.output_p + self.b * self.output_p + self.c \
                    + self.d * self.output_h * self.output_h + self.e * self.output_h + self.f * self.output_p * self.output_h

    def get(self):
        p = self.output_p * self.num
        h = self.output_h * self.num
        cost = self.cost * self.num
        return (p, h, cost)


class HeatOnlyUnits(object):
    def __init__(self, a, b, c, hmin, hmax, num):
        self.a = a
        self.b = b
        self.c = c
        self.min = hmin
        self.max = hmax
        self.num = num
        self.output_h = 0
        self.cost = 0

    def reset(self, h=None):
        if h:
            self.output_h = h
        else:
            self.output_h = self.min + np.random.uniform(0, 1) * (self.max - self.min)
        # self.output_h = self.min + np.random.uniform(low=0, high=1) * (self.max - self.min)
        self.cost = self.a * self.output_h * self.output_h + self.b * self.output_h + self.c

    def _step(self, action):
        self.output_h = self.output_h + action * (self.max - self.min)
        if self.output_h > self.max:
            self.output_h = self.max
        if self.output_h < self.min:
            self.output_h = self.min
        self.cost = self.a * self.output_h * self.output_h + self.b * self.output_h + self.c

    def set(self, delta_h):
        self.output_h += delta_h / self.num
        if self.output_h > self.max:
            self.output_h = self.max
        if self.output_h < self.min:
            self.output_h = self.min
        self.cost = self.a * self.output_h * self.output_h + self.b * self.output_h + self.c

    def get(self):
        h = self.output_h * self.num
        cost = self.cost * self.num
        return (h, cost)


class Env(object):
    action_dim = 9
    state_dim_p = 10
    state_dim_m = 14
    state_dim_h = 10

    def __init__(self):
        self.power_only = []
        self.CHP = []
        self.heat_only = []
        self.power_demand = 0
        self.heat_demand = 0

    def construct(self):
        # power only
        power_only1 = PowerOnlyUnits(0.00028, 8.1, 550, 300, 0.035, 0, 680, 1)
        self.power_only.append(power_only1)
        power_only2 = PowerOnlyUnits(0.00056, 8.1, 309, 200, 0.042, 0, 360, 2)
        self.power_only.append(power_only2)
        power_only3 = PowerOnlyUnits(0.00324, 7.74, 240, 150, 0.063, 60, 180, 6)
        self.power_only.append(power_only3)
        power_only4 = PowerOnlyUnits(0.00284, 8.6, 126, 100, 0.084, 55, 120, 4)
        self.power_only.append(power_only4)

        # CHP
        chp1 = CHPUnits(0.0345, 14.5, 2650, 0.03, 4.2, 0.031, [[98.8, 0], [81, 104.8], [215, 180], [247, 0]], 2)
        self.CHP.append(chp1)
        chp2 = CHPUnits(0.0435, 36, 1250, 0.027, 0.6, 0.011,
                        [[44, 0], [44, 15.9], [40, 75], [110.2, 135.6], [125.8, 32.4], [125.8, 0]], 2)
        self.CHP.append(chp2)
        chp3 = CHPUnits(0.1035, 34.5, 2650, 0.025, 2.203, 0.051, [[20, 0], [10, 40], [45, 55], [60, 0]], 1)
        self.CHP.append(chp3)
        chp4 = CHPUnits(0.072, 20, 1565, 0.02, 2.34, 0.04, [[35, 0], [35, 20], [90, 45], [90, 25], [105, 0]], 1)
        self.CHP.append(chp4)

        # heat_only
        heat_only1 = HeatOnlyUnits(0.038, 2.0109, 950, 0, 2695.20, 1)
        self.heat_only.append(heat_only1)
        heat_only2 = HeatOnlyUnits(0.038, 2.0109, 950, 0, 60, 2)
        self.heat_only.append(heat_only2)
        heat_only3 = HeatOnlyUnits(0.052, 3.0651, 480, 0, 120, 2)
        self.heat_only.append(heat_only3)

    def reset(self):
        # for p in self.power_only:
        #     p.reset()
        # for m in self.CHP:
        #     m.reset()
        # for h in self.heat_only:
        #     h.reset()

        #easy mode
        self.power_only[0].reset(500)
        self.power_only[1].reset(200)
        self.power_only[2].reset(80)
        self.power_only[3].reset(60)
        self.CHP[0].reset()
        self.CHP[1].reset()
        self.CHP[2].reset()
        self.CHP[3].reset()
        self.heat_only[0].reset(500)
        self.heat_only[1].reset(40)
        self.heat_only[2].reset(100)
        self.power_demand = 2350
        self.heat_demand = 1250
        self.balance()
        return self._get_obs()

    def balance(self):
        # p_positive = [0, 1, 2, 3]
        # p_negative = [3, 2, 1, 0]
        # h_positive = [0, 1, 2]
        # h_negative = [2, 1, 0]
        p_generate = self.power_only[0].get()[0] + self.power_only[1].get()[0] + self.power_only[3].get()[0] + \
                    self.power_only[2].get()[0] + \
                     self.CHP[0].get()[0] + self.CHP[1].get()[0] + self.CHP[2].get()[0] + self.CHP[3].get()[0]

        h_generate = self.heat_only[1].get()[0] + self.heat_only[2].get()[0] + self.heat_only[0].get()[0] + \
                    self.CHP[0].get()[1] + self.CHP[1].get()[1] + self.CHP[2].get()[1] + self.CHP[3].get()[1]
        dis_p = self.power_demand - p_generate
        dis_h = self.heat_demand - h_generate
        self.power_only[0].set(dis_p)
        self.heat_only[0].set(dis_h)

        # extra balance
        p_generate = self.power_only[0].get()[0] + self.power_only[1].get()[0] + self.power_only[3].get()[0] + \
                     self.power_only[2].get()[0] + \
                     self.CHP[0].get()[0] + self.CHP[1].get()[0] + self.CHP[2].get()[0] + self.CHP[3].get()[0]
        dis_p = self.power_demand - p_generate
        if dis_p > 50:
            self.power_only[1].set(dis_p)
        #
        # num_p = 0
        # num_h = 0
        # if dis_p > 0:
        #     while True:
        #         self.power_only[p_positive[num_p]].set(dis_p)
        #         p_generate = self.power_only[0].get()[0] + self.power_only[1].get()[0] + self.power_only[3].get()[0] + \
        #                      self.power_only[2].get()[0] + \
        #                      self.CHP[0].get()[0] + self.CHP[1].get()[0] + self.CHP[2].get()[0] + self.CHP[3].get()[0]
        #         dis_p = self.power_demand - p_generate
        #         if dis_p < 50:
        #             break
        #         num_p += 1
        # if dis_p < 0:
        #     while True:
        #         self.power_only[p_negative[num_p]].set(dis_p)
        #         p_generate = self.power_only[0].get()[0] + self.power_only[1].get()[0] + self.power_only[3].get()[0] + \
        #                      self.power_only[2].get()[0] + \
        #                      self.CHP[0].get()[0] + self.CHP[1].get()[0] + self.CHP[2].get()[0] + self.CHP[3].get()[0]
        #         dis_p = self.power_demand - p_generate
        #         if dis_p > -50:
        #             break
        #         num_p += 1
        # if dis_h > 0:
        #     while True:
        #         self.heat_only[h_positive[num_h]].set(dis_h)
        #         h_generate = self.heat_only[1].get()[0] + self.heat_only[2].get()[0] + self.heat_only[0].get()[0] + \
        #                      self.CHP[0].get()[1] + self.CHP[1].get()[1] + self.CHP[2].get()[1] + self.CHP[3].get()[1]
        #         dis_h = self.heat_demand - h_generate
        #         if dis_h < 50:
        #             break
        #         num_h += 1
        # if dis_h < 0:
        #     while True:
        #         self.heat_only[h_negative[num_h]].set(dis_h)
        #         h_generate = self.heat_only[1].get()[0] + self.heat_only[2].get()[0] + self.heat_only[0].get()[0] + \
        #                      self.CHP[0].get()[1] + self.CHP[1].get()[1] + self.CHP[2].get()[1] + self.CHP[3].get()[1]
        #         dis_h = self.heat_demand - h_generate
        #         if dis_h > -50:
        #             break
        #         num_h += 1

    def step(self, action):
        # power_only3 is the slack node of electric network
        # heat_only1 is the slack node of heat network

        # before_cost
        p_cost_before = self.power_only[0].get()[1] + self.power_only[1].get()[1] + self.power_only[3].get()[1] + \
                        self.power_only[2].get()[1]
        h_cost_before = self.heat_only[1].get()[1] + self.heat_only[2].get()[1] + self.heat_only[0].get()[1]
        chp_cost_before = self.CHP[0].get()[2] + self.CHP[1].get()[2] + self.CHP[2].get()[2] + self.CHP[3].get()[2]
        # p_generate_before = self.power_only[0].get()[0] + self.power_only[1].get()[0] + self.power_only[3].get()[0] + \
        #                    self.power_only[2].get()[0] + \
        #                     self.CHP[0].get()[0] + self.CHP[1].get()[0] + self.CHP[2].get()[0] + self.CHP[3].get()[0]
        #
        # h_generate_before = self.heat_only[1].get()[0] + self.heat_only[2].get()[0] + self.heat_only[0].get()[0] + \
        #                     self.CHP[0].get()[1] + self.CHP[1].get()[1] + self.CHP[2].get()[1] + self.CHP[3].get()[1]
        # dis_p = (self.power_demand - p_generate_before) if (self.power_demand - p_generate_before) >= 0 else 0
        # dis_q = (self.heat_demand - h_generate_before) if (self.heat_demand - h_generate_before) >= 0 else 0
        # punish_cost_before = 0.1035 * dis_p * dis_p + 34.5 * dis_p + 0.052 * dis_q * dis_q + 4.2 * dis_q + 0.05*dis_p*dis_q

        # take action
        self.power_only[1]._step(action[0]*0.002)
        self.power_only[2]._step(action[1]*0.002)
        self.power_only[3]._step(action[2]*0.002)
        self.CHP[0]._step(action[3]*0.002)
        self.CHP[1]._step(action[4]*0.002)
        self.CHP[2]._step(action[5]*0.002)
        self.CHP[3]._step(action[6]*0.002)
        self.heat_only[1]._step(action[7]*0.002)
        self.heat_only[2]._step(action[8]*0.002)
        self.balance()
        # after_cost
        p_generate_after = self.power_only[0].get()[0] + self.power_only[1].get()[0] + self.power_only[3].get()[0] + \
                           self.power_only[2].get()[0] + \
                            self.CHP[0].get()[0] + self.CHP[1].get()[0] + self.CHP[2].get()[0] + self.CHP[3].get()[0]

        h_generate_after = self.heat_only[1].get()[0] + self.heat_only[2].get()[0] + self.heat_only[0].get()[0] + \
                            self.CHP[0].get()[1] + self.CHP[1].get()[1] + self.CHP[2].get()[1] + self.CHP[3].get()[1]

        chp_cost_after = self.CHP[0].get()[2] + self.CHP[1].get()[2] + self.CHP[2].get()[2] + self.CHP[3].get()[2]
        p_cost_after = self.power_only[0].get()[1] + self.power_only[1].get()[1] + self.power_only[3].get()[1] + \
                       self.power_only[2].get()[1]
        h_cost_after = self.heat_only[1].get()[1] + self.heat_only[2].get()[1] + self.heat_only[0].get()[1]
        # dis_p = (self.power_demand - p_generate_after) if (self.power_demand - p_generate_after) >= 0 else 0
        # dis_q = (self.heat_demand - h_generate_after) if (self.heat_demand - h_generate_after) >= 0 else 0
        # punish_cost_after = 0.1035 * dis_p * dis_p + 34.5 * dis_p + 0.052 * dis_q * dis_q + 4.2 * dis_q + 0.05*dis_p*dis_q

        # reward = - p_cost_after - h_cost_after - chp_cost_after
        reward = (p_cost_before - p_cost_after) + (h_cost_before - h_cost_after) + (
                    chp_cost_before - chp_cost_after)
        done = 1 if (p_cost_after + h_cost_after + chp_cost_after) < 58000 else 0

        return self._get_obs(), reward/100, done

    def _get_obs(self):
        p_cost = self.power_only[0].get()[1] + self.power_only[1].get()[1] + self.power_only[3].get()[1] + \
                       self.power_only[2].get()[1]
        h_cost = self.heat_only[1].get()[1] + self.heat_only[2].get()[1] + self.heat_only[0].get()[1]
        chp_cost = self.CHP[0].get()[2] + self.CHP[1].get()[2] + self.CHP[2].get()[2] + self.CHP[3].get()[2]
        p_generate = self.power_only[0].get()[0] + self.power_only[1].get()[0] + self.power_only[3].get()[0] + self.power_only[2].get()[0] + \
                     self.CHP[0].get()[0] + self.CHP[1].get()[0] + self.CHP[2].get()[0] + self.CHP[3].get()[0]
        h_generate = self.heat_only[1].get()[0] + self.heat_only[2].get()[0] + self.heat_only[0].get()[0] + \
                     self.CHP[0].get()[1] + self.CHP[1].get()[1] + self.CHP[2].get()[1] + self.CHP[3].get()[1]
        # dis_p = (self.power_demand - p_generate) if (self.power_demand - p_generate) >= 0 else 0
        # dis_q = (self.heat_demand - h_generate) if (self.heat_demand - h_generate) >= 0 else 0
        # punish_cost = 0.1035 * dis_p * dis_p + 34.5 * dis_p + 0.052 * dis_q * dis_q + 4.2 * dis_q + 0.05*dis_p*dis_q
        #
        # distance_p = (self.power_demand - p_generate)
        # distance_q = (self.heat_demand - h_generate)

        p_only_state = np.zeros((4, 10))
        for i, p in enumerate(self.power_only):
            p_only_state[i][0] = p.alpha * 1000
            p_only_state[i][1] = p.beta * 0.1
            p_only_state[i][2] = p.gamma / 1000
            p_only_state[i][3] = p.get()[0] / self.power_demand
            p_only_state[i][4] = p.get()[1] / (p_cost + chp_cost + h_cost)
            p_only_state[i][5] = p.get()[1] / p_cost
            p_only_state[i][6] = (p.output_p-p.min) / (p.max - p.min)
            p_only_state[i][7] = p_cost/(p_cost + chp_cost + h_cost)
            p_only_state[i][8] = chp_cost / (p_cost + chp_cost + h_cost)
            p_only_state[i][9] = h_cost / (p_cost + chp_cost + h_cost)

        chp_state = np.zeros((4, 14))
        for i, chp in enumerate(self.CHP):
            chp_state[i][0] = chp.a * 10
            chp_state[i][1] = chp.b / 100
            chp_state[i][2] = chp.c / 10000
            chp_state[i][3] = chp.d * 10
            chp_state[i][4] = chp.e / 10
            chp_state[i][5] = chp.f * 10
            chp_state[i][6] = chp.get()[0] / self.power_demand
            chp_state[i][7] = chp.get()[1] / self.heat_demand
            chp_state[i][8] = chp.get()[2] / (p_cost + chp_cost + h_cost)
            chp_state[i][9] = chp.get()[2] / chp_cost
            chp_state[i][10] = (chp.output_p - chp.min) / (chp.max - chp.min)
            chp_state[i][11] = p_cost/(p_cost + chp_cost + h_cost)
            chp_state[i][12] = chp_cost / (p_cost + chp_cost + h_cost)
            chp_state[i][13] = h_cost / (p_cost + chp_cost + h_cost)

        h_only_state = np.zeros((3, 10))
        for i, h in enumerate(self.heat_only):
            h_only_state[i][0] = h.a * 10
            h_only_state[i][1] = h.b / 10
            h_only_state[i][2] = h.c / 1000
            h_only_state[i][3] = h.get()[0] / self.heat_demand
            h_only_state[i][4] = h.get()[1] / (p_cost + chp_cost + h_cost)
            h_only_state[i][5] = h.get()[1] / h_cost
            h_only_state[i][6] = (h.output_h - h.min) / (h.max - h.min)
            h_only_state[i][7] = p_cost / (p_cost + chp_cost + h_cost)
            h_only_state[i][8] = chp_cost / (p_cost + chp_cost + h_cost)
            h_only_state[i][9] = h_cost / (p_cost + chp_cost + h_cost)



        return (p_only_state, chp_state, h_only_state, p_cost, h_cost, chp_cost, p_generate, h_generate)


if __name__ == '__main__':
    env = Env()
    env.construct()
    s = env.reset()
    print(s)
