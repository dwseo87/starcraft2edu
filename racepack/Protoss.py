from pysc2.agents import base_agent
from pysc2.lib import actions, features, units
from .utils import QLearningTable
import pandas as pd
import numpy as np
import random
import time
import os
import math

DATA_FILE = 'rlagent_learning_data_protoss'

ACTION_DO_NOTHING = 'donothing'
ACTION_SELECT_PROBE = 'selectprobe'
ACTION_BUILD_PYLON = 'buildpylon'
ACTION_BUILD_GATEWAYS = 'buildgateways'
ACTION_SELECT_GATEWAYS = 'selectgateways'
ACTION_SELECT_NEXUS = 'selectnexus'
ACTION_TRAIN_PROBE = 'trainprobe'
ACTION_TRAIN_ZEALOT = 'trainzealot'
ACTION_SELECT_ARMY = 'selectarmy'
ACTION_ATTACK = 'attack'
ACTION_HARVEST_MINERAL = 'harvestmineral'

smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_SELECT_PROBE,
    ACTION_BUILD_PYLON,
    ACTION_SELECT_NEXUS,
    ACTION_SELECT_GATEWAYS,
    ACTION_BUILD_GATEWAYS,
    ACTION_TRAIN_ZEALOT,
    ACTION_SELECT_ARMY,
    ACTION_TRAIN_PROBE,
    ACTION_ATTACK,
    ACTION_HARVEST_MINERAL,
]

#for mm_x in range(0, 64):
#    for mm_y in range(0, 64):
#        smart_actions.append(ACTION_ATTACK + '_' + str(mm_x) + '_' + str(mm_y))

for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 16 == 0 and (mm_y + 1) % 16 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 8) + '_' + str(mm_y - 8))

for mm_x in range(0, 16):
    for mm_y in range(0, 16):
        if (mm_x + 1) % 4 == 0 and (mm_y + 1) % 4 == 0:
            smart_actions.append(ACTION_BUILD_PYLON + '_' + str(mm_x) + '_' + str(mm_y))
            smart_actions.append(ACTION_BUILD_GATEWAYS + '_' + str(mm_x) + '_' + str(mm_y))

KILL_UNIT_REWARD = 0.2
KILL_BUILDING_REWARD = 0.5
PRODUCE_UNIT_REWARD = 0.1
PRODUCE_BUILDING_REWARD = 0.1
DO_NOTHING_PANALTY = -0.1


class ProtossRLAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ProtossRLAgent, self).__init__()

        self.base_top_left = None
        self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))

        self.previous_killed_unit_score = 0
        self.previous_killed_building_score = 0
        self.previous_produce_score = 0

        self.previous_action = None
        self.previous_state = None

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')

    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    def getMeanLocation(self, unitList):
        sum_x = 0
        sum_y = 0
        for unit in unitList:
            sum_x += unit.x
            sum_y += unit.y
        mean_x = sum_x / len(unitList)
        mean_y = sum_y / len(unitList)

        return [mean_x, mean_y]

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def get_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def step(self, obs):
        super(ProtossRLAgent, self).step(obs)

        # time.sleep(0.5)

        if obs.last():
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

        if obs.first():
            player_y, player_x = (
                        obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        pylon_count = len(self.get_units_by_type(obs, units.Protoss.Pylon))

        gateway_count = len(self.get_units_by_type(obs, units.Protoss.Gateway))

        supply_limit = obs.observation.player.food_cap
        army_supply = obs.observation.player.food_used

        killed_unit_score = obs.observation.score_cumulative.killed_value_units
        killed_building_score = obs.observation.score_cumulative.killed_value_structures
        total_unit_score = obs.observation.score_cumulative.total_value_units
        total_building_score = obs.observation.score_cumulative.total_value_structures

        #        current_state = np.zeros(5000)
        #        current_state[0] = supply_depot_count
        #        current_state[1] = barracks_count
        #        current_state[2] = supply_limit
        #        current_state[3] = army_supply
        #
        #        hot_squares = np.zeros(4096)
        #        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
        #        for i in range(0, len(enemy_y)):
        #            y = int(enemy_y[i])
        #            x = int(enemy_x[i])
        #
        #            hot_squares[((y - 1) * 64) + (x - 1)] = 1
        #
        #        if not self.base_top_left:
        #            hot_squares = hot_squares[::-1]
        #
        #        for i in range(0, 4096):
        #            current_state[i + 4] = hot_squares[i]

        current_state = np.zeros(20)
        current_state[0] = pylon_count
        current_state[1] = gateway_count
        current_state[2] = supply_limit
        current_state[3] = army_supply

        hot_squares = np.zeros(16)
        enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 16))
            x = int(math.ceil((enemy_x[i] + 1) / 16))

            hot_squares[((y - 1) * 4) + (x - 1)] = 1

        if not self.base_top_left:
            hot_squares = hot_squares[::-1]

        for i in range(0, 16):
            current_state[i + 4] = hot_squares[i]

        if self.previous_action is not None:
            reward = 0

            if killed_unit_score > self.previous_killed_unit_score:
                reward += KILL_UNIT_REWARD

            if killed_building_score > self.previous_killed_building_score:
                reward += KILL_BUILDING_REWARD

            if total_unit_score > self.previous_unit_score:
                reward += PRODUCE_UNIT_REWARD

            # if total_building_score > self.previous_building_score:
            #     reward += PRODUCE_BUILDING_REWARD

            if killed_unit_score == self.previous_killed_unit_score and \
                killed_building_score == self.previous_killed_building_score and \
                total_unit_score == self.previous_unit_score and \
                total_building_score == self.previous_building_score:
                reward += DO_NOTHING_PANALTY

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, str(current_state))

        rl_action = self.qlearn.choose_action(str(current_state))
        smart_action = smart_actions[rl_action]

        self.previous_killed_unit_score = killed_unit_score
        self.previous_killed_building_score = killed_building_score
        self.previous_unit_score = total_unit_score
        self.previous_building_score = total_building_score
        self.previous_state = current_state
        self.previous_action = rl_action

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')
            x = int(x)
            y = int(y)

        if smart_action == ACTION_DO_NOTHING:
            return actions.FUNCTIONS.no_op()

        elif smart_action == ACTION_SELECT_PROBE:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                if self.can_do(obs, actions.FUNCTIONS.select_idle_worker.id):
                    return actions.FUNCTIONS.select_idle_worker("select")
                else:
                    probes = self.get_units_by_type(obs, units.Protoss.Probe)
                    if len(probes) > 0:
                        probe = random.choice(probes)
                        if probe.x >= 0 and probe.y >= 0:
                            return actions.FUNCTIONS.select_point("select", (probe.x,
                                                                             probe.y))

        elif smart_action == ACTION_HARVEST_MINERAL:
            if self.can_do(obs, actions.FUNCTIONS.Harvest_Gather_screen.id):
                minerals = self.get_units_by_type(obs, units.Neutral.MineralField)
                if len(minerals) > 0:
                    mineral = random.choice(minerals)
                    target = (mineral.x, mineral.y)
                    # mean_x, mean_y = self.getMeanLocation(mineral)
                    if target[0] > 0 and target[1] >0:
                        return actions.FUNCTIONS.Harvest_Gather_screen("now",target)

        elif smart_action == ACTION_BUILD_PYLON:
            if self.can_do(obs, actions.FUNCTIONS.Build_Pylon_screen.id) and \
                                                supply_limit - army_supply <= 2:
                pylon = self.get_completed_units_by_type(obs, units.Protoss.Pylon)
                nexus = self.get_completed_units_by_type(obs, units.Protoss.Nexus)
                if len(pylon)>0:
                    pivot = pylon
                elif len(nexus)>0:
                    pivot = nexus
                else:
                    pivot = []
                if len(pivot) > 0:
                    mean_x, mean_y = self.getMeanLocation(pivot)
                    target = self.transformDistance(int(mean_x), x, int(mean_y), y)
                    if target[0] >= 0 and target[1]>=0:
                        return actions.FUNCTIONS.Build_Pylon_screen("now", target)

        elif smart_action == ACTION_BUILD_GATEWAYS:
            if self.can_do(obs, actions.FUNCTIONS.Build_Gateway_screen.id):
                pylons = self.get_completed_units_by_type(obs, units.Protoss.Pylon)
                if len(pylons) > 0:
                    mean_x, mean_y = self.getMeanLocation(pylons)
                    target = self.transformDistance(int(mean_x), x, int(mean_y), y)
                    if target[0] >= 0 and target[1] >= 0:
                        return actions.FUNCTIONS.Build_Gateway_screen("now", target)

        elif smart_action == ACTION_SELECT_GATEWAYS:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                gateways = self.get_units_by_type(obs, units.Protoss.Gateway)
                if len(gateways) > 0:
                    gateway = random.choice(gateways)
                    if gateway.x >= 0 and gateway.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (gateway.x,
                                                                         gateway.y))

        elif smart_action == ACTION_SELECT_NEXUS:
            if self.can_do(obs, actions.FUNCTIONS.select_point.id):
                nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
                if len(nexus) > 0:
                    nexus = random.choice(nexus)
                    if nexus.x >= 0 and nexus.y >= 0:
                        return actions.FUNCTIONS.select_point("select", (nexus.x,
                                                                         nexus.y))

        elif smart_action == ACTION_TRAIN_ZEALOT:
            if self.can_do(obs, actions.FUNCTIONS.Train_Zealot_quick.id):
                return actions.FUNCTIONS.Train_Zealot_quick("queued")

        elif smart_action == ACTION_TRAIN_PROBE:
            if self.can_do(obs, actions.FUNCTIONS.Train_Probe_quick.id):
                return actions.FUNCTIONS.Train_Probe_quick("queued")

        elif smart_action == ACTION_SELECT_ARMY:
            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")

        elif smart_action == ACTION_ATTACK:
            # if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
            if not self.unit_type_is_selected(obs, units.Protoss.Probe) and self.can_do(obs,
                                                                                     actions.FUNCTIONS.Attack_minimap.id):
                return actions.FUNCTIONS.Attack_minimap("now", self.transformLocation(int(x), int(y)))

        return actions.FUNCTIONS.no_op()


class ProtossBasicAgent(base_agent.BaseAgent):
    def __init__(self):
        super(ProtossBasicAgent, self).__init__()

        self.base_top_left = None
        self.supply_depot_built = False
        self.barracks_built = False
        self.barracks_rallied = False
        self.army_rallied = False

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def getMeanLocation(self, unitList):
        sum_x = 0
        sum_y = 0
        for unit in unitList:
            sum_x += unit.x
            sum_y += unit.y
        mean_x = sum_x / len(unitList)
        mean_y = sum_y / len(unitList)

        return [mean_x, mean_y]

    def unit_type_is_selected(self, obs, unit_type):
        if (len(obs.observation.single_select) > 0 and
                obs.observation.single_select[0].unit_type == unit_type):
            return True

        if (len(obs.observation.multi_select) > 0 and
                obs.observation.multi_select[0].unit_type == unit_type):
            return True

        return False

    def get_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type]

    def can_do(self, obs, action):
        return action in obs.observation.available_actions

    def get_completed_units_by_type(self, obs, unit_type):
        return [unit for unit in obs.observation.feature_units
                if unit.unit_type == unit_type
                and unit.build_progress == 100
                and unit.alliance == features.PlayerRelative.SELF]

    def step(self, obs):
        super(ProtossBasicAgent, self).step(obs)

        time.sleep(0.5)

        if obs.first():
            self.base_top_left = None
            self.pylon_built = False
            self.gateway_built = False
            self.gateway_rallied = False
            self.army_rallied = False

            player_y, player_x = (
                    obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

        probes = self.get_units_by_type(obs, units.Protoss.Probe)
        supply = obs.observation.player.food_cap - obs.observation.player.food_used
        completed_pylons = self.get_completed_units_by_type(obs, units.Protoss.Pylon)
        gateways = self.get_units_by_type(obs, units.Protoss.Gateway)

        if not self.pylon_built:
            if self.unit_type_is_selected(obs, units.Protoss.Probe):
                if self.can_do(obs, actions.FUNCTIONS.Build_Pylon_screen.id):
                    nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
                    if len(nexus) > 0:
                        mean_x, mean_y = self.getMeanLocation(nexus)
                        target = self.transformLocation(int(mean_x), 0, int(mean_y), 20)
                        self.pylon_built = True

                        return actions.FUNCTIONS.Build_Pylon_screen("now", target)
            if self.can_do(obs, actions.FUNCTIONS.select_idle_worker.id):
                return actions.FUNCTIONS.select_idle_worker("select")
            elif len(probes) > 0:
                probe = random.choice(probes)
                return actions.FUNCTIONS.select_point("select", (probe.x,
                                                                 probe.y))

        elif (len(completed_pylons) > 0 and not self.gateway_built and
                obs.observation.player.minerals >= 150):
            if self.can_do(obs, actions.FUNCTIONS.Build_Gateway_screen.id):
                mean_x, mean_y = self.getMeanLocation(completed_pylons)
                target = self.transformLocation(int(mean_x), 10, int(mean_y), 0)
                self.gateway_built = True
                return actions.FUNCTIONS.Build_Gateway_screen("now", target)
            probes = self.get_units_by_type(obs, units.Protoss.Probe)

            if self.can_do(obs, actions.FUNCTIONS.select_idle_worker.id):
                return actions.FUNCTIONS.select_idle_worker("select")
            elif len(probes) > 0:
                probe = random.choice(probes)
                return actions.FUNCTIONS.select_point("select", (probe.x,
                                                                 probe.y))

        # elif len(probes) < 13 or len(probes) < obs.observation.player.food_cap* 0.5:
        #     nexus = self.get_units_by_type(obs, units.Protoss.Nexus)
        #     if self.can_do(obs, actions.FUNCTIONS.Train_Probe_quick.id):
        #         return actions.FUNCTIONS.Train_Probe_quick("queued")
        #     if len(nexus)>0:
        #         nex = random.choice(nexus)
        #         return actions.FUNCTIONS.select_point("select", (nex.x, nex.y))

        elif supply > 1:
            if self.can_do(obs, actions.FUNCTIONS.Train_Zealot_quick.id):
                return actions.FUNCTIONS.Train_Zealot_quick("queued")
            if len(gateways) > 0:
                gateway = random.choice(gateways)
                return actions.FUNCTIONS.select_point("select", (gateway.x, gateway.y))

        elif not self.army_rallied:
            if self.can_do(obs, actions.FUNCTIONS.Attack_minimap.id):
                self.army_rallied = True

                if self.base_top_left:
                    return actions.FUNCTIONS.Attack_minimap("now", [39, 45])
                else:
                    return actions.FUNCTIONS.Attack_minimap("now", [21, 24])

            if self.can_do(obs, actions.FUNCTIONS.select_army.id):
                return actions.FUNCTIONS.select_army("select")


        return actions.FUNCTIONS.no_op()