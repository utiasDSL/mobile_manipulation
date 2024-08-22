#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging

import numpy as np

import mmseq_plan.EEPlanner as eep
import mmseq_plan.BasePlanner as basep
from mmseq_utils import parsing
from mmseq_utils.math import wrap_pi_scalar
from mmseq_plan.CPCPlanner import CPCPlanner
from mmseq_plan.SequentialPlanner import SequentialPlanner

class SoTBase(ABC):
    def __init__(self, config):
        self.config = config
        self.started = False

        self.planners = []
        for task_entry in config["tasks"]:
            if task_entry["name"] == "whole_body":
                if task_entry["planner_type"] == "SequentialPlanner":
                    whole_body_planner = SequentialPlanner(config)
                else:
                    whole_body_planner = CPCPlanner(config)
                # generate plan

                # generate partial planners

            elif task_entry["name"][:2] == "EE":
                planner_class = getattr(eep, task_entry["planner_type"])
            else:
                planner_class = getattr(basep, task_entry["planner_type"])

            planner = planner_class(task_entry)
            self.planners.append(planner)

        self.planner_num = len(self.planners)

        self.logger = logging.getLogger("Planner")

    def activatePlanners(self):
        for planner in self.planners:
            planner.started = True

    @abstractmethod
    def getPlanners(self, num_planners=2):
        pass

    @abstractmethod
    def update(self, t, states):
        pass

class SoTStatic(SoTBase):
    def __init__(self, config):
        self.curr_task_id = 0
        super().__init__(config)

    def getPlanners(self, num_planners=2):
        # get the top #num_planners in the stack
        end_id = min(self.curr_task_id + num_planners, self.planner_num)
        start_id = max(0, end_id - num_planners)
        return self.planners[start_id:end_id]

    def update(self, t, states):
        Pee = states["EE"][0]
        Pb = states["base"][:2]
        # check if current task is finished
        planner = self.planners[self.curr_task_id]
        finished = False
        if planner.type == "EE":
            finished = planner.checkFinished(t, Pee)
        elif planner.type == "base":
            finished = planner.checkFinished(t, Pb)

        # current task is finished move on to next task, unless it's the last task in the stack
        if finished:
            if self.curr_task_id != self.planner_num - 1:
                self.curr_task_id = min(self.planner_num-1, self.curr_task_id + 1)
                self.logger.info("SoT finished %d/%d tasks.", self.curr_task_id, self.planner_num)
            else:
                self.logger.info("SoT finished %d/%d tasks.", self.curr_task_id+1, self.planner_num)

        return finished, 1

class SoTCycle(SoTBase):
    def __init__(self, config):
        # in the class, we assume that tasks come in base and ee pairs with a base task preceding an ee task.
        # This implies that at even indices in self.planners are base tasks and at odd indices are ee task.
        self.curr_task_id = 1
        self.task_increment = 2
        self.shuffle_is_triggered = False
        super().__init__(config)

    def getPlanners(self, num_planners=2):
        # get the top #num_planners in the CYCLIC stack
        indices = np.arange(self.curr_task_id, self.curr_task_id + num_planners)
        indices %= self.planner_num

        return [self.planners[i] for i in indices]

    def update(self, t, states):
        Pee = states["EE"][0]
        Pb = states["base"][:2]
        # check if current task is finished
        planner = self.planners[self.curr_task_id]
        finished = False
        if planner.type == "EE":
            finished = planner.checkFinished(t, Pee)
        elif planner.type == "base":
            finished = planner.checkFinished(t, Pb)

        # current task is finished move on to next task, unless it's the last task in the stack
        if finished:
            self.planners[self.curr_task_id].reset()
            self.curr_task_id += self.task_increment
            self.curr_task_id %= self.planner_num
            self.logger.info("SoT current task is %d/%d", self.curr_task_id, self.planner_num)

        if self.shuffle_is_triggered:
            self.shuffle()

        return finished, self.task_increment

    def shuffle(self):
        curr_ee_task = self.curr_task_id + (1 - self.curr_task_id % 2)
        curr_ee_task_idx = (curr_ee_task - 1) // 2

        seq_curr = np.arange(self.planner_num).reshape((self.planner_num//2, 2))
        if np.random.random() > 0.0:
            if curr_ee_task != self.planner_num - 1:
                seq_to_shuffle = np.vstack((seq_curr[:curr_ee_task_idx], seq_curr[curr_ee_task_idx+1:]))
            else:
                seq_to_shuffle = seq_curr[:curr_ee_task_idx]

            np.random.shuffle(seq_to_shuffle)
            if curr_ee_task != self.planner_num - 1:
                seq_new = np.vstack((seq_to_shuffle[:curr_ee_task_idx], seq_curr[curr_ee_task_idx], seq_to_shuffle[curr_ee_task_idx:]))
            else:
                seq_new = np.vstack((seq_to_shuffle, seq_curr[curr_ee_task_idx]))

            seq_new = seq_new.flatten()
        else:
            np.random.shuffle(seq_curr)
            seq_new = seq_curr.flatten()

        self.planners = [self.planners[i] for i in seq_new]
        self.shuffle_is_triggered = False

        planner_name_new = [self.planners[2*i + 1].name for i in range(self.planner_num//2)]
        self.logger.info("Curr Task: {} New Task Seq: {}".format(self.curr_task_id, planner_name_new))

class SoTSequentialTasks(SoTCycle):
    def __init__(self, config):
        self.target_num = len(config["initial_waypoints_index"])
        self.ee_target_pos_xy = config["ee_target_pos_xy"]
        self.base_target_pos = config["base_target_pos"]
        self.base_tracking_err_tol = config["base_tracking_err_tol"]
        self.base_cruise_speed = config["base_cruise_speed"]
        self.curr_waypoints_index = config["initial_waypoints_index"]
        self.curr_ee_target_pos_z = config["initial_ee_target_pos_z"]
        self.seat_pos = np.array(config["seat_pos"])

        ee_target_pos, base_target_pos = self._get_waypoints(self.curr_waypoints_index,
                                                             self.curr_ee_target_pos_z)
        task_list = self._get_task_config_list(ee_target_pos, base_target_pos, config["initial_base_pos"])
        config["tasks"] = task_list

        super().__init__(config)
        self.curr_task_id = 0

    def update_planner(self, human_pos, robot_states):
        base_pos = robot_states["base"][0][:2]
        if len(human_pos) != self.target_num:
            print("human_pos length isn't correct expected {} got {}".format(self.target_num, len(human_pos)))
            return

        new_waypoints_index = self._get_new_waypoints_index(human_pos)
        new_ee_target_pos_z = self._get_new_ee_target_pos_z(human_pos)
        new_ee_target_pos, new_base_target_pos = self._get_waypoints(new_waypoints_index, new_ee_target_pos_z)
        new_ee_target_pos = np.array(new_ee_target_pos)
        new_base_target_pos = np.array(new_base_target_pos)

        for i in range(self.target_num):
            prev_base_task_index = i*2
            ee_task_index = prev_base_task_index + 1
            if self.started:
                next_base_task_index = (prev_base_task_index + 2)%self.planner_num
            else:
                next_base_task_index = prev_base_task_index + 2

            if new_waypoints_index[i] != self.curr_waypoints_index[i]:
                # Future/Current task changed
                if ee_task_index >= self.curr_task_id:
                    self.curr_waypoints_index[i] = new_waypoints_index[i]
                    self.curr_ee_target_pos_z[i] = new_ee_target_pos_z[i]
                    self.planners[ee_task_index].target_pos = new_ee_target_pos[i]

                    # Corresponding base task hasn't been executed yet
                    if prev_base_task_index > self.curr_task_id+1:
                        self.planners[prev_base_task_index].target_pos = new_base_target_pos[i]
                        self.planners[prev_base_task_index].regeneratePlan()

                        if next_base_task_index < self.planner_num:
                            self.planners[next_base_task_index].initial_pos = new_base_target_pos[i]
                            self.planners[next_base_task_index].regeneratePlan()

                    elif prev_base_task_index == self.curr_task_id or prev_base_task_index == (self.curr_task_id + 1):
                        self.planners[prev_base_task_index].initial_pos = base_pos
                        self.planners[prev_base_task_index].target_pos = new_base_target_pos[i]
                        self.planners[prev_base_task_index].regeneratePlan()

                        if next_base_task_index < self.planner_num:
                            self.planners[next_base_task_index].initial_pos = new_base_target_pos[i]
                            self.planners[next_base_task_index].regeneratePlan()
                    elif prev_base_task_index == self.curr_task_id - 1:
                        if next_base_task_index < self.planner_num:
                            self.planners[next_base_task_index].initial_pos = base_pos
                            self.planners[next_base_task_index].regeneratePlan()
            else:
                if np.abs(new_ee_target_pos_z[i] - self.curr_ee_target_pos_z[i]) > 0.2:
                    if ee_task_index >= self.curr_task_id:
                        self.curr_ee_target_pos_z[i] = new_ee_target_pos_z[i]
                        self.planners[ee_task_index].target_pos = new_ee_target_pos[i]

    def _get_new_waypoints_index(self, human_pos):
        human_pos_xy = np.expand_dims(human_pos[:, :2], axis=1)
        seat_pos = np.expand_dims(self.seat_pos, axis=0)
        human_seat_dist = np.linalg.norm(human_pos_xy - seat_pos, axis=-1)
        new_waypoints_index = np.argmin(human_seat_dist, axis=1)

        return new_waypoints_index

    def _get_new_ee_target_pos_z(self, human_pos):
        # ee_pos_z = human_pos[:, 2] - 0.8
        # ee_pos_z = np.where(ee_pos_z < 0.8, 0.8, ee_pos_z)
        # ee_pos_z = np.where(ee_pos_z > 1.5, 1.5, ee_pos_z)
        ee_pos_z = np.ones((human_pos.shape[0])) * 0.8

        return ee_pos_z

    def _get_waypoints(self, wpt_index, ee_target_pos_z):
        if max(wpt_index) < len(self.ee_target_pos_xy):
            ee_target_xy = [self.ee_target_pos_xy[i] for i in wpt_index]
            ee_target_pos = np.hstack((ee_target_xy, np.expand_dims(ee_target_pos_z, axis=-1)))

            base_target_pos = [self.base_target_pos[i] for i in wpt_index]

            return ee_target_pos, base_target_pos
        else:
            raise ValueError("EE wpt index {} out of bounds".format(wpt_index))

    def _get_task_config_list(self, ee_target_pos, base_target_pos, initial_base_pos):
        task_config_list = []

        for i in range(self.target_num):
            base_config = basep.BasePosTrajectoryLine.getDefaultParams()
            base_config["name"] = "Base Pos " + str(i+1)

            if i == 0:
                base_config["initial_pos"] = initial_base_pos
            else:
                base_config["initial_pos"] = base_target_pos[i-1]

            base_config["target_pos"] = base_target_pos[i]
            base_config["tracking_err_tol"] = self.base_tracking_err_tol
            base_config["cruise_speed"] = self.base_cruise_speed

            ee_config = eep.EESimplePlanner.getDefaultParams()
            ee_config["frame_id"] = "base"
            ee_config["name"] = "EE Pos "+ str(i+1)
            ee_config["target_pos"] = ee_target_pos[i]
            ee_config["hold_period"] = 1.

            task_config_list.append(base_config)
            task_config_list.append(ee_config)

        return task_config_list

    def update(self, t, states):
        Pee = states["EE"][0]
        Pb = states["base"][:2]
        if not self.config.get("use_joy", False) or "joy" not in states.keys():

            # check if current task is finished
            planner = self.planners[self.curr_task_id]
            if planner.type == "EE":
                finished = planner.checkFinished(t, Pee)
            elif planner.type == "base":
                finished = planner.checkFinished(t, Pb)
            else:
                finished = False

            if finished:
                task_id_increment = 1
            else:
                task_id_increment = 0

        else:

            # joy is 1 when an EE task is finished
            planner = self.planners[self.curr_task_id]

            if states["joy"] == 1:
                finished = True
                if planner.type == "EE":
                    task_id_increment = 1
                elif planner.type == "base":
                    task_id_increment = 2
                else:
                    finished = False
                    task_id_increment = 0

            else:
                if planner.type == "base":
                    finished = planner.checkFinished(t, Pb)
                    if finished:
                        task_id_increment = 1
                    else:
                        task_id_increment = 0
                else:
                    finished = False
                    task_id_increment = 0

            print(task_id_increment)

        # current task is finished move on to next task, unless it's the last task in the stack
        if finished:
            for i in range(task_id_increment):
                self.planners[self.curr_task_id+i].reset()
            self.curr_task_id += task_id_increment
            self.curr_task_id %= self.planner_num
            for i in range(2):
                next_task_id = (self.curr_task_id + i) % self.planner_num
                if self.planners[next_task_id].type=="base":
                    self.planners[next_task_id].initial_pos = Pb[0][:2]
                    self.planners[next_task_id].regeneratePlan()
                    self.planners[next_task_id].start_time = 0

            self.logger.info("SoT current task is %d/%d", self.curr_task_id, self.planner_num)

        return finished, task_id_increment

class SoTSequentialTasksBaseline(SoTCycle):

    def __init__(self, config):
        super().__init__(config)
        self.curr_task_id = 0
        self.task_increment = 1

    def update(self, t, states):
        Pee = states["EE"][0]
        Pb = states["base"][:2]
        if not self.config.get("use_joy", False) or "joy" not in states.keys():

            # check if current task is finished
            planner = self.planners[self.curr_task_id]
            if planner.type == "EE":
                finished = planner.checkFinished(t, Pee)
            elif planner.type == "base":
                finished = planner.checkFinished(t, Pb)
            else:
                finished = False

            if finished:
                task_id_increment = 1
            else:
                task_id_increment = 0

        else:

            # joy is 1 when an EE task is finished
            planner = self.planners[self.curr_task_id]

            if planner.type == "base":
                finished = planner.checkFinished(t, Pb)
            else:
                if states["joy"] == 1:
                    finished = True
                else:
                    finished = False

            if finished:
                task_id_increment = 1
            else:
                task_id_increment = 0


            print(task_id_increment)

        # current task is finished move on to next task, unless it's the last task in the stack
        if finished:
            for i in range(task_id_increment):
                self.planners[self.curr_task_id+i].reset()
            self.curr_task_id += task_id_increment
            self.curr_task_id %= self.planner_num
            next_task_id = (self.curr_task_id + 1) % self.planner_num
            if self.planners[next_task_id].type =="base":
                Pb[0][2] = wrap_pi_scalar(Pb[0][2])
                self.planners[next_task_id].initial_pose = Pb[0]
                self.planners[next_task_id].regeneratePlan()
                self.planners[next_task_id].start_time = 0

            self.logger.info("SoT current task is %d/%d", self.curr_task_id, self.planner_num)

        return finished, task_id_increment



class SoTTimed(SoTBase):
    def __init__(self, config):
        self.curr_base_task_id = 1
        super().__init__(config)

        self.task_switching_time = self.config["task_switching_time"]
        assert(len(self.task_switching_time)+2 == len(self.planners))

    def getPlanners(self, num_planners=2):
        return [self.planners[0], self.planners[self.curr_base_task_id]]

    def update(self, t, states):
        if self.curr_base_task_id < self.planner_num - 1 and t > self.task_switching_time[self.curr_base_task_id-1]:
            self.curr_base_task_id += 1
            self.curr_base_task_id = min(self.curr_base_task_id, self.planner_num - 1)

            self.logger.info("SoT on base task %d.", self.curr_base_task_id)

        return True, 1

class SoTBottomTaskFixed(SoTBase):
    def __init__(self, config):
        super().__init__(config)
        self.fixed_task_id = self.planner_num - 1
        self.curr_task_id = 0


    def getPlanners(self, num_planners=2):
        return [self.planners[self.curr_task_id], self.planners[self.fixed_task_id]]

    def update(self, t, states):
        Pee = states["EE"][0]
        Pb = states["base"][:2]
        # check if current task is finished
        planner = self.planners[self.curr_task_id]
        finished = False
        if planner.type == "EE":
            finished = planner.checkFinished(t, Pee)
        elif planner.type == "base":
            finished = planner.checkFinished(t, Pb)

        if finished and self.curr_task_id < self.fixed_task_id - 1:
            self.curr_task_id += 1

            self.logger.info("SoT on base task %d.", self.curr_task_id)

        return finished, 1

if __name__ == "__main__":
    config_path = "$PROJECTMM3D_HOME/experiments/config/sim/simulation.yaml"
    config = parsing.load_config(config_path)

    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    planner_log = logging.getLogger("Planner")
    planner_log.setLevel(logging.INFO)
    planner_log.addHandler(ch)

    sot = SoTCycle(config["planner"])
    sot.curr_task_id = 5
    sot.shuffle()



