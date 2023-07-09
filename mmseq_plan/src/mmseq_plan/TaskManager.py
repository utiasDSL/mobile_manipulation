#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import logging

import numpy as np

import mmseq_plan.EEPlanner as eep
import mmseq_plan.BasePlanner as basep
from mmseq_utils import parsing

class SoTBase(ABC):
    def __init__(self, config):
        self.config = config

        self.planners = []
        for task_entry in config["tasks"]:
            if task_entry["name"][:2] == "EE":
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

class SoTCycle(SoTBase):
    def __init__(self, config):
        # in the class, we assume that tasks come in base and ee pairs with a base task preceding an ee task.
        # This implies that at even indices in self.planners are base tasks and at odd indices are ee task.
        self.curr_task_id = 1
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
            self.curr_task_id += 2
            self.curr_task_id %= self.planner_num
            self.logger.info("SoT current task is %d/%d", self.curr_task_id, self.planner_num)

        if self.shuffle_is_triggered:
            self.shuffle()


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



