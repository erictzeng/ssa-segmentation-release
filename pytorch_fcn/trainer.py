import logging
import pickle
from collections import deque
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pytorch_fcn.util import step_lr


class TaskTrainer:

    def __init__(self, output, net, tasks, *, lr=0.001, momentum=0.9, iterations=85000, step_lr=30000):
        self.output = Path(output)
        self.make_output_directories()

        self.net = net
        self.tasks = tasks

        self.iterations = iterations
        self.base_lr = lr
        self.step_lr = step_lr

        weight_decay_params = []
        no_weight_decay_params = []
        for module in net.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.BatchNorm2d)):
                if module.weight is not None:
                    weight_decay_params.append(module.weight)
                if module.bias is not None:
                    no_weight_decay_params.append(module.bias)
        print(len(list(net.parameters())), len(weight_decay_params) + len(no_weight_decay_params))
        self.opt = torch.optim.SGD(
            net.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=0.0001)
        self.logger = ValueLogger()

        self.display = 1
        self.snapshot_interval = 1000
        self.eval_interval = 500

    def make_output_directories(self):
        if not self.output.exists():
            self.output.mkdir()
        self.snapshot_dir = self.output / 'snapshot'
        if not self.snapshot_dir.exists():
            self.snapshot_dir.mkdir()

    def step(self):
        self.opt.zero_grad()
        for task in self.tasks:
            loss = task.step()
            loss.backward()
            self.logger.add(task.name, loss.item())
        self.opt.step()

    def run(self):
        for iteration in range(1, self.iterations + 1):
            losses = self.step()
            if iteration % self.display == 0:
                logging.info(f'Iteration {iteration}: {self.logger}')
            lr = self.base_lr * (1 - iteration / self.iterations) ** 0.9
            for param_group in self.opt.param_groups:
                param_group['lr'] = lr
            if iteration % self.snapshot_interval == 0:
                torch.save(
                    self.net.module.net.state_dict(),
                    str(self.snapshot_dir / f'net-iter{iteration:06d}.pth')
                )
                torch.save(
                    self.net.module.heads.state_dict(),
                    str(self.snapshot_dir / f'heads-iter{iteration:06d}.pth')
                )
            if iteration % self.eval_interval == 0:
                self.run_eval(iteration)

    def run_eval(self, iteration):
        path = self.output / 'results.pkl'
        if path.exists():
            with open(path, 'rb') as f:
                results = pickle.load(f)
        else:
            results = {}
        results[iteration] = {}
        for task in self.tasks:
            logging.info(f'  Evaluating {task.name}...')
            task_results = task.eval()
            results[iteration].update(task_results)
        tmp_path = path.with_suffix('.pkl.tmp')
        with open(tmp_path, 'wb') as f:
            pickle.dump(results, f)
        tmp_path.rename(path)
        logging.info(f'  Evaluation complete. Results updated at {path}.')


class ValueLogger:

    def __init__(self, average=10):
        self.values = OrderedDict()
        self.average = average

    def add(self, name, value):
        if name not in self.values:
            self.values[name] = deque(maxlen=self.average)
        self.values[name].append(value)

    def __str__(self):
        chunks = []
        for name, values in self.values.items():
            avg = np.mean(values)
            chunks.append(f'{name.rjust(15)}: {avg:8.04f}')
        return '  '.join(chunks)
