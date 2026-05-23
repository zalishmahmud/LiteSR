# profile the experiment with tensorboard

from __future__ import annotations

import os.path
import time
from datetime import datetime
from typing import List, Dict
import logging
import json
from litesr.solver_agent import code_manipulation
from torch.utils.tensorboard import SummaryWriter


class Profiler:
    def __init__(
            self,
            log_dir: str | None = None,
            pkl_dir: str | None = None,
            max_log_nums: int | None = None,
    ):
        """
        Args:
            log_dir     : folder path for tensorboard log files.
            pkl_dir     : save the results to a pkl file.
            max_log_nums: stop logging if exceeding max_log_nums.
        """
        logging.getLogger().setLevel(logging.INFO)
        self._log_dir = log_dir
        self._json_dir = os.path.join(log_dir, 'samples')
        os.makedirs(self._json_dir, exist_ok=True)
        self._max_log_nums = max_log_nums
        self._num_samples = 0
        self._cur_best_program_sample_order = None
        self._cur_best_program_score = -float('inf')
        self._cur_best_program_str = None
        self._evaluate_success_program_num = 0
        self._evaluate_failed_program_num = 0
        self._tot_sample_time = 0
        self._tot_evaluate_time = 0
        self._all_sampled_functions: Dict[int, code_manipulation.Function] = {}
        self._start_time: float = time.time()
        self._best_history: Dict[str, dict] = {}
        run_tag = datetime.fromtimestamp(self._start_time).strftime("%d_%m_%Y_%H_%M_%S")
        self._run_dir = os.path.join(log_dir, run_tag)
        os.makedirs(self._run_dir, exist_ok=True)

        if log_dir:
            self._writer = SummaryWriter(log_dir=log_dir)

        self._each_sample_best_program_score = []
        self._each_sample_evaluate_success_program_num = []
        self._each_sample_evaluate_failed_program_num = []
        self._each_sample_tot_sample_time = []
        self._each_sample_tot_evaluate_time = []

    def _write_tensorboard(self):
        if not self._log_dir:
            return

        self._writer.add_scalar(
            'Best Score of Function',
            self._cur_best_program_score,
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Legal/Illegal Function',
            {
                'legal function num': self._evaluate_success_program_num,
                'illegal function num': self._evaluate_failed_program_num
            },
            global_step=self._num_samples
        )
        self._writer.add_scalars(
            'Total Sample/Evaluate Time',
            {'sample time': self._tot_sample_time, 'evaluate time': self._tot_evaluate_time},
            global_step=self._num_samples
        )
        
        # Log the function_str
        if self._cur_best_program_str is not None:
            self._writer.add_text(
                'Best Function String',
                self._cur_best_program_str,
                global_step=self._num_samples
            )

    def _write_json(self, programs: code_manipulation.Function):
        sample_order = programs.global_sample_nums
        sample_order = sample_order if sample_order is not None else 0
        function_str = str(programs)
        score = programs.score
        content = {
            'sample_order': sample_order,
            'function': function_str,
            'score': score,
            'optimized_params': getattr(programs, 'optimized_params', None),
        }
        path = os.path.join(self._json_dir, f'samples_{sample_order}.json')
        with open(path, 'w') as json_file:
            json.dump(content, json_file)

    def _write_best_history(self, programs: code_manipulation.Function, score_improved: bool):
        """Append one sample entry nested under its epoch in best_history.json."""
        epoch = programs.epoch_num if programs.epoch_num is not None else 0
        sample_num = programs.global_sample_nums if programs.global_sample_nums is not None else 0
        score = programs.score

        epoch_key = str(epoch)
        if epoch_key not in self._best_history:
            self._best_history[epoch_key] = {
                "best_score":   None,
                "best_program": None,
                "samples":      {},
            }

        epoch_entry = self._best_history[epoch_key]
        epoch_entry["samples"][str(sample_num)] = {
            "score":             score,
            "score_improved":    score_improved,
            "elapsed_seconds":   round(time.time() - self._start_time, 3),
            "sample_time":       programs.sample_time,
            "evaluate_time":     programs.evaluate_time,
            "optimized_params":  getattr(programs, 'optimized_params', None),
        }

        # keep epoch-level best up to date
        if score is not None and (epoch_entry["best_score"] is None or score > epoch_entry["best_score"]):
            epoch_entry["best_score"]   = score
            epoch_entry["best_program"] = self._cur_best_program_str

        path = os.path.join(self._run_dir, 'best_history.json')
        with open(path, 'w') as f:
            json.dump(self._best_history, f, indent=2)

    def register_function(self, programs: code_manipulation.Function):
        if self._max_log_nums is not None and self._num_samples >= self._max_log_nums:
            return

        sample_orders: int = programs.global_sample_nums
        if sample_orders not in self._all_sampled_functions:
            self._num_samples += 1
            self._all_sampled_functions[sample_orders] = programs
            prev_best = self._cur_best_program_score
            self._record_and_verbose(sample_orders)
            score_improved = bool(self._cur_best_program_score > prev_best)
            self._write_tensorboard()
            self._write_json(programs)
            self._write_best_history(programs, score_improved)

    def _record_and_verbose(self, sample_orders: int):
        function = self._all_sampled_functions[sample_orders]
        function_str = str(function).strip('\n')
        sample_time = function.sample_time
        evaluate_time = function.evaluate_time
        score = function.score
        # log attributes of the function
        print(f'================= Evaluated Function =================')
        print(f'{function_str}')
        print(f'------------------------------------------------------')
        print(f'Score        : {str(score)}')
        print(f'Sample time  : {str(sample_time)}')
        print(f'Evaluate time: {str(evaluate_time)}')
        print(f'Sample orders: {str(sample_orders)}')
        print(f'======================================================\n\n')

        # update best function in curve
        if function.score is not None and score > self._cur_best_program_score:
            self._cur_best_program_score = score
            self._cur_best_program_sample_order = sample_orders
            self._cur_best_program_str = function_str

        # update statistics about function
        if score:
            self._evaluate_success_program_num += 1
        else:
            self._evaluate_failed_program_num += 1

        if sample_time:
            self._tot_sample_time += sample_time
        if evaluate_time:
            self._tot_evaluate_time += evaluate_time
