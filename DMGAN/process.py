"""
Discription: A Python Progress Meter
Author: Nianzu Ethan Zheng
Date: 2018-1-21
Copyright
"""
import time
import math


class Process:
    """
    There are three method.
    log_process              --> Save arguments key and value into a text
    format_meter             --> Process bar with time and arguments summary
    display_current_results  --> Print detailed arguments with table style

    """
    def __init__(self):
        self.parent_start_time = 0
        self.child_start_time = 0
        self.start_epoch()

    def start_epoch(self):
        """
        Start a Session to track the time
        Use the current epoch to estimate the left time
        """
        current = time.time()
        if self.parent_start_time == 0:
            self.parent_start_time = current
        self.child_start_time = current

    def time_convert(self, seconds):
        minutes, s = divmod(int(seconds), 60)
        h, m = divmod(minutes, 60)
        if h:
            return '{:02d}:{:02d}:{:02d}'.format(h, m, s)
        else:
            return '{:02d}:{:02d}'.format(m, s)

    def elapsed_time(self):
        return self.time_convert(time.time() - self.parent_start_time)

    def left_time(self, current_step, total_steps):
        elapsed_time = time.time() - self.child_start_time
        return self.time_convert(elapsed_time * (total_steps - current_step - 1))

    def arg_parse(self, args):
        """
         transform dictionary argument into format and value
        """
        key_list = []
        value_list = []
        for key in sorted(args.keys()):
            key_list.append(key)
            value_list.append(args[key])

        num_list = len(key_list)
        key_format = (num_list - 1) * ' {:^15s} |' + '{:^15s}'
        value_format = ''
        for m in range(num_list):
            if m == num_list - 1:
                value_format += '{%s:^15.5f}' % m
            else:
                if abs(value_list[m]) < 0.001:
                    value_format += ' {%s:^15.2E} |' % m
                else:
                    value_format += ' {%s:^15.3f} |' % m
        return key_format, key_list, value_format, value_list

    def log_process(self, current_step, args, file_dir=None):
        with open(file_dir + '/process_log.txt', 'a') as file:  # "a" represent the append mode, won't override the old one
            key_format, key_list, value_format, value_list = self.arg_parse(args)
            if current_step == 0:
                file.write(key_format.format(*key_list))
            else:
                file.write(value_format.format(*value_list))
                file.write('\n')

    def display_current_results(self, current_step, total_steps, args):
        num_digits = int(math.log10(total_steps)) + 1

        print('\nIteration:{1:>{0}d}({2} in total)\t \t Time left:{3}'.format(
            num_digits, current_step, total_steps, self.left_time(current_step, total_steps)
        ))

        key_format, key_list, value_format, value_list = self.arg_parse(args)
        num_ = len(key_list) * 17 + 2
        print('-' * num_)
        print(key_format.format(*key_list))
        print('-' * num_)

        print(value_format.format(*value_list))
        print('-' * num_)
        print('\n')

    def format_meter(self, current_step, total_steps, args, num_segments=10):
        args_str = ''
        for key in sorted(args.keys()):
            if abs(args[key]) < 0.001:
                args_str += ', {:s}: {:5.2E}'.format(key, args[key])
            else:
                args_str += ', {:s}: {:5.3f}'.format(key, args[key])

        if current_step == total_steps - 1:
            total_steps = None

        frac = float(current_step) / total_steps
        percentage = '{:>5.1%}'.format(frac)
        bar_length = int(frac * num_segments)
        process_bar = "#" * bar_length + '-' * (num_segments - bar_length)
        elapsed_str = self.elapsed_time()
        left_str = self.left_time(current_step, total_steps)

        if total_steps:
            return print('{:s}|{:s}| {:d}/{:d} [elapsed: {:s}, left: {:s}, {:s}]'.format(
                percentage, process_bar, current_step, total_steps, elapsed_str, left_str, args_str))
        else:
            return print('{:d} [elapsed: {:s}, left: {:s}, {:s}]'.format(current_step, elapsed_str, left_str, args))


if __name__ == '__main__':
    process = Process()
    process.start_epoch()
    args = {'apple': 0.001, 'bananabababa': 0.000231, 'goodripe': 0.0023}
    process.display_current_results(12, 100, args)
    process.format_meter(12, 100, args)
    process.format_meter(25, 100, args)
    process.log_process(12, args, file_dir='./')

    with open('./process_log.txt', 'r') as f:
        print(f.read())

