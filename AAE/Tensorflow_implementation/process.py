import time, math


def get_process_bar(current_step, total_steps, num_segments=100):
    str = '['
    base = total_steps / float(num_segments)

    for seg in range(num_segments):
        if base * (seg + 1) < current_step:
            str += '='
        else:
            if str[-1] == '=':
                str += '>'
            else:
                str += '.'
    str = str[:num_segments] + ']'
    return str


def get_args_str(args):
    args_str = ''
    for key in sorted(args.keys()):
        if abs(args[key]) < 0.01:
            args_str += ' - {}: {:5.5E} - '.format(key, args[key])
        else:
            args_str += ' - {}: {:5.5f} - '.format(key, args[key])
    return args_str


def get_args_key_value(args):
    key_list = []
    value_list = []
    for key in sorted(args.keys()):
        key_list.append(key)
        value_list.append(args[key])
    return key_list, value_list


def get_args_format(key_list, value_list):
    str_head = ''
    num_list = len(key_list)
    for n in range(num_list):
        if n == num_list - 1:
            str_head += '{%s:^15s}'% n # give number
        else:
            str_head += '{%s:^15s} | '% n

    str_value = ''
    for m in range(num_list):
        if m == num_list - 1:
            str_value += '{%s:^15.5f}'% m
        else:
            if abs(value_list[m]) < 0.01:
                str_value += '{%s:^15.5E} | '% m
            else:
                str_value += '{%s:^15.5f} | '% m
    return str_head, str_value


def write_txt_log(current_step, args, file_dir = None):
    # mode a              ---> append value and key
    with open(file_dir + '/log.txt','a') as f:
        key_list, value_list = get_args_key_value(args)
        format_head, format_value = get_args_format(key_list, value_list)
        if current_step == 0:
            f.write(format_head.format(*key_list))
        else:
            f.write(format_value.format(*value_list))
            f.write('\n')

class Process():
    def __init__(self):
        self.start_time = 0
        self.epoch_start_time = 0

    def time_convert(self, seconds):
        return '{:02d}:{:02d}:{:02d}'.format(int(seconds / 3600), int(seconds % 3600 / 60), int(seconds % 60))

    def start_epoch(self, current_epoch, total_epoch):
        current = time.time()
        if self.start_time == 0:
            self.start_time == current
        self.epoch_start_time = current

    def get_total_elapsed_time(self):
        return self.time_convert(time.time() - self.start_time)

    def get_left_time(self, current_step, total_steps):
        elapsed_time = time.time() - self.epoch_start_time
        return self.time_convert(elapsed_time * (total_steps - current_step - 1))

    def show_table(self, current_step, total_steps, args):
        num_digits = int(math.log10(total_steps)) + 1
        print('\nIteration:{1:>{0}d}({2} in total)\t Time left:{3}'.format(
            num_digits, current_step, total_steps, self.get_left_time(current_step, total_steps)
        ))
        print('-' * 49)
        print('{0:^15s} | {1:^15s} | {2:^15s}'.format(
            'construction', 'discriminator', 'generator'
        ))
        print('-' * 49)
        print('{0:^15.5f} | {1:^15.5f} | {2:^15.5f}\n'.format(
            *args
        ))

    def show_table_2d(self, current_step, total_steps, args):
        num_digits = int(math.log10(total_steps)) + 1
        print('\nIteration:{1:>{0}d}({2} in total)\t Time left:{3}'.format(
            num_digits, current_step, total_steps, self.get_left_time(current_step, total_steps)
        ))

        key_list, value_list = get_args_key_value(args)
        format_head, format_value = get_args_format(key_list, value_list)
        num_ = len(key_list) * 17 + 2
        print('-' * num_)
        print(format_head.format(*key_list))
        print('-' * num_)

        print(format_value.format(*value_list))
        print('-' * num_)
        print('\n')

    def show_bar(self, current_step, total_steps, args):
        num_digits = int(math.log10(total_steps)) + 1
        process_bar = get_process_bar(current_step, total_steps)
        prefix = 'Epoch:{0:>{1}}/{2}{3}'.format(current_step, num_digits, total_steps, process_bar)
        args = get_args_str(args)
        if current_step == total_steps - 1:
            print('{}-{}--{}'.format(prefix, self.get_total_elapsed_time(), args))
        else:
            print('{}--{}--{}'.format(prefix, self.get_left_time(current_step, total_steps), args))


