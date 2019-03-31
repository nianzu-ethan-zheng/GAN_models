"""
Author : Nianzu Ethan Zheng
Datetime : 2018-2-2
Place: Shenyang China
Copyright
"""
from sequential.utils import check_dir
import os
import datetime

# config setting include two parts:
# First part is the aee.py method's parameters, which are used for initializing the method
# according to different occasion
# Second part is for model parameters, which are used for build the graph


class BaseConfig:
    def __init__(self):
        self.ndim_x = 15
        self.ndim_d = 8
        self.ndim_r = 7
        self.ndim_y = 2
        self.ndim_z = 2
        self.date = "2018-02-04"
        self.type = 'hybrid_GAN'
        self.mixer = 'lin'
        self.block_name = 'res'
        self.coeff_rest = 1
        self.coeff_lat = 1.5
        self.coeff_z = 0.01
        self.coeff_tv = 0.01
        self.drop = True
        self.distribution_sampler = 'sr'
        self.block_mode = 'bc'
        self.block_layers = 2
        self.distribution_z = 'gs'                   # determinstic or gaussian
        self.description = self.get_description()
        self.is_train = True
        self.optimizer_is_adam = True
        self.lr_schedule = "constant"
        self.momentum = 0.5
        self.batch_size = 64
        self.num_vad = 74
        self.device = '/gpu:0'
        self.ckpt_path = os.path.normpath('./check_point/{}'.format(self.get_description()))
        self.logs_path = os.path.normpath('./logs')
        self.assets_path = os.path.normpath('./assets/{}'.format(self.get_description()))
        self.latent_path = os.path.join(self.assets_path, 'latents_{}'.format(self.distribution_sampler))
        self.t_p = os.path.join(self.assets_path, 'true_and_predict')
        self.loss_path = os.path.join(self.assets_path, 'loss_vis')
        self.history_path = os.path.normpath('./historical_records')
        self.history_test_path = os.path.join(self.history_path, 'test_result')
        self.history_train_path = os.path.join(self.history_path, 'train_loss')
        self.history_result_path = os.path.join(self.history_path, 'result_diagram')

    def get_description(self):
        mixer_str = 'cc' if self.mixer is "concat" else self.mixer
        z_str = str(self.coeff_z) if self.coeff_z != 0 else 'no'
        img_str = str(self.coeff_rest) if self.coeff_rest != 0 else 'no'
        lat_str = str(self.coeff_lat) if self.coeff_lat != 0 else 'no'
        tv_str = str(self.coeff_tv) if self.coeff_tv != 0 else 'no'
        drop_str = 'd' if self.drop else 'x'
        block_mode = '' if self.block_mode is 'plain' else '-bc'
        block_str = block_mode+'' if self.block_layers == 2 else block_mode+'-{}'.format(self.block_layers)
        if self.date is None:
            time_stp = str(datetime.datetime.now().isoformat())
        else:
            time_stp = self.date
        return '{}_{}_{}-Dz={}-R={}-Lat={}-Tv={}-{}-{}{}-{}-{}'.format(
            self.type, mixer_str, self.block_name, z_str,
            img_str, lat_str, tv_str, drop_str, self.distribution_sampler,
            block_str, self.distribution_z, time_stp[:10]
        )

    def folder_init(self):
        """folder hierarchy:
        - check_point  # pre_trained model
        - logs # for the pickle file and logging file
        - assets # for the current train result
           - latents
        - historical_records  # for the lasting training file
           - test_result
           - train_loss
        """
        check_dir(self.ckpt_path, is_restart=False)
        check_dir(self.logs_path, is_restart=False)
        check_dir(self.assets_path, is_restart=False)
        check_dir(self.latent_path, is_restart=False)
        check_dir(self.t_p, is_restart=False)
        check_dir(self.loss_path, is_restart=False)
        check_dir(self.history_path, is_restart=False)
        check_dir(self.history_test_path, is_restart=False)
        check_dir(self.history_train_path, is_restart=False)


if __name__ == "__main__":
    config = BaseConfig()
    print(os.path.join(config.logs_path, config.description + '-train.pkl'))
    print(config.history_train_path)