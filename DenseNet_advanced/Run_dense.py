import argparse
from DenseNet import DenseNet
from data_providers.utils import get_data_provider_by_name

# train parameters
train_params_spectrum = {
    'n_epochs': 3000,
    'batch_size': 64,
    'initial_learning_rate': 1e-4,
    'reduce_lr_epoch_1': 3000,  # epochs * 0.5
    'reduce_lr_epoch_2': 3000,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every epoch
}

model_params_spectrum = {
    'growth_rate': 12,
    'depth': 44,
    'total_blocks': 5,
    'keep_prob': 0.8,
    'weight_decay': 1e-4,
    'momentum': 0.9,
    'should_save_logs': True,
    'should_save_model': True,
    'renew_logs': True,
    'reduction': 1,
    'run_from_checkpoint': False,
}

if __name__ == '__main__':
    # model parameters
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pretrained model exists'
             'If provided together with`--train` flag, testing will be '
             'performed right after training')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC'],
        default='DenseNet',
        help='What type of model to use')
    parser.add_argument(
        '--dataset', '-ds', type=str,
        default='Spectrum',
        help='What dataset should be used')

    args = parser.parse_args()
    args.keep_prob = 0.8

    if args.model_type == 'DenseNet':
        args.bc_mode = False
        args.reduction = 1
    elif args.model_type == 'DenseNet-BC':
        args.bc_mode = True

    model_params = vars(args)
    model_params = {**model_params_spectrum, **model_params}
    if not args.train and not args.test:
        print('You should train or test your network,Please check Params')
        exit()

    # some default params dataset/architecture related
    train_params = train_params_spectrum
    print('Params:')
    for k, v in model_params.items():
        print('\t%s: %s' % (k, v))
    print('Train params:')
    for k, v in model_params.items():
        print('\t%s: %s' % (k, v))

    print("Prepare training data...")
    data_provider = get_data_provider_by_name(args.dataset, train_params)

    print('Initialize the model...')
    model = DenseNet(data_provider=data_provider, **model_params)
    if args.train:
        print('Data provider train images:', data_provider.train.num_examples)
        model.train_all_epochs(train_params)
    if args.test:
        if not args.train:
            model.load_model()
        print("Data provider test images", data_provider.train.num_examples)
        print("Testing...")
        loss, accuracy = model.test(data_provider.test, batch_size=64)
        print("mean cross_entropy:%f, mean_accuracy: %f" % (loss, accuracy))















