from .spectra import OilDataProvider


def get_data_provider_by_name(name, train_params):
    """Return required data provider class"""
    if name == 'Spectrum':
        return OilDataProvider(**train_params)
    else:
        print('Sorry ,data provider for %s dataset'
              'was not implemented yet' % name)
        exit()
