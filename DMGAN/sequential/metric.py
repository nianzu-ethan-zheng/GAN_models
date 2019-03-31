"""
Author: Nianzu Ethan Zheng
Datetime: 2018-1-31
Place: Shenyang China
Copyright
"""
import numpy as np


def parzen_estimation(mu, sigma, mode='gauss'):
    """
    Implementation of a parzen-window estimation
    
    Keyword arguments:
        x: A "nxd"-dimentional numpy array, which each sample is
                  stored in a separate row (=training example)
        mu: point x for density estimation, "dx1"-dimensional numpy array
        sigma: window width
        
    Return the density estimate p(x)
    """

    def log_mean_exp(a):
        max_ = a.max(axis=1)
        return max_ + np.log(np.exp(a - np.expand_dims(max_, axis=-1) + 1e-200).mean(1))

    def gaussian_window(x, mu, sigma):
        a = (np.expand_dims(x, axis=1) - np.expand_dims(mu, axis=0)) / sigma
        b = np.sum(- 0.5 * (a ** 2), axis=-1)
        E = log_mean_exp(b)
        Z = mu.shape[1] * np.log(sigma * np.sqrt(np.pi * 2) + 1e-200)
        return np.exp(E - Z)

    return lambda x: gaussian_window(x, mu, sigma)


def get_ll(x, parzen, batch_size=0):
    """
    Get the likelihood of the input sample x,  not put all the sample into the
                parzen for the sake of computionalresource
                
    Keyword arguments:
        x : A nxp dimensional
        parzen: A window estimation function
        batch_size: A singular
        
    Return A singular, which represent the likelihood
    """
    inds = np.arange(x.shape[0])
    n_batches = int(np.ceil(len(inds) / batch_size))
    nlls = []
    for i in range(n_batches):
        nll = np.log(parzen(x[inds[i::n_batches]]) + 1e-200)
        nlls.extend(nll)
    return np.mean(np.array(nlls))


def cross_validate_sigma(samples, data, sigmas, batch_size):
    """
    Get the best sigma,i.e., the window width, by cross validation on the
    validation set
    
    Keyword arguments:
        samples: A "nxp"- like numpy array, the training set
        data: A "mxp" -like numpy array, the validation set
        sigmas: Sigma candidates
        batch_size: Batch processing
        
    Return the best sigma, and the likelihoods corresponding to the sigmas
    """
    lls = []
    for sigma in sigmas:
        parzen = parzen_estimation(samples, sigma)
        tmp = get_ll(data, parzen, batch_size=batch_size)
        lls.append(tmp)
    ind = np.argmax(lls)
    return sigmas[ind], np.array(lls)


def bayesian_infer(x_vec, kdes):
    """
    Classifies an input into class w_j determined by
    maximizing the class conditional probability for p(x|w_j)

    Keyword arguments:
        x_vec: A dx1 dimensional numpy array representing the sample
        kdes: List of the gaussian_kde estimates

    Return a tuple p(w_j|x)
    """
    p = []
    for kde in kdes:
        p.append(kde(x_vec))
    p = np.array(p)
    vals = p/sum(p)
    return vals


def entropy(vals):
    """
    Compute the entropy of conditional probability p(C|x)

    Keyword arguments:
        p_vals: A cx1 array-like sequence

    Return a singular , the entropy
    """
    return -sum([var * np.log(var + 1e-200) for var in vals])


def mi(train_set, test_set, prior=(0.5, 0.5), sigma=0.2233, batch_size=500):
    """
    Obtain Mutual information
    Keyword arguments:
        train_set: A "nxp"- like numpy array
        test_set: A "mxp"- like numpy array,
        prior: Class probability
        sigma: Window width

    Return a tupe(distance, Mutual information, A series of instances of Mutual information)
    """
    inds = np.arange(test_set.shape[0])
    n_batches = int(np.ceil(len(inds) / batch_size))
    class1_kde = parzen_estimation(train_set, sigma=sigma)
    mis = []
    arg_hs = []
    Hs = []
    for i in range(n_batches):
        class2_kde = parzen_estimation(test_set[inds[i::n_batches]], sigma=sigma)
        Hcx = []
        for ins in test_set:
            p_vals = bayesian_infer(ins.reshape(1, -1), [class1_kde, class2_kde])
            h = entropy(p_vals)
            Hcx.append(h)
        arg_hcx = np.mean(Hcx)
        Hc = entropy(prior)
        H = Hc - Hcx
        arg_h = Hc - arg_hcx
        d = arg_h / Hc
        mis.append(d)
        arg_hs.append(arg_h)
        Hs.append(H)
    return np.mean(np.array(mis)), np.std(np.array(mis)), np.mean(np.array(arg_hs)), np.array(Hs).reshape(-1, 1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.io as scio
    from sequential.nn_options import NetConfig
    opt = NetConfig()
    save_path = opt.assets_path
    # Load data
    data_path = '../data_providers/data/inds.mat'
    all_samples = np.array(scio.loadmat(data_path)['EG'])

    # Split data set
    train_set = all_samples[:279]
    test_set = all_samples[279:]

    # Obtain the sigma
    sigmas = np.linspace(1e-4, 2, 100)
    sigma, vals = cross_validate_sigma(train_set, test_set, sigmas, batch_size=59)
    # Visualize the likelihoods
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.plot(sigmas, vals, '-*r')
    ax.set(xlabel='$\sigma$', ylabel='Likelihood')
    yd, yu = ax.get_ylim()
    ax.text(1.5, yd * 0.8 + yu * 0.2, "$best \sigma = {:0.4f}$".format(sigma))
    fig.savefig(save_path + '/Industry_window_width.png')
    plt.close()

    # Obtain Mutual information
    class1_kde = parzen_estimation(train_set, sigma=0.2233)
    class2_kde = parzen_estimation(test_set, sigma=0.2233)
    class2_kde(test_set[1].reshape(1, -1))

    Hcx = []
    for ins in test_set:
        p_vals = bayesian_infer(ins.reshape(1, -1), [class1_kde, class2_kde])
        h = entropy(p_vals)
        Hcx.append(h)
    arg_hcx = np.mean(Hcx)

    Hc = entropy([0.5, 0.5])
    arg_h = Hc - arg_hcx
    H = Hc - Hcx
    # Visualize the instance Mutual information
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    ax.plot(H, '-*b')
    ax.set(xlabel='Sample', ylabel='Mutual information')
    fig.savefig(save_path + '/Industry_Instance_MI.png')
    plt.close()

    sns.set(style='dark')
    sns.distplot(H, rug=True, color='r')
    plt.savefig(save_path + '/Hist_MI.png')
    plt.close()

    # Test the distance
    # Make data
    np.random.seed(2017)
    mu_vecs = {}
    for i, j in zip(range(1, 4), [[0, 0], [1, 0], [1.2, 0]]):
        mu_vecs[i] = np.array(j).reshape(2, 1)

    cov_mats = {}
    for i in range(1, 4):
        cov_mats[i] = np.eye(2)

    all_samples = {}
    for i in range(1, 4):
        # generating 40x2 dimensional arrays with random Gaussian-distributed samples
        class_samples = np.random.multivariate_normal(mu_vecs[i].ravel(), cov_mats[i], 40)
        all_samples[i] = class_samples

    d12, s12, M12, H12 = mi(all_samples[1], all_samples[2])
    d13, s13, M13, H13 = mi(all_samples[1], all_samples[3])
    print(d12,s12, d13, s13)

    #
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    ax1.plot(H12, '-*r')
    ax1.set(xlabel='Sample', ylabel='Mutual information', title='average value {:0.3f}'.format(M12))
    ax2.plot(H13, '-*b')
    ax2.set(xlabel='Sample', ylabel='Mutual information', title='average value {:0.3f}'.format(M13))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    for i, c in zip(range(1, 4), ['r', 'g', 'b']):
        ax3.plot(all_samples[i][:, 0], all_samples[i][:, 1], '*' + c)
    plt.show()