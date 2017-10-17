
def constant(initial_learning_rate=0.0001):
    return initial_learning_rate


def ladder(current_epoch, initial_learning_rate=0.0001):
    if current_epoch < 50:
        learning_rate = initial_learning_rate
    elif current_epoch < 1000:
        learning_rate = initial_learning_rate * 0.1
    else:
        learning_rate = initial_learning_rate * 0.1 ** 2
    return learning_rate


def exponential_decay(current_epoch, initial_learning_rate=0.002, decay_rate=0.1):
    if current_epoch < 500:
        learning_rate = initial_learning_rate * decay_rate ** (current_epoch / 500)
    elif current_epoch < 1000:
        learning_rate = 0.0002
    else:
        learning_rate = 0.0002
    return learning_rate
