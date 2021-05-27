def tversky(y_true, y_pred, smooth=0.000001):
    # Define alpha and beta
    alpha = 0.3
    beta = 0.7
    # Calculate Tversky for each class
    axis = identify_axis(y_true.get_shape())
    tp = K.sum(y_true * y_pred, axis=axis)
    fn = K.sum(y_true * (1 - y_pred), axis=axis)
    fp = K.sum((1 - y_true) * y_pred, axis=axis)
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return tversky_class


def tversky_loss(y_true, y_pred):
    n = K.cast(K.shape(y_true)[-1], 'float32')
    tver = K.sum(tversky(y_true, y_pred, smooth=0.000001), axis=[-1])
    return n - tver


def tversky_crossentropy(y_truth, y_pred):
    # Obtain Tversky Loss
    tver = tversky_loss(y_truth, y_pred)
    # Obtain Crossentropy
    crossentropy = Kcategorical_crossentropy(y_truth, y_pred)
    crossentropy = K.mean(crossentropy)
    # Return sum
    return tver + crossentropy


def dsc(y_true, y_pred, smooth=0.00001):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / \
           (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient(y_true, y_pred, numlabels=4):
    dice = 0
    for index in range(numlabels):
        dice += dsc(y_true[:, :, :, index], y_pred[:, :, :, index])
    return dice / numlabels
