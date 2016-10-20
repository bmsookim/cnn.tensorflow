####################################################################
########## Configuration file for CIFAR-10 classification ##########
####################################################################

# dataset specification configuration
dataset = 'cifar10'

# image specification configuration
w, h = 32, 32
uw, uh = 36, 36
channels = 3

# training specification configuration
epochs = 200
batch_size = 128
dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
keep_prob = [1-0.1, 1-0.2, 1-0.3, 1-0.4, 1-0.5]
lr_decay = 5e-5

# model specification configuration
model = 'vggnet'
train = True    # of train is set to 'False', we skip the training process.
mode = 'train'  # if mode is set to 'test', we use the moving average.

# step specification configuration
display_iter = (batch_size*75)
step1 = (60*((50000/batch_size)+1))+1
step2 = (120*((50000/batch_size)+1))+1
step3 = (160*((50000/batch_size)+1))+1
