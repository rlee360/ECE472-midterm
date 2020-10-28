import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pickle

# set matplotlib fonts
# sauce: https://stackoverflow.com/a/27697390/2397327
# mpl.rcParams['mathtext.fontset'] = 'stix'
# mpl.rcParams['font.family'] = 'STIXGeneral'

# set fonts
font = {'family': 'cmr10', 'size': 18}
mpl.rc('font', **font)  # change the default font to Computer Modern Roman
mpl.rcParams['axes.unicode_minus'] = False  # because cmr10 does not have a Unicode minus sign
plt.rcParams.update({
    'text.usetex': True
})

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))

def plt_func(model_name, history):
    fig, (ax1, ax2) = plt.subplots(figsize=(8.5, 11), nrows=2, ncols=1)
    ax1.plot(history['accuracy'], label='Training Accuracy')
    ax1.plot(history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Ratio')
    ax1.set_ylim(0, 1)
    ax1.legend()

    ax2.plot(history['loss'], label='Training Loss')
    ax2.plot(history['val_loss'], label='Validation Loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.set_ylim(0, 2)
    ax2.legend()

    fig.suptitle(model_name, fontsize=24)
    fig.tight_layout()
    #fig.savefig('hw4-cifar' + str(num_classes) + '-accuracy-loss.pdf', format='pdf')

# these two are the latest ones for mobilev1 and mobilev2
# results = pickle.load(open('mobilev1-full-cifar10-20-10-28-0002.pkl', 'rb'))
results = pickle.load(open('mobilev2-full-cifar10-20-10-28-0011.pkl', 'rb'))

# dimensions: model, alpha, rho, [num_weights, accuracy]
accuracy_weights = np.zeros((2, 6, 6, 2))

model_dict = {'v1': 0, 'v2': 1}
alpha_dict = {2: 0, 1: 1, 0.9: 2, 0.8: 3, 0.75: 4, 0.5: 5}
rho_dict = {1: 0, 30/32: 1, 28/32: 2, 24/32: 3, 20/32: 4, 16/32: 5}

for model in results:
    # plt_func(f"CIFAR-MobileNet-{model['metadata']['model']}; "
    #          f"alpha={model['metadata']['params']['alpha']}; "
    #          f"rho={model['metadata']['params']['rho']}",
    #          model['history'])

    # store accuracy vs. number of weights
    # accuracy_weights.append([model['metadata']['num_trainable_params'], model['test_accuracy']['accuracy']])
    accuracy_weights[model_dict[model['metadata']['model']], alpha_dict[model['metadata']['params']['alpha']], rho_dict[model['metadata']['params']['rho']]] = np.array([model['metadata']['num_trainable_params'], model['test_accuracy']['accuracy']])

    # num epochs
    # print(len(model['history']['accuracy']))

    print(model['metadata']['model'], model['metadata']['params']['alpha'], model['metadata']['params']['rho'], np.around(model['test_accuracy']['accuracy'], 2))

# plot v1 alphas vs accuracies
weight_counts = np.array(accuracy_weights)[1, :, :, 0]
accuracies = np.array(accuracy_weights)[1, :, :, 1]
print(weight_counts)
print(accuracies)

plt.figure(figsize=(16, 10))
plt.plot(weight_counts, accuracies)
for i, line in enumerate(plt.gca().get_lines()):
    line.set_color(['#000000', '#333333', '#555555', '#888888', '#aaaaaa', '#cccccc'][i])
for weight in list(weight_counts.flatten()):
    plt.axvline(x=weight, color='black', dashes=[40, 100], linewidth=0.2)

alphas = alpha_dict.keys()
positions = np.flip(np.unique(weight_counts))
for alpha, position in zip(alphas, positions):
    if alpha == 0.5:
        plt.annotate('$\\alpha=$', (position*0.92, 0.592))
    plt.annotate(str(np.around(alpha, 2)), (position*1.01, 0.592))
# plt.annotate('2', (1.57e04, 0.59))
# plt.annotate('1', (1.57e04, 0.59))
# plt.annotate('0.9', (1.57e04, 0.59))
# plt.annotate('0.8', (1.57e04, 0.59))
# plt.annotate('0.75', (1.57e04, 0.59))
# plt.annotate('2', (1.57e04, 0.59))
# minx, maxx = weight_counts.min(), weight_counts.max()
# plt.xlim(minx, maxx)
# plt.xticks(ticks=[1.55e4, 3.17e4, 3.52e4, 4.37e4, 5.35e4, 1.97e5])
# plt.gca().set_xticklabels(['$\\alpha=0.5$', '$\\alpha=0.5$', '$\\alpha=0.5$', '$\\alpha=0.5$', '$\\alpha=0.5$', '$\\alpha=0.5$'])
plt.xscale('log')
plt.xticks([2e4, 3e4, 5e4, 1e5, 2e5])
plt.gca().xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
# plt.gca().set_xticklabels(['test', 'test', 'test', 'test', 'test', 'test'])
# plt.gca().get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
plt.ylabel('Test accuracy')
plt.xlabel('Number of trainable weights')
plt.legend(['$\\rho=1$', '$\\rho=30/32$', '$\\rho=28/32$', '$\\rho=24/32$', '$\\rho=20/32$', '$\\rho=16/32$'])
plt.title('CIFAR-10-MobileNet-v2 Accuracy vs. Trainable Weights')
plt.savefig('cifar10_v2_accuracy_vs_weights.eps')
plt.savefig('cifar10_v2_accuracy_vs_weights.pdf')
# plt.show()
