import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import PurePath


def read_version(path):
    '''
    Read the version number of the log
    '''
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    ver = data['version'] if 'version' in data else 'none'

    return ver

def read_log_v2(test_path, target_dataset=None, max_reproj_px=None):
    '''
    Parse the test log file
    '''
    checkpoints = []

    with open(test_path, 'r') as file:
        data = file.read()
        data = data.replace('<<< ', '\n---\n')       # duct tape for splitting an yaml with duplicate keys into parts

        for part in yaml.load_all(data, Loader=yaml.Loader):
            if 'Test scores' not in part or 'Starting testing' not in part:
                continue
            params = part['Starting testing']
            scores = part['Test scores']
            num_imgs = int(params['Test size'])

            if target_dataset is not None:
                dataset = PurePath(params['Images dir']).parts[-2]
                if target_dataset != dataset:
                    print ('Log dataset does not match the target dataset. The log will be skipped!')
                    continue

            reproj_px = float(scores['Reprojection px'])
            if max_reproj_px is not None and reproj_px > max_reproj_px:
                continue

            entry = {'epoch': params['Model file'].split('/')[-1],
                     'reproj_px': reproj_px,
                     'reproj_rmse': float(scores['Reprojection RMSE']),
                     'segm_ce': float(scores['Segmentation CE']),
                     'rec_mse': float(scores['Reconstruction MSE']),
                     'imgs_per_sec': num_imgs / float(scores['Elapsed msec']) * 1000}
            checkpoints.append(entry)

    return checkpoints

def parse_model_dir(model_dir, target_dataset, max_reproj_px=None):
    '''
    Parse the log file of a given model
    '''
    conf_path = os.path.join(model_dir, 'conf.yaml')
    test_path = os.path.join(model_dir, 'test_scores.txt')
    if not os.path.isfile(conf_path):
        print ('Directory does not contain conf.yaml file and will be skipped!')
        return None
    if not os.path.isfile(test_path):
        print ('Directory does not contain test_scores.txt file and will be skipped!')
        return None

    # Read the version number of model log:
    ver = read_version(conf_path)

    # Parse log files and get scores:
    if ver == 'v2':
        checkpoints = read_log_v2(test_path, target_dataset, max_reproj_px)
    elif ver == 'v1':
        checkpoints = read_log_v2(test_path, target_dataset, max_reproj_px)
    else:
        print('Undefined version of the log file:', test_path)
        checkpoints = {}

    # Choose the checkpoint with the best reproj_px score:
    min_score = float('inf')
    min_score_index = None
    for i, entry in enumerate(checkpoints):
        reproj_px = entry['reproj_px']
        if reproj_px < min_score:
            min_score = reproj_px
            min_score_index = i

    if min_score_index is not None:
        return checkpoints[min_score_index]
    else:
        return None

def plot_chart(scores, ykey, xkey, ylabel='', xlabel='', legend=True):
    '''
    Plot a scores chart with the legend
    '''
    num_scores = len(scores)
    color_map = iter(cm.rainbow(np.linspace(0, 1, num_scores)))
    plot_name = xkey + '-vs-' + ykey + '.png'
    plt.figure(figsize=(16,6), num=plot_name)
    ax = plt.subplot(111)
    ax.grid(True)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    for i, score in enumerate(scores):
        number = i + 1
        name = score['name']
        x = score[xkey]
        y = score[ykey]
        color = next(color_map)
        ax.scatter(x, y, color=color, label='{} {}'.format(number,name))
        ax.text(x, y, str(number))

    # Plot a legend:
    if legend:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

def make_score_log(model_scores, score_keys, dst_path=None):
    log = []

    for key in score_keys:
        data = []
        for i, model in enumerate(model_scores):
            score = model[key]
            text = '{:.6f} : ({}) {} ({})'.format(score, i+1, model['name'], model['epoch'])
            data.append((score,text))
        data.sort(key=lambda pair: pair[0])
        log.append('>>>{}:'.format(key))
        for line in data:
            log.append(line[1])
        log.append('')

    if dst_path is not None:
        with open(dst_path, 'w') as f:
            for l in log:
                f.write("%s\n" % l)

    return log


def vizualize_metrics(src_dir, dst_dir=None, target_dataset=None, max_reproj_px=None, show=True):
    '''
    This function reads the log file of each model,
    selects the checkpoint with the highest test score,
    and builds a chart containing the best scores for all models.
    '''
    log = []

    # Get model dirs:
    model_names = os.listdir(src_dir)

    # Parse the log file of each model:
    model_scores = []
    for name in model_names:
        model_dir = os.path.join(src_dir, name)
        print ('Parsing {}...'.format(model_dir))
        score = parse_model_dir(model_dir, target_dataset, max_reproj_px)
        if score is not None:
            score['name'] = name
            model_scores.append(score)

    if dst_dir is not None:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

    # Plot a reproj_px/img_per_sec chart:
    xkey, xlabel = 'reproj_px', 'Reprojection RMSE (pixels)'
    ykey, ylabel = 'imgs_per_sec', 'imgs/sec'
    plot_chart(model_scores, ykey, xkey, ylabel, xlabel, legend=True)
    if dst_dir is not None:
        name = xkey + '-vs-' + ykey + '.png'
        dst_path = os.path.join(dst_dir, name)
        plt.savefig(dst_path)
        print ('Chart has been saved to {}'.format(dst_path))
    if show:
        plt.show()
    plt.close()

    # Plot a reproj_rmse/imgs_per_sec chart:
    xkey, xlabel = 'reproj_rmse', 'Reprojection RMSE'
    ykey, ylabel = 'imgs_per_sec', 'imgs/sec'
    plot_chart(model_scores, ykey, xkey, ylabel, xlabel, legend=True)
    if dst_dir is not None:
        name = xkey + '-vs-' + ykey + '.png'
        dst_path = os.path.join(dst_dir, name)
        plt.savefig(dst_path)
        print('Chart has been saved to {}'.format(dst_path))
    if show:
        plt.show()
    plt.close()

    # Plot a reproj_px/segm_ce chart:
    xkey, xlabel = 'reproj_px', 'Reprojection RMSE (pixels)'
    ykey, ylabel = 'segm_ce', 'Segmentation Cross-Entropy'
    plot_chart(model_scores, ykey, xkey, ylabel, xlabel, legend=True)
    if dst_dir is not None:
        name = xkey + '-vs-' + ykey + '.png'
        dst_path = os.path.join(dst_dir, name)
        plt.savefig(dst_path)
        print('Chart has been saved to {}'.format(dst_path))
    if show:
        plt.show()
    plt.close()

    # Plot a reproj_px/rec_mse chart:
    xkey, xlabel = 'reproj_px', 'Reprojection RMSE (pixels)'
    ykey, ylabel = 'rec_mse', 'Reconstruction MSE'
    plot_chart(model_scores, ykey, xkey, ylabel, xlabel, legend=True)
    if dst_dir is not None:
        name = xkey + '-vs-' + ykey + '.png'
        dst_path = os.path.join(dst_dir, name)
        plt.savefig(dst_path)
        print('Chart has been saved to {}'.format(dst_path))
    if show:
        plt.show()
    plt.close()

    # Save the log:
    if dst_dir is not None:
        log_path = os.path.join(dst_dir, 'scores.txt')
        score_keys = ['reproj_px', 'reproj_rmse', 'segm_ce', 'rec_mse', 'imgs_per_sec']
        make_score_log(model_scores, score_keys, log_path)
        print('Log has been saved to {}'.format(log_path))

    print ('All done!')


if __name__ == "__main__":
    src_dir = '/home/darkalert/builds/sports-field-homography/checkpoints/pitch/'
    target_dataset = 'sota-pitch-test'
    dst_dir = '/home/darkalert/builds/sports-field-homography/checkpoints/pitch/charts/'
    dst_dir = os.path.join(dst_dir, target_dataset)

    vizualize_metrics(src_dir, dst_dir, target_dataset, max_reproj_px=None, show=False)