from options import Options
from dataloader import load_data
from utils import load_model
from evaluate import evaluate
import visualize
import torch
import numpy as np
from collections import OrderedDict
import time
import os

##
def main():
    """ Training
    run python main.py --dataset toumingzhi --model myganomaly --load_final_weights --batchsize 30
    """
    opt = Options().parse()
    dataloader = load_data(opt)
    model = load_model(opt, dataloader)
    print(model.device)
    model.load_weights()
    model.netd.eval()
    model.netg.eval()
    epoch = 0
    print(f">> Evaluating {model.name} on {model.opt.dataset} to detect {model.opt.abnormal_class}")
    threshold = 0.016
    # the code runs much slower in the first time, so run twice to measure its real speed
    for i in range(2):
        time_start = time.time()
        with torch.no_grad():
            # Create big error tensor for the test set.
            an_scores = torch.zeros(size=(len(model.dataloader.valid.dataset),), dtype=torch.float32, device=model.device)
            gt_labels = torch.zeros(size=(len(model.dataloader.valid.dataset),), dtype=torch.long, device=model.device)
            times = []
            for i, data in enumerate(model.dataloader.valid, 0):
                time_i = time.time()
                inputdata = data[0].to(model.device)
                label = data[1].to(model.device)
                fake, latent_i, latent_o = model.netg(inputdata)

                _, feat_real = model.netd(inputdata)
                _, feat_fake = model.netd(fake)
                error = torch.mean(torch.pow((latent_i - latent_o), 2), dim=1)

                time_o = time.time()

                an_scores[i * model.opt.batchsize: i * model.opt.batchsize + error.size(0)] = error.reshape(
                    error.size(0))
                gt_labels[i * model.opt.batchsize: i * model.opt.batchsize + error.size(0)] = label.reshape(
                    error.size(0))

                if model.opt.save_test_images and i == 0:
                    test_img_dst = os.path.join(model.opt.outfolder, model.opt.name, model.opt.abnormal_class, 'test',
                                                'images')
                    visualize.save_images(test_img_dst, epoch, inputdata, fake)

                # if model.opt.visulize_feature and i == 0:
                #     feature_img_dst = os.path.join(model.opt.outtrain_dir, 'features')
                #     visualize.tsne_3D(feature_img_dst, epoch, 'feature',
                #                       feat_real.reshape(feat_real.size(0), -1).cpu().numpy(),
                #                       label.reshape(label.size(0), -1).cpu().numpy())
                #     visualize.tsne_2D(feature_img_dst, epoch, 'feature',
                #                       feat_real.reshape(feat_real.size(0), -1).cpu().numpy(),
                #                       label.reshape(label.size(0), -1).cpu().numpy())

                times.append(time_o - time_i)

            times = np.array(times)
            times = np.mean(times[:100] * 1000)
        # Scale error vector between [0, 1]
        an_scores = (an_scores - torch.min(an_scores)) / (
                torch.max(an_scores) - torch.min(an_scores))

        nrm_trn_idx = torch.from_numpy(np.where(an_scores.cpu().numpy() < threshold)[0])
        """ predicte labels for test images, 0 is normal, 1 is anomaly
        """
        pre_labels = torch.ones_like(gt_labels)
        pre_labels[nrm_trn_idx] = 0
        auc = evaluate(gt_labels, an_scores, metric=model.opt.metric)
        # normal_scores = an_scores[np.where(gt_labels.cpu() == 0)[0]]
        # abnormal_scores = an_scores[np.where(gt_labels.cpu() != 0)[0]]
        # print("normal_scores:")
        # print(normal_scores)
        # print('abnormal_scores:')
        # print(abnormal_scores)
        # print(gt_labels)
        print('pre_labels:')
        print(pre_labels)
        performance = OrderedDict([('Avg Run Time (ms/batch)', times), ('AUC', auc)])
        print(performance)
        print('total time:')
        print((time.time() - time_start)*1000)


if __name__ == '__main__':
    main()
