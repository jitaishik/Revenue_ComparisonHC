import numpy as np
import matplotlib.pyplot as plt
import time
import random as rd
import sklearn.metrics
import itertools
from tqdm import tqdm
import argparse
from comparisonHC import HandlerQuadruplets, OracleQuadruplets, get_AddS_quadruplets, get_MulK_quadruplets, ComparisonHC
from utils import planted_model

def cmdline_args():
    # Make parser object
    p = argparse.ArgumentParser()
    p.add_argument("-n", "--n_examples_pure", type=int, default=30, help="")
    p.add_argument("-l", "--levels", type=int, default=3, help="")
    p.add_argument("-mu", "--mean", type=float, default=0.8, help="")
    p.add_argument("-del", "--delta", type=float, default=0.02, help="")
    p.add_argument("-sig", "--sigma", type=float, default=0.1, help="")
    p.add_argument("-r", "--runs", type=int, default=10, help="")
    p.add_argument("-k", "--proportion", type=float, default=1, help="")
    p.add_argument("-noise", "--proportion_noise", type=float, default=0, help="")

    return (p.parse_args())


if __name__ == '__main__':

    args = cmdline_args()
    clusters, dendrogram_truth, similarities = planted_model(n_examples_pure=args.n_examples_pure,levels=args.levels,mu=args.mean,delta=args.delta,sigma=args.sigma)
    
    n = similarities.shape[0]

    # plt.figure()
    # plt.imshow(similarities)
    # plt.show()

    map = np.random.permutation(n)

    similarities_random = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            similarities_random[map[i]][map[j]] = similarities[i][j]

    # plt.figure()
    # plt.imshow(similarities_random)
    # plt.show()

    clusters_random = []
    for cluster in clusters:
        cluster_random =[]
        for index in cluster:
            cluster_random.append(map[index])
        clusters_random.append(cluster_random)

    adds4_ari = []
    adds4_rev = []
    al4k_ari = []
    al4k_rev = []

    for i in tqdm(range(args.runs)):
        Oracle = OracleQuadruplets(similarities_random,n,n_quadruplets=int(2*n*n),proportion_noise=0.05)
        adds_similarities = get_AddS_quadruplets(Oracle,n)
        chc = ComparisonHC(adds_similarities,n)
        chc.fit([[j] for j in range(n)])

        # print("ComparisonHC ran for {:.2f} seconds.".format(chc.time_elapsed))
        score_chc = chc.average_ARI(args.levels,dendrogram_truth,clusters_random)
        adds4_ari.append(score_chc)
        adds4_rev.append(-chc.cost_dasgupta(adds_similarities))

        mulk_similarities = get_MulK_quadruplets(Oracle,n)

        al4k_similarities = mulk_similarities + 2*adds_similarities
        chc_al4k = ComparisonHC(al4k_similarities,n)
        chc_al4k.fit([[j] for j in range(n)])

        # print("ComparisonHC ran for {:.2f} seconds.".format(chc_al4k.time_elapsed))
        score_chc_al4k = chc_al4k.average_ARI(args.levels,dendrogram_truth,clusters_random)
        al4k_ari.append(score_chc_al4k)
        al4k_rev.append(-chc_al4k.cost_dasgupta(adds_similarities))

    adds4_ari = np.array(adds4_ari)
    adds4_rev = np.array(adds4_rev)

    adds4_ari_mean = np.mean(adds4_ari)
    adds4_ari_std = np.std(np.mean(adds4_ari,axis=1))
    adds4_rev_mean = np.mean(adds4_rev)
    adds4_rev_std = np.std(adds4_rev)

    al4k_ari = np.array(al4k_ari)
    al4k_rev = np.array(al4k_rev)

    al4k_ari_mean = np.mean(al4k_ari)
    al4k_ari_std = np.std(np.mean(al4k_ari,axis=1))
    al4k_rev_mean = np.mean(al4k_rev)
    al4k_rev_std = np.std(al4k_rev)

    print("The results over ",args.runs," runs is:")
    print("\t Mean of AARI using AddS-4: ",adds4_ari_mean)
    print("\t Standard Deviation of AARI using AddS-4: ",adds4_ari_std)
    print("\t Mean of Revenue using AddS-4: ",adds4_rev_mean)
    print("\t Standard Deviation of Revenue using AddS-4: ",adds4_rev_std)
    print("\n \n")
    print("\t Mean of AARI using 4K-AL: ",al4k_ari_mean)
    print("\t Standard Deviation of AARI using 4K-AL: ",al4k_ari_std)
    print("\t Mean of Revenue using 4K-AL: ",al4k_rev_mean)
    print("\t Standard Deviation of Revenue using 4K-AL: ",al4k_rev_std)