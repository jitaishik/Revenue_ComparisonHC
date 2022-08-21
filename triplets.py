import numpy as np
from tqdm import tqdm
import argparse
from comparisonHC import HandlerTriplets, OracleTriplets, get_AddS_triplets, get_MulK_triplets, get_tSTE_triplets, ComparisonHC
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
    map = np.random.permutation(n)

    similarities_random = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            similarities_random[map[i]][map[j]] = similarities[i][j]

    clusters_random = []
    for cluster in clusters:
        cluster_random =[]
        for index in cluster:
            cluster_random.append(map[index])
        clusters_random.append(cluster_random)

    adds3_ari = []
    adds3_rev = []
    tste_ari = []
    tste_rev = []
    mulk_ari = []
    mulk_rev = []

    for i in tqdm(range(args.runs)):
        Oracle = OracleTriplets(similarities_random,n,n_triplets=int((args.proportion)*n*n),proportion_noise=args.proportion_noise)
        adds_similarities = get_AddS_triplets(Oracle,n)
        chc = ComparisonHC(adds_similarities,n)
        chc.fit([[j] for j in range(n)])

        score_chc = chc.average_ARI(args.levels,dendrogram_truth,clusters_random)
        adds3_ari.append(score_chc)
        adds3_rev.append(-chc.cost_dasgupta(adds_similarities))

        tste_similarities = get_tSTE_triplets(Oracle,n)
        chc_tste = ComparisonHC(tste_similarities,n)
        chc_tste.fit([[j] for j in range(n)])

        score_chc_tste = chc_tste.average_ARI(args.levels,dendrogram_truth,clusters_random)
        tste_ari.append(score_chc_tste)
        tste_rev.append(-chc_tste.cost_dasgupta(adds_similarities))

        mulk_similarities = get_MulK_triplets(Oracle,n)
        chc_mulk = ComparisonHC(mulk_similarities,n)
        chc_mulk.fit([[j] for j in range(n)])

        score_chc_mulk = chc_mulk.average_ARI(args.levels,dendrogram_truth,clusters_random)
        mulk_ari.append(score_chc_mulk)
        mulk_rev.append(-chc_mulk.cost_dasgupta(adds_similarities))

    adds3_ari = np.array(adds3_ari)
    adds3_rev = np.array(adds3_rev)

    adds3_ari_mean = np.mean(adds3_ari)
    adds3_ari_std = np.std(np.mean(adds3_ari,axis=1))
    adds3_rev_mean = np.mean(adds3_rev)
    adds3_rev_std = np.std(adds3_rev)

    tste_ari = np.array(tste_ari)
    tste_rev = np.array(tste_rev)

    tste_ari_mean = np.mean(tste_ari)
    tste_ari_std = np.std(np.mean(tste_ari,axis=1))
    tste_rev_mean = np.mean(tste_rev)
    tste_rev_std = np.std(tste_rev)

    mulk_ari = np.array(mulk_ari)
    mulk_rev = np.array(mulk_rev)

    mulk_ari_mean = np.mean(mulk_ari)
    mulk_ari_std = np.std(np.mean(mulk_ari,axis=1))
    mulk_rev_mean = np.mean(mulk_rev)
    mulk_rev_std = np.std(mulk_rev)

    print("The results over ",args.runs," runs is:")
    print("\t Mean of AARI using AddS-3: ",adds3_ari_mean)
    print("\t Standard Deviation of AARI using AddS-3: ",adds3_ari_std)
    print("\t Mean of Revenue using AddS-3: ",adds3_rev_mean)
    print("\t Standard Deviation of Revenue using AddS-3: ",adds3_rev_std)
    print("\n \n")
    print("\t Mean of AARI using t-STE: ",tste_ari_mean)
    print("\t Standard Deviation of AARI using t-STE: ",tste_ari_std)
    print("\t Mean of Revenue using t-STE: ",tste_rev_mean)
    print("\t Standard Deviation of Revenue using t-STE: ",tste_rev_std)
    print("\n \n")
    print("\t Mean of AARI using MulK-3: ",mulk_ari_mean)
    print("\t Standard Deviation of AARI using MulK-3: ",mulk_ari_std)
    print("\t Mean of Revenue using MulK-3: ",mulk_rev_mean)
    print("\t Standard Deviation of Revenue using MulK-3: ",mulk_rev_std)