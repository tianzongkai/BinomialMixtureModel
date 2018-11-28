import numpy as np
from numpy.random import dirichlet as dir
from scipy.stats import binom
import random
from scipy.special import digamma, gammaln, betaln, beta
from scipy.misc import comb
import math
import matplotlib.pyplot as plt
import time

X = np.loadtxt("data/X.csv") # (2000,)
n = X.shape[0] # n =2000

check = 22

def em():
    random.seed(972)
    K = [3,9,15]
    T = 50
    for idx, k in enumerate(K):
        print "\n*** k = %d ***"%k
        pi = np.repeat(1.0/k, k) # shape (k,)
        # pi = np.asarray([0.2,0.3,0.5])
        theta = np.random.random_sample((k,)) # shape (k,)
        phi = np.zeros((n,k))
        # print pi, theta, phi[check]

        log_likelihood_array = []
        for t in range(T):
            if (t+1)%10 == 0: print "t =",t+1
            # print phi[0]

            # E-step
            for i, x_i in enumerate(X):
                denominator = np.sum(pi * binom.pmf(x_i,20,theta))
                # print denominator
                phi[i] = pi * binom.pmf(x_i,20,theta) / denominator
            # print "phi[check]", phi[check]


            # M-step
            N = np.sum(phi,axis=0) # shape (k,); each item is n_j at M-step
            # print "n_j", N
            phi_xi = np.apply_along_axis(lambda phi_i: phi_i*X, 0, phi) # shape (n,k) apply along each column of phi
            sum_phi_xi = np.sum(phi_xi, axis = 0) # shape (k,)
            theta = sum_phi_xi/20.0/N
            # print sum_phi_xi
            pi = N/n # shape (k,)
            # print "theta",theta
            # print "pi", pi


            # calculate f_t
            log_likelihood = 0.0
            for i in range(n):
                log_likelihood += math.log(np.sum(pi * binom.pmf(X[i],20,theta)))

            log_likelihood_array.append(log_likelihood)

        cluster_list = []
        for x in range(21):
            # print "x=",x
            numerator_list = pi * binom.pmf(x, 20, theta)
            denominator = np.sum(numerator_list)
            p_ci_list = numerator_list / denominator
            # print np.sum(p_ci_list)
            most_prob_cluster = np.argmax(p_ci_list)
            cluster_list.append(most_prob_cluster)

        plt.figure(k)
        plt.scatter(range(21),cluster_list)
        plt.xlabel("X")
        plt.xticks(range(21))
        plt.yticks(range(k))
        plt.grid(True)
        plt.ylabel("index of most probable cluster")
        plt.title("Most probable cluster, K=%d"%k)
        plt.savefig(("EM_part_c_k%d.png") % k)

        plt.figure(20)
        plt.plot(range(T)[2:],log_likelihood_array[2:],label=("K=%d"%k))
        plt.yticks(range(31))
        plt.xlabel("T")
        plt.ylabel("log likelihood")
        plt.legend()
        plt.title("EM")
    plt.savefig("EM_part_b.png")
# em()


def vi():
    # random.seed(23)
    K = [3,15,50]
    T = 1000
    for k in K:
        print "\n*** k = %d ***"%k

        # initialization
        a0 = np.repeat(0.5,k)
        b0 = np.repeat(0.5,k)
        a = np.repeat(0.5,k)
        b = np.repeat(0.5,k)
        alpha0 = np.repeat(0.1,k)
        alpha = np.repeat(0.1,k)

        digamma_a = digamma(a) # shape (K,)
        digamma_b = digamma(b) # shape (K,)
        digamma_a_plus_b = digamma(a+b) # shape (K,)
        digamma_alpha = digamma(alpha) # shape (K,)
        digamma_sum_alpha = digamma(np.sum(alpha)) # scalar
        t2 = digamma_alpha - digamma_sum_alpha  # shape (K,)
        phi = np.random.dirichlet(alpha0, n)
        # print phi[5]

        # dir_sampling = np.random.dirichlet(alpha0, 1)
        # for x_i in range(21):
        #     phi[X==x_i] = np.random.dirichlet(alpha0, 1)

        L_list = []
        for t in range(T):
            if (t+1)%200 == 0: print "t =",t+1

            # update q(pi) = Dirichlet(alpha)
            n_j = np.sum(phi,axis=0) # shape (K,)
            alpha = alpha0 + n_j # shape (K,)

            # update q(theta) for j = 1...K
            phi_xi = np.apply_along_axis(
                lambda phi_i: phi_i * X, 0, phi)  # shape (n,k) apply along each column of phi
            sum_phi_xi = np.sum(phi_xi, axis=0)  # shape (k,)
            phi_20_min_xi = np.apply_along_axis(
                lambda phi_i: phi_i * (20-X), 0, phi)  # shape (n,k) apply along each column of phi
            sum_phi_20_min_xi = np.sum(phi_20_min_xi, axis=0)  # shape (k,)
            a = a0 + sum_phi_xi
            b = b0 + sum_phi_20_min_xi


            # update q(c_i) for i=1...n
            digamma_a = digamma(a) # shape (K,)
            digamma_b = digamma(b) # shape (K,)
            digamma_a_plus_b = digamma(a+b) # shape (K,)
            digamma_alpha = digamma(alpha) # shape (K,)
            digamma_sum_alpha = digamma(np.sum(alpha)) # scalar
            t2 = digamma_alpha - digamma_sum_alpha  # shape (K,)
            exp_t1_plus_t2 = np.exp( # shape (n,K)
                [x_i*(digamma_a-digamma_b)+20*(digamma_b-digamma_a_plus_b)+t2 for x_i in X])
            phi = np.apply_along_axis(
                lambda phi_i: phi_i/np.sum(phi_i), 1, exp_t1_plus_t2) # apply on each row

            # objective function
            lngamma_sum_alpha = gammaln(np.sum(alpha))
            sum_lngamma_alpha = np.sum(gammaln(alpha))
            sum_beta_ab = np.sum(betaln(a,b))
            A = digamma_a - digamma_a_plus_b # shape (K,)
            B = digamma_a - digamma_a_plus_b # shape (K,)
            W = t2 # shape (K,)


            first_term = np.sum(
                phi * (np.outer(X,A) + np.outer(20-X,B) + np.tile(W,n).reshape((n,k))))
            second_term = np.sum((a0 - a)*A + (b0 - b)*B)
            third_term = np.sum((alpha0 - alpha)*W)
            fourth_term = np.sum(phi*np.log(phi))
            firth_term = sum_beta_ab + sum_lngamma_alpha - lngamma_sum_alpha

            L = first_term + second_term + third_term - fourth_term + firth_term
            if t < 5: print "L =", L
            L_list.append(L)
        # print phi[X==5][:3]

        cluster_list = []
        for x_i in range(21):
            cluster_list.append(np.argmax(phi[X==x_i][0]))

        # help me understand the results
        # cluster_of_each_data_point = np.zeros(n)
        # for x, cluster in enumerate(cluster_list):
        #     cluster_of_each_data_point[X==x] = cluster
        #
        # cluster_assignment_subtotal= []
        # for j in range(k):
        #     cluster_assignment_subtotal.append(np.sum(cluster_of_each_data_point==j))
        # print "actual cluster assignment statistics", cluster_assignment_subtotal
        # print "n_j(sum phi_j):", np.sum(phi, axis=0)
        # print "alpha:", alpha
        # print "pi:", np.random.dirichlet(alpha, 1)


        plt.figure(k)
        plt.scatter(range(21),cluster_list)
        plt.xlabel("X")
        plt.xticks(range(21))
        plt.yticks(range(k))
        plt.grid(True)
        plt.ylabel("index of most probable cluster")
        plt.title("Most probable cluster, K=%d"%k)
        plt.savefig(("VI_part_c_k%d.png") % k)


        plt.figure(20)
        plt.plot(range(T)[2:],L_list[2:],label=("K=%d"%k))
        plt.xlabel("T")
        plt.ylabel("variational objective function")
        plt.legend()
        plt.title("VI")
    plt.savefig("VI_part_b.png")
# vi()


def gibbs():
    alpha0 = 0.75
    a0 = 0.5
    b0 = 0.5
    T = 1000
    # initialize each c_i
    # uniformly randomly initialize 30 clusters, index 0-29
    # half-open interval [low,high)
    clusters_assignment = np.random.randint(low=0, high=30, size=n)
    num_clusters = np.amax(clusters_assignment) + 1 # 0-based cluster index
    theta = np.random.beta(a0, b0, size=num_clusters)

    # a list of n_j
    each_cluster_size = np.asarray(
        [np.sum(clusters_assignment==cluster_idx) for cluster_idx in range(num_clusters)])
    num_clusters_list = [num_clusters]
    six_largest_clusters_size = []

    for t in range(T):
        if t%100 == 0: print "t =",t
        # step 1
        for cluster_idx, x_i in enumerate(X):
            # the last index is for new value of j'
            phi_i = np.zeros(num_clusters+1)

            # step 1.a) for all j s.t. n_j^(-1) > 0
            for j in range(num_clusters):
                phi_i[j] = binom.pmf(x_i,20,theta[j]) * each_cluster_size[j] \
                           / (alpha0 + n - 1)

            # step 1.b) for a new value j'
            phi_i[num_clusters] = comb(20,x_i) * \
                                  (math.exp(betaln(a0+x_i,b0+20-x_i) - betaln(a0,b0)) *
                                   (alpha0/(alpha0+n-1)))
            phi_i = phi_i / np.sum(phi_i)

            # step 1.c) sample c_i from a discrete distribution
            c_i = np.random.choice(range(num_clusters+1),p=phi_i)
            old_c_i = clusters_assignment[cluster_idx]
            clusters_assignment[cluster_idx] = c_i

            # step 1.d) generate a new theta_j' if c_i creates a new cluster
            if c_i == num_clusters:
                theta = np.append(theta,np.random.beta(a0, b0))

            # keep n_j up-to-date
            each_cluster_size[old_c_i] -= 1
            if c_i == num_clusters:
                each_cluster_size = np.append(each_cluster_size, 1)
                num_clusters += 1
            else:
                each_cluster_size[c_i] += 1

        # re-index clusters
        cluster_idx_change = 0
        each_cluster_size_new = []
        for cluster_idx, size in enumerate(each_cluster_size):
            if size == 0:
                cluster_idx_change += 1
            elif cluster_idx_change > 0:
                clusters_assignment[clusters_assignment == cluster_idx] -= cluster_idx_change

            if size != 0:
                each_cluster_size_new.append(size)
        num_clusters = len(each_cluster_size_new)
        num_clusters_list.append(num_clusters)
        each_cluster_size = np.asarray(each_cluster_size_new)

        # print num_clusters
        # print each_cluster_size

        # step 2.
        theta = []
        for j in range(num_clusters):
            a_j = a0 + np.sum(X[clusters_assignment==j])
            twenty_minus_X = 20 - X
            b_j = b0 + np.sum(twenty_minus_X[clusters_assignment==j])
            theta.append(np.random.beta(a_j, b_j))
        theta = np.asarray(theta)

        ordered_size_single_iter = -np.sort(-each_cluster_size, kind='mergesort')
        ordered_size_single_iter = np.append(ordered_size_single_iter, np.zeros(5))[:6]
        # print ordered_size_single_iter
        six_largest_clusters_size.append(ordered_size_single_iter)
    six_largest_clusters_size = np.asarray(six_largest_clusters_size) # shape (T,6)
    # print six_largest_clusters_size.shape

    plt.figure(figsize=(12, 6))
    for j in range(6):
        plt.plot(range(T), six_largest_clusters_size[:,j], label=("cluster %d"%j))
    plt.title("6 most probable clusters")
    plt.xlabel("T")
    plt.ylabel("size of clusters")
    plt.savefig("gibbs_part_b.png")

    plt.figure(figsize=(12, 6))
    plt.plot(range(T+1),num_clusters_list)
    plt.xlabel("T")
    plt.yticks(range(np.amin(num_clusters_list), np.amax(num_clusters_list)+1))
    plt.title("number of clusters")
    plt.savefig("gibbs_part_c.png")


start = time.clock()
gibbs()
end = time.clock()
print "running time %.2f minutes" % ((end-start)/60.0)



