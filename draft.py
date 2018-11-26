import numpy as np
from numpy.random import dirichlet as dir
from scipy.stats import binom
import random
from scipy.special import digamma, gammaln, betaln
import math
import matplotlib.pyplot as plt

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

        dir_sampling = np.random.dirichlet(alpha0, 1)
        for x_i in range(21):
            phi[X==x_i] = np.random.dirichlet(alpha0, 1)

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
    #uniformly randomly initialize 30 clusters, index 0-29
    # half-open interval [low,high)
    clusters_assignment = np.random.randint(low=0, high=30, size=n)
    num_clusters = np.amax(clusters_assignment) + 1 # 0-based cluster index

    alpha0 = 0.75
    a0 = 0.5
    b0 = 0.5

    each_cluster_size = np.asarray(
        [np.sum(clusters_assignment==idx) for idx in range(num_clusters)])
    for idx, x_i in enumerate(X):
        phi_i = np.zeros(num_clusters)
        for j in range(num_clusters):
            phi_i[j] = binom.pmf(x_i,20,theta)


gibbs()



