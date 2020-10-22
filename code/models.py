"""
GAN model zoo
"""
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import itertools
import torch.nn.utils.spectral_norm as sn
import torch.nn.functional as F


#  mlp model
class MLP(nn.Module):
    def __init__(self, num_layer, num_nodes, relu_final=False):
        super(MLP, self).__init__()
        main = nn.Sequential()
        for l in np.arange(num_layer - 1):
            main.add_module('linear{0}'.format(l), nn.Linear(num_nodes[l], num_nodes[l + 1]))
            if relu_final:
                main.add_module('relu{0}'.format(l), nn.ReLU())
            else:
                if num_layer > 2 and l < num_layer - 2: # 2 layers = linear network, >2 layers, relu net
                    main.add_module('relu{0}'.format(l), nn.ReLU())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


# x = z + mu(y)+mu(d)
class Linear_Generator(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, cl_dim, do_dim, z_dim, num_layer=1, num_nodes=64, is_reg=False, dagMat=None):
        super(Linear_Generator, self).__init__()
        self.lc = nn.Linear(cl_num, cl_dim, bias=False)
        self.ld = nn.Linear(do_num, do_dim, bias=False)
        self.le = nn.Linear(cl_dim, i_dim * do_dim, bias=False)
        self.i_dim = i_dim
        self.cl_num = cl_num
        self.do_dim = do_dim

    def forward(self, noise, input_c, input_d, device='cpu'):
        output_c = self.lc(input_c)
        output_d = self.ld(input_d)
        output_e = self.le(input_c)
        output_d = torch.repeat_interleave(output_d, self.i_dim, dim=1)
        output_ed = output_d * output_e
        output_sq = torch.zeros(noise.shape[0], self.i_dim, device=device)
        for i in range(self.do_dim):
            output_sq += output_ed[:, i*self.i_dim:(i+1)*self.i_dim]
        output = noise + output_c + output_sq
        return output


# a MLP generator
class MLP_Generator(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, cl_dim, do_dim, z_dim, num_layer=1, num_nodes=64, is_reg=False, prob=True):
        super(MLP_Generator, self).__init__()
        self.prob = prob
        if prob:
            # VAE posterior parameters, Gaussian
            self.mu = nn.Parameter(torch.zeros(do_num, do_dim))
            self.sigma = nn.Parameter(torch.zeros(do_num, do_dim))
        else:
            self.ld = nn.Linear(do_num, do_dim, bias=False)
        self.lc = nn.Linear(cl_num, cl_dim, bias=False)
        self.decoder = MLP(num_layer + 2, [z_dim+cl_dim+do_dim] + [num_nodes]*num_layer + [i_dim])
        self.is_reg = is_reg

    def forward(self, noise, input_c, input_d, noise_d=None):
        if self.is_reg:
            output_c = input_c
        else:
            output_c = self.lc(input_c)
        if self.prob:
            theta = self.mu + torch.mul(torch.log(1+torch.exp(self.sigma)), noise_d)
            output_d = torch.matmul(input_d, theta)
        else:
            output_d = self.ld(input_d)
        output = self.decoder(torch.cat((output_c, output_d, noise), axis=1))
        if self.prob:
            KL_reg = 1 + torch.log(torch.log(1+torch.exp(self.sigma))**2) - self.mu**2 - torch.log(1+torch.exp(self.sigma))**2
            if KL_reg.shape[1] > 1:
                KL_reg = KL_reg.sum(axis=1)
            return output, -KL_reg
        else:
            return output


# a MLP auxiliary classifier, shared auxiliary classifiers
class MLP_AuxClassifier(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, num_layer=1, num_nodes=64, is_reg=False):
        super(MLP_AuxClassifier, self).__init__()
        self.cls = MLP(num_layer + 2, [i_dim] + [num_nodes]*num_layer +[cl_num])
        self.common_net = MLP(num_layer + 1, [i_dim] + [num_nodes]*num_layer, relu_final=True)
        if is_reg:
            self.aux_c = nn.Linear(num_nodes, 1)
            self.aux_c_tw = nn.Linear(num_nodes, 1)
        else:
            self.aux_c = nn.Linear(num_nodes, cl_num)
            self.aux_c_tw = nn.Linear(num_nodes, cl_num)
        self.aux_d = nn.Linear(num_nodes, do_num)
        self.aux_d_tw = nn.Linear(num_nodes, do_num)

    def forward(self, input0):
        input = self.common_net(input0)
        output_c = self.aux_c(input)
        output_c_tw = self.aux_c_tw(input)
        output_d = self.aux_d(input)
        output_d_tw = self.aux_d_tw(input)
        output_cls = self.cls(input0)
        return output_c, output_c_tw, output_d, output_d_tw, output_cls


# # a MLP auxiliary classifier, older version (separate classifier for each domain)
# class MLP_AuxClassifier(nn.Module):
#     def __init__(self, i_dim, cl_num, do_num, do_dim, num_layer=1, num_nodes=[64], is_reg=False):
#         super(MLP_AuxClassifier, self).__init__()
#         self.do_num = do_num
#         self.cl_num = cl_num
#         self.common_dnet = MLP(num_layer + 1, [i_dim] + num_nodes, relu_final=True)
#         self.aux_d = nn.Linear(num_nodes[-1], do_num)
#         self.aux_d_tw = nn.Linear(num_nodes[-1], do_num)
#
#         self.common_cnet = nn.ModuleList()
#         self.aux_c = nn.ModuleList()
#         self.aux_c_tw = nn.ModuleList()
#         for i in range(do_num):
#             self.common_cnet.append(MLP(num_layer + 1, [i_dim] + num_nodes, relu_final=True))
#             self.aux_c.append(nn.Linear(num_nodes[-1], cl_num))
#             self.aux_c_tw.append(nn.Linear(num_nodes[-1], cl_num))
#
#         self.aux_c_minCh = MLP(num_layer + 2, [i_dim] + num_nodes + [cl_num])
#
#     def forward(self, input_x, input_d, device='cpu'):
#         common_d = self.common_dnet(input_x)
#         output_d = self.aux_d(common_d)
#         output_d_tw = self.aux_d_tw(common_d)
#         output_c = torch.zeros(input_x.size(0), self.cl_num, device=device)
#         output_c_tw = torch.zeros(input_x.size(0), self.cl_num, device=device)
#         for i in range(self.do_num):
#             id_i = (input_d == i)
#             common_c = self.common_cnet[i](input_x[id_i])
#             output_c[id_i] = self.aux_c[i](common_c)
#             output_c_tw[id_i] = self.aux_c_tw[i](common_c)
#
#         return output_c, output_c_tw, output_d, output_d_tw
#
#     def forward_minCh(self, input):
#         output_c = self.aux_c_minCh(input)
#         return output_c
#
#     def forward_test(self, input):
#         inputs_c = self.common_cnet[self.do_num-1](input)
#         output_c = self.aux_c[self.do_num-1](inputs_c)
#         return output_c


# a MLP classifier
class MLP_Classifier(nn.Module):
    def __init__(self, i_dim, cl_num, num_layer=1, num_nodes=64):
        super(MLP_Classifier, self).__init__()
        self.net = MLP(num_layer + 2, [i_dim] + [num_nodes]*num_layer +[cl_num])

    def forward(self, input):
        output_c = self.net(input)
        return output_c


# The graph nodes.
class Data(object):
    def __init__(self, name):
        self.__name = name
        self.__links = set()

    @property
    def name(self):
        return self.__name

    @property
    def links(self):
        return set(self.__links)

    def add_link(self, other):
        self.__links.add(other)
        other.__links.add(self)


# Class to represent a graph, for topological sort of DAG
class Graph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.V = vertices  # No. of vertices

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, v, visited, stack):

        # Mark the current node as visited.
        visited[v] = True

        # Recur for all the vertices adjacent to this vertex
        for i in self.graph[v]:
            if visited[i] is False:
                self.topologicalSortUtil(i, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, v)

    # The function to do Topological Sort. It uses recursive
    # topologicalSortUtil()
    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = [False] * self.V
        stack = []

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for i in range(self.V):
            if visited[i] is False:
                self.topologicalSortUtil(i, visited, stack)

        # Return contents of stack
        return stack


# a decoder according to a DAG
class DAG_Generator(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, cl_dim, do_dim, z_dim, num_layer=1, num_nodes=64, is_reg=False, dagMat=None, prob=True):
        super(DAG_Generator, self).__init__()
        # create a dag
        dag = Graph(i_dim)

        for i in range(i_dim):
            for j in range(i_dim):
                if dagMat[j, i]:
                    dag.addEdge(i, j)

        # extract y and d signs
        self.yd_sign = dagMat[:, -2:]
        dagMat = dagMat[:, :-2]

        # topological sort
        nodeSort = dag.topologicalSort()
        numInput = dagMat.sum(1)

        # define class and domain conditional networks
        self.prob = prob
        if prob:
            # VAE posterior parameters, Gaussian
            self.mu = nn.Parameter(torch.zeros(do_num, do_dim * i_dim))
            self.sigma = nn.Parameter(torch.zeros(do_num, do_dim * i_dim))
        else:
            self.dnet = nn.Linear(do_num, do_dim * i_dim, bias=False)
        if not is_reg:
            self.cnet = nn.Linear(cl_num, cl_dim * i_dim, bias=False)

        # construct generative network according to the dag
        nets = nn.ModuleList()
        for i in range(i_dim):
            num_nodesIn = int(numInput[i]) + cl_dim + do_dim + z_dim
            num_nodes_i = [num_nodesIn] + [num_nodes]*num_layer + [1]
            netMB = MLP(num_layer + 2, num_nodes_i)
            nets.append(netMB)

        # prediction network
        self.nets = nets
        self.nodeSort = nodeSort
        self.nodesA = np.array(range(i_dim)).reshape(i_dim, 1).tolist()
        self.i_dim = i_dim
        self.i_dimNew = i_dim
        self.do_num = do_num
        self.cl_num = cl_num
        self.cl_dim = cl_dim
        self.do_dim = do_dim
        self.z_dim = z_dim
        self.dagMat = dagMat
        self.numInput = numInput
        self.is_reg = is_reg
        self.ischain = False

        # inputs: class indicator, domain indicator, noise, features
        # separate forward for each factor
    def forward_indep(self, noise, input_c, input_d, input_x, noise_d=None, device='cpu'):
        # class parameter network
        batch_size = input_c.size(0)
        if self.is_reg:
            inputs_c = input_c.view(batch_size, 1)
        else:
            inputs_c = self.cnet(input_c)
        if self.prob:
            theta = self.mu + torch.mul(torch.log(1+torch.exp(self.sigma)), noise_d)
            inputs_d = torch.matmul(input_d, theta)
        else:
            inputs_d = self.dnet(input_d)

        inputs_n = noise
        inputs_f = input_x

        # create output array
        output = torch.zeros((batch_size, len(self.nodeSort)))
        output = output.to(device)

        # create a network for each module
        for i in self.nodeSort:
            inputs_pDim = self.numInput[i]
            if inputs_pDim > 0:
                index = np.argwhere(self.dagMat[i, :])
                index = index.flatten()
                index = [int(j) for j in index]
                inputs_p = inputs_f[:, index] # get the parent data from real data, not fake data!!!
            if not self.is_reg:
                inputs_ci = inputs_c[:, i*self.cl_dim:(i+1)*self.cl_dim]
            else:
                inputs_ci = inputs_c
            inputs_di = inputs_d[:, i*self.do_dim:(i+1)*self.do_dim]
            inputs_ni = inputs_n[:, i*self.z_dim:(i+1)*self.z_dim]
            if inputs_pDim > 0:
                inputs_i = torch.cat((inputs_ci, inputs_di, inputs_ni, inputs_p), 1)
            else:
                inputs_i = torch.cat((inputs_ci, inputs_di, inputs_ni), 1)

            output[:, i] = self.nets[i](inputs_i).squeeze()

        if self.prob:
            KL_reg = 1 + torch.log(torch.log(1+torch.exp(self.sigma))**2) - self.mu**2 - torch.log(1+torch.exp(self.sigma))**2
            if KL_reg.shape[1] > 1:
                KL_reg = KL_reg.sum(axis=1)
            return output, -KL_reg
        else:
            return output

    # inputs: class indicator, domain indicator, noise
    # forward for all factors in a graph
    def forward(self, noise, input_c, input_d, device='cpu', noise_d=None):
        # class parameter network
        batch_size = input_c.size(0)
        if self.is_reg:
            inputs_c = input_c.view(batch_size, 1)
        else:
            inputs_c = self.cnet(input_c)
        if self.prob:
            theta = self.mu + torch.mul(torch.log(1+torch.exp(self.sigma)), noise_d)
            inputs_d = torch.matmul(input_d, theta)
        else:
            inputs_d = self.dnet(input_d)

        inputs_n = noise

        output = torch.zeros((batch_size, len(self.nodeSort)))
        output = output.to(device)

        # create a network for each module
        for i in self.nodeSort:
            inputs_pDim = self.numInput[i]
            if inputs_pDim > 0:
                index = np.argwhere(self.dagMat[i, :])
                index = index.flatten()
                index = [int(j) for j in index]
                inputs_p = output[:, index]

            if not self.is_reg:
                inputs_ci = inputs_c[:, i * self.cl_dim:(i + 1) * self.cl_dim]
            else:
                inputs_ci = inputs_c
            inputs_di = inputs_d[:, i * self.do_dim:(i + 1) * self.do_dim]
            inputs_ni = inputs_n[:, i * self.z_dim:(i + 1) * self.z_dim]
            if inputs_pDim > 0:
                inputs_i = torch.cat((inputs_ci, inputs_di, inputs_ni, inputs_p), 1)
            else:
                inputs_i = torch.cat((inputs_ci, inputs_di, inputs_ni), 1)

            output[:, i] = self.nets[i](inputs_i).squeeze()

        return output


# a decoder according to a partial DAG
class PDAG_Generator(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, cl_dim, do_dim, z_dim, num_layer=1, num_nodes=64, is_reg=False, dagMat=None, prob=True):
        super(PDAG_Generator, self).__init__()

        # find undirected groups of variables
        nodes = list()
        for i in range(i_dim):
            nodes.append(Data(i))

        # extract y and d signs
        self.yd_sign = dagMat[:, -2:]
        dagMat = dagMat[:, :-2]

        # remove directed edges in the adjcency matrix
        dagMatUnD = np.copy(dagMat)
        dagMatUnD[dagMatUnD == 1] = 0

        # add links between nodes
        for i in range(i_dim):
            for j in range(i_dim):
                if dagMatUnD[j, i] == -1:
                    nodes[i].add_link(nodes[j])

        # find connected components
        self.nodes = set(nodes)
        nodesUnD = list()
        nodesD = list()
        for components in self.connected_components():
            nodesTmp = list()
            for node in components:
                nodesTmp.append(node.name)
            if len(components) > 1:
                nodesUnD.append(nodesTmp)
            else:
                nodesD.append(nodesTmp)

        # sort the nodes ascend
        nodesD.sort()
        for i in range(len(nodesUnD)):
            nodesUnD[i].sort()
        nodesUnD.sort()

        nodesA = nodesD + nodesUnD
        nodesD_flat = list(itertools.chain.from_iterable(nodesD))

        # modify the adjcency matrix by considering undirected groups as a variable
        dagMatNew = np.zeros((len(nodesA), len(nodesA)), dtype=int)
        dagMatNew[0:len(nodesD), 0:len(nodesD)] = dagMat[np.ix_(nodesD_flat, nodesD_flat)]

        idx = len(nodesD)
        for i in nodesUnD:
            matOut = dagMat[np.ix_(nodesD_flat, i)]
            matIn = dagMat[np.ix_(i, nodesD_flat)]
            matOut = matOut.sum(1)
            matIn = matIn.sum(0)
            dagMatNew[0:len(nodesD), idx] = matOut
            dagMatNew[idx, 0:len(nodesD)] = matIn

        # create a dag
        i_dimNew = len(nodesA)
        dag = Graph(i_dimNew)
        for i in range(i_dimNew):
            for j in range(i_dimNew):
                if dagMatNew[j, i]:
                    dag.addEdge(i, j)

        # topological sort
        nodeSort = dag.topologicalSort()
        numInput = dagMatNew.sum(1)

        # define class and domain conditional networks
        self.prob = prob
        if prob:
            # VAE posterior parameters, Gaussian
            self.mu = nn.Parameter(torch.zeros(do_num, do_dim * i_dimNew))
            self.sigma = nn.Parameter(torch.ones(do_num, do_dim * i_dimNew))
        else:
            self.dnet = nn.Linear(do_num, do_dim * i_dimNew, bias=False)
        if not is_reg:
            self.cnet = nn.Linear(cl_num, cl_dim * i_dimNew, bias=False) # need to fix the dimension to cl_dim*i_dimNew

        nets = nn.ModuleList()
        dimNoise = np.zeros(i_dimNew, dtype=int)
        for i in range(i_dimNew):
            num_nodesIn = int(numInput[i]) + cl_dim + do_dim + z_dim * len(nodesA[i])
            num_nodes_i = [num_nodesIn] + [num_nodes]*num_layer + [len(nodesA[i])]
            netMB = MLP(num_layer + 2, num_nodes_i)
            nets.append(netMB)
            dimNoise[i] = len(nodesA[i])

        self.nets = nets
        self.nodeSort = nodeSort
        self.i_dim = i_dim
        self.i_dimNew = i_dimNew
        self.cl_num = cl_num
        self.do_num = do_num
        self.cl_dim = cl_dim
        self.do_dim = do_dim
        self.z_dim = z_dim
        self.dimNoise = dimNoise
        self.dagMat = dagMat
        self.dagMatNew = dagMatNew
        self.numInput = numInput
        self.nodesA = nodesA
        self.ischain = True
        self.is_reg = is_reg

    # input: class indicator, domain indicator, noise, inputs, separate learning of modules
    def forward_indep(self, noise, input_c, input_d, input_x,  device='cpu', noise_d=None):
        # class parameter network
        batch_size = input_c.size(0)
        if self.is_reg:
            inputs_c = input_c.view(batch_size, 1)
        else:
            inputs_c = self.cnet(input_c)
        if self.prob:
            theta = self.mu + torch.mul(torch.log(1+torch.exp(self.sigma)), noise_d[:, :self.i_dimNew * self.do_dim])
            inputs_d = torch.matmul(input_d, theta)
        else:
            inputs_d = self.dnet(input_d)

        inputs_n = noise
        inputs_f = input_x

        output = torch.zeros((batch_size, self.i_dim)).to(device)

        # create a network for each module
        for i in self.nodeSort:
            input_pDim = self.numInput[i]
            if input_pDim > 0:
                if len(self.nodesA[i]) == 1:
                    index = np.argwhere(self.dagMat[self.nodesA[i][0], :])
                    index = index.flatten()
                    index = [int(j) for j in index]
                    input_p = inputs_f[:, index]
                else:
                    index = np.argwhere(self.dagMatNew[i, :])
                    index = index.flatten()
                    index = [self.nodesA[j] for j in index]
                    index = list(itertools.chain.from_iterable(index))
                    index = [int(j) for j in index]
                    input_p = inputs_f[:, index]

            if not self.is_reg:
                input_ci = inputs_c[:, i * self.cl_dim:(i + 1) * self.cl_dim]
            else:
                input_ci = inputs_c
            input_di = inputs_d[:, i * self.do_dim:(i + 1) * self.do_dim]
            input_ni = inputs_n[:, self.dimNoise[0:i].sum():self.dimNoise[0:i + 1].sum()]
            if input_pDim > 0:
                input_i = torch.cat((input_ci, input_di, input_ni, input_p), 1)
            else:
                input_i = torch.cat((input_ci, input_di, input_ni), 1)

            output[:, self.nodesA[i]] = self.nets[i](input_i)
        if self.prob:
            KL_reg = 1 + torch.log(torch.log(1+torch.exp(self.sigma)) ** 2) - self.mu ** 2 - torch.log(1+torch.exp(self.sigma)) ** 2
            if KL_reg.shape[1] > 1:
                KL_reg = KL_reg.sum(axis=1)
            return output, -KL_reg
        else:
            return output

    # input: class indicator, domain indicator, noise, joint learning of modules
    def forward(self, noise, input_c, input_d, device='cpu', noise_d=None):
        # class parameter network
        batch_size = input_c.size(0)

        if self.is_reg:
            inputs_c = input_c.view(batch_size, 1)
        else:
            inputs_c = self.cnet(input_c)
        if self.prob:
            # theta = self.mu + torch.mul(self.sigma, noise_d)
            theta = self.mu + torch.mul(torch.log(1+torch.exp(self.sigma)), noise_d[:, :self.i_dimNew * self.do_dim])

            inputs_d = torch.matmul(input_d, theta)
        else:
            inputs_d = self.dnet(input_d)

        inputs_n = noise

        output = torch.zeros((batch_size, self.i_dim)).to(device)

        # create a network for each module
        for i in self.nodeSort:
            input_pDim = self.numInput[i]
            if input_pDim > 0:
                if len(self.nodesA[i]) == 1:
                    index = np.argwhere(self.dagMat[self.nodesA[i][0], :])
                    index = index.flatten()
                    index = [int(j) for j in index]
                    input_p = output[:, index]
                else:
                    index = np.argwhere(self.dagMatNew[i, :])
                    index = index.flatten()
                    index = [self.nodesA[j] for j in index]
                    index = list(itertools.chain.from_iterable(index))
                    index = [int(j) for j in index]
                    input_p = output[:, index]

            if not self.is_reg:
                input_ci = inputs_c[:, i * self.cl_dim:(i + 1) * self.cl_dim]
            else:
                input_ci = inputs_c
            input_di = inputs_d[:, i * self.do_dim:(i + 1) * self.do_dim]
            input_ni = inputs_n[:, self.dimNoise[0:i].sum():self.dimNoise[0:i + 1].sum()]
            if input_pDim > 0:
                input_i = torch.cat((input_ci, input_di, input_ni, input_p), 1)
            else:
                input_i = torch.cat((input_ci, input_di, input_ni), 1)

            output[:, self.nodesA[i]] = self.nets[i](input_i)

        return output

    # The function to look for connected components.
    def connected_components(self):

        nodes = self.nodes
        # List of connected components found. The order is random.
        result = []
        nodes = set(nodes)

        # Iterate while we still have nodes to process.
        while nodes:

            # Get a random node and remove it from the global set.
            n = nodes.pop()
            group = {n}
            queue = [n]

            # Iterate the queue.
            # When it's empty, we finished visiting a group of connected nodes.
            while queue:
                # Consume the next item from the queue.
                n = queue.pop(0)
                neighbors = n.links
                neighbors.difference_update(group)
                nodes.difference_update(neighbors)
                group.update(neighbors)
                queue.extend(neighbors)

            # Add the group to the list of groups.
            result.append(group)

        # Return the list of groups.
        return result


class CNN_Classifier(nn.Module):
    def __init__(self, i_dim, cl_num, ch):
        super(CNN_Classifier, self).__init__()
        self.i_dim = i_dim
        self.ch = ch
        self.cl_num = cl_num

        self.common_net = nn.Sequential(
            nn.Conv2d(i_dim, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch*2, 4, 2, 1),
            nn.BatchNorm2d(ch*2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch*2, ch*4, 3, 2, 1),
            nn.BatchNorm2d(ch*4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.aux_c = nn.Linear(4*4*4*ch, self.cl_num)

    def forward(self, input0):
        input = self.common_net(input0)
        input = input.view(-1, 4*4*4*self.ch)
        output_c = self.aux_c(input)
        return output_c


class CNN_AuxClassifier(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, ch=64):
        super(CNN_AuxClassifier, self).__init__()
        self.i_dim = i_dim
        self.ch = ch
        self.cl_num = cl_num
        self.do_num = do_num

        self.common_net = nn.Sequential(
            nn.Conv2d(i_dim, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.aux_c = nn.Linear(4*4*4*ch, self.cl_num)
        self.aux_d = nn.Linear(4*4*4*ch, self.do_num)
        self.aux_c_tw = nn.Linear(4*4*4*ch, self.cl_num)
        self.aux_d_tw = nn.Linear(4*4*4*ch, self.do_num)
        self.disc = nn.Linear(4*4*4*ch, 1)
        self.cls = CNN_Classifier(i_dim, cl_num, ch)

    def forward(self, input0):
        input = self.common_net(input0)
        input = input.view(-1, 4*4*4*self.ch)
        output_c = self.aux_c(input)
        output_c_tw = self.aux_c_tw(input)
        output_d = self.aux_d(input)
        output_d_tw = self.aux_d_tw(input)
        output_disc = self.disc(input)
        output_cls = self.cls(input0)

        return output_c, output_c_tw, output_d, output_d_tw, output_cls, output_disc


# a CNN generator
class CNN_Generator(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, cl_dim, do_dim, z_dim, ch=64, prob=True):
        super(CNN_Generator, self).__init__()
        self.prob = prob
        self.do_dim = do_dim
        self.ch = ch
        if prob:
            # VAE posterior parameters, Gaussian
            self.mu = nn.Parameter(torch.zeros(do_num, do_dim))
            self.sigma = nn.Parameter(torch.zeros(do_num, do_dim))
        else:
            self.ld = nn.Linear(do_num, do_dim, bias=False)
        self.lc = nn.Linear(cl_num, cl_dim, bias=False)

        self.decoder1 = nn.Sequential(
            nn.Linear(z_dim + cl_dim + do_dim, ch*4*4*4),
            nn.BatchNorm1d(ch*4*4*4),
            nn.ReLU(True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch*2, ch, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch, i_dim, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, input_c, input_d, noise_d=None):
        embed_c = self.lc(input_c)
        if self.prob:
            theta = self.mu + torch.mul(torch.log(1+torch.exp(self.sigma)), noise_d)
            embed_d = torch.matmul(input_d, theta)
        else:
            embed_d = self.ld(input_d)
        output = self.decoder1(torch.cat((embed_c, noise, embed_d), axis=1))
        output = output.view(output.size(0), self.ch*4, 4, 4)
        output = self.decoder2(output)
        if self.prob:
            KL_reg = 1 + torch.log(torch.log(1+torch.exp(self.sigma))**2) - self.mu**2 - torch.log(1+torch.exp(self.sigma))**2
            if KL_reg.shape[1] > 1:
                KL_reg = KL_reg.sum(axis=1)
            return output, -KL_reg
        else:
            return output

# a CNN generator
class CNN_Generator_Exp(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, cl_dim, do_dim, z_dim, ch=64, prob=True):
        super(CNN_Generator_Exp, self).__init__()
        self.prob = prob
        self.do_dim = do_dim
        self.ch = ch
        if prob:
            # VAE posterior parameters, Gaussian
            self.mu = nn.Parameter(torch.zeros(do_num, do_dim))
            self.sigma = nn.Parameter(torch.zeros(do_num, do_dim))
        else:
            self.ld = nn.Linear(do_num, do_dim, bias=False)
        self.lc = nn.Linear(cl_num, cl_dim, bias=False)

        self.decoder1 = nn.Sequential(
            nn.Linear(z_dim + cl_dim + do_dim, ch*4*4*4),
            nn.BatchNorm1d(ch*4*4*4),
            nn.ReLU(True),
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(ch*4, ch*2, kernel_size=2, stride=2),
            nn.BatchNorm2d(ch*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch*2, ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch, i_dim, kernel_size=2, stride=2),
            nn.Tanh()
        )

    def forward(self, noise, input_c, input_d, noise_d=None):
        embed_c = self.lc(input_c)
        if self.prob:
            theta = self.mu + torch.mul(torch.log(1+torch.exp(self.sigma)), noise_d)
            embed_d = torch.matmul(input_d, theta)
        else:
            embed_d = self.ld(input_d)
        output = self.decoder1(torch.cat((embed_c, noise, embed_d), axis=1))
        output = output.view(-1, self.ch*4, 4, 4)
        output = self.decoder2(output)
        if self.prob:
            KL_reg = 1 + torch.log(torch.log(1+torch.exp(self.sigma))**2) - self.mu**2 - torch.log(1+torch.exp(self.sigma))**2
            if KL_reg.shape[1] > 1:
                KL_reg = KL_reg.sum(axis=1)
            return output, -KL_reg
        else:
            return output


class CNN_Classifier_Exp(nn.Module):
    def __init__(self, i_dim, cl_num, ch):
        super(CNN_Classifier_Exp, self).__init__()
        self.i_dim = i_dim
        self.ch = ch
        self.cl_num = cl_num

        self.common_net = nn.Sequential(
            nn.Conv2d(i_dim, ch, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(ch, ch*2, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(ch*2, ch*4, 3, 2, 1),
            nn.LeakyReLU(),
        )
        self.aux_c = nn.Linear(4*4*4*ch, self.cl_num)

    def forward(self, input0):
        input = self.common_net(input0)
        input = input.view(-1, 4*4*4*self.ch)
        output_c = self.aux_c(input)
        return output_c


class CNN_AuxClassifier_Exp(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, ch=64):
        super(CNN_AuxClassifier_Exp, self).__init__()
        self.i_dim = i_dim
        self.ch = ch
        self.cl_num = cl_num
        self.do_num = do_num

        self.common_net = nn.Sequential(
            nn.Conv2d(i_dim, ch, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(ch, ch * 2, 3, 2, 1),
            nn.LeakyReLU(),
            nn.Conv2d(ch * 2, ch * 4, 3, 2, 1),
            nn.LeakyReLU(),
        )
        self.aux_c = nn.Linear(4*4*4*ch, self.cl_num)
        self.aux_d = nn.Linear(4*4*4*ch, self.do_num)
        self.aux_c_tw = nn.Linear(4*4*4*ch, self.cl_num)
        self.aux_d_tw = nn.Linear(4*4*4*ch, self.do_num)
        self.disc = nn.Linear(4*4*4*ch, 1)
        self.cls = CNN_Classifier_Exp(i_dim, cl_num, ch)

    def forward(self, input0):
        input = self.common_net(input0)
        input = input.view(-1, 4*4*4*self.ch)
        output_c = self.aux_c(input)
        output_c_tw = self.aux_c_tw(input)
        output_d = self.aux_d(input)
        output_d_tw = self.aux_d_tw(input)
        output_disc = self.disc(input)
        output_cls = self.cls(input0)

        return output_c, output_c_tw, output_d, output_d_tw, output_cls, output_disc


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=[3, 3],
                 padding=1, stride=1, n_class=None, bn=False,
                 activation=F.relu, upsample=True, downsample=False, SN=False, emb=None, rate=1.0):
        super().__init__()

        gain = 2 ** 0.5

        self.emb = emb

        if SN:
            self.conv1 = sn(nn.Conv2d(in_channel, out_channel,
                                                 kernel_size, stride, padding,
                                                 bias=False if bn else True))
            self.conv2 = sn(nn.Conv2d(out_channel, out_channel,
                                                 kernel_size, stride, padding,
                                                 bias=False if bn else True))
        else:
            self.conv1 = nn.Conv2d(in_channel, out_channel,
                                                 kernel_size, stride, padding,
                                                 bias=False if bn else True)
            self.conv2 = nn.Conv2d(out_channel, out_channel,
                                                 kernel_size, stride, padding,
                                                 bias=False if bn else True)

        self.skip_proj = False
        if in_channel != out_channel or upsample or downsample:
            if SN == True:
                self.conv_skip = sn(nn.Conv2d(in_channel, out_channel,
                                                         1, 1, 0))
            else:
                self.conv_skip = nn.Conv2d(in_channel, out_channel,
                                                         1, 1, 0)

            self.skip_proj = True

        self.upsample = upsample
        self.downsample = downsample
        self.activation = activation
        self.bn = bn

    def forward(self, input, class_id=None):
        out = input

        out = self.activation(out)
        if self.upsample:
            out = F.upsample(out, scale_factor=2)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        if self.skip_proj:
            skip = input
            if self.upsample:
                skip = F.upsample(skip, scale_factor=2)
            skip = self.conv_skip(skip)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)

        else:
            skip = input

        return out + skip


# a CNN generator
class RES_Generator(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, cl_dim, do_dim, z_dim, ch=64, prob=True):
        super(RES_Generator, self).__init__()
        self.prob = prob
        self.do_dim = do_dim
        self.ch = ch
        if prob:
            # VAE posterior parameters, Gaussian
            self.mu = nn.Parameter(torch.zeros(do_num, do_dim))
            self.sigma = nn.Parameter(torch.zeros(do_num, do_dim))
        else:
            self.ld = nn.Linear(do_num, do_dim, bias=False)
        self.lc = nn.Linear(cl_num, cl_dim, bias=False)

        self.decoder0 = nn.Sequential(
            nn.Linear(z_dim + cl_dim + do_dim, ch * 4 * 4),
        )
        self.decoder = nn.Sequential(
            ConvBlock(ch, ch),
            ConvBlock(ch, ch),
            ConvBlock(ch, i_dim),
            nn.Tanh()
        )

    def forward(self, noise, input_c, input_d, noise_d=None):
        embed_c = self.lc(input_c)
        if self.prob:
            theta = self.mu + torch.mul(torch.log(1+torch.exp(self.sigma)), noise_d)
            embed_d = torch.matmul(input_d, theta)
        else:
            embed_d = self.ld(input_d)
        z_cat = torch.cat((embed_c, noise, embed_d), axis=1)
        z_cat = self.decoder0(z_cat)
        z_cat = z_cat.view(z_cat.size(0), self.ch, 4, 4)
        output = self.decoder(z_cat)
        if self.prob:
            KL_reg = 1 + torch.log(torch.log(1+torch.exp(self.sigma))**2) - self.mu**2 - torch.log(1+torch.exp(self.sigma))**2
            if KL_reg.shape[1] > 1:
                KL_reg = KL_reg.sum(axis=1)
            return output, -KL_reg
        else:
            return output


class RES_Classifier(nn.Module):
    def __init__(self, i_dim, cl_num, ch):
        super(RES_Classifier, self).__init__()
        self.i_dim = i_dim
        self.ch = ch
        self.cl_num = cl_num

        self.common_net = nn.Sequential(
            ConvBlock(i_dim, ch, upsample=False, downsample=True),
            ConvBlock(ch, ch, upsample=False, downsample=True),
            ConvBlock(ch, ch, upsample=False, downsample=True),
        )
        self.aux_c = nn.Linear(4 * 4 * ch, self.cl_num)

    def forward(self, input0):
        input = self.common_net(input0)
        input = input.view(-1, 4 * 4 * self.ch)
        output_c = self.aux_c(input)
        return output_c


class RES_AuxClassifier(nn.Module):
    def __init__(self, i_dim, cl_num, do_num, ch=64):
        super(RES_AuxClassifier, self).__init__()
        self.i_dim = i_dim
        self.ch = ch
        self.cl_num = cl_num
        self.do_num = do_num

        self.common_net = nn.Sequential(
            ConvBlock(i_dim, ch, upsample=False, downsample=True),
            ConvBlock(ch, ch, upsample=False, downsample=True),
            ConvBlock(ch, ch, upsample=False, downsample=True),
        )
        self.aux_c = nn.Linear(4 * 4 * ch, self.cl_num)
        self.aux_d = nn.Linear(4 * 4 * ch, self.do_num)
        self.aux_c_tw = nn.Linear(4 * 4 * ch, self.cl_num)
        self.aux_d_tw = nn.Linear(4 * 4 * ch, self.do_num)
        self.disc = nn.Linear(4 * 4 * ch, 1)
        self.cls = RES_Classifier(i_dim, cl_num, ch)

    def forward(self, input0):
        input = self.common_net(input0)
        input = input.view(-1, 4 * 4 * self.ch)
        output_c = self.aux_c(input)
        output_c_tw = self.aux_c_tw(input)
        output_d = self.aux_d(input)
        output_d_tw = self.aux_d_tw(input)
        output_disc = self.disc(input)
        output_cls = self.cls(input0)
        return output_c, output_c_tw, output_d, output_d_tw, output_cls, output_disc
