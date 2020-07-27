import numpy as np
import math
from models.abstract import AbstractModel
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from scipy.optimize import fmin_l_bfgs_b


class LatentFactor(object):
    def __init__(self, n=5, learning_rate=0.001, lmbda=0.05, verbose=False):
        '''

        :param n: int, default is 5
            Number of latent factors
        :param learning_rate: float, default is 0.001
            Learning rate for Stochastic Gradient Descent
        :param lmbda: float, default is 0.05
            Regularization constant
        :param verbose: bool, default is False
            Controls the verbosity of the code
        '''

        self.n = n
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.verbose = verbose

        # Initializing Latent factor matrices
        self.P = (np.random.random((6040 + 1, self.n)) * 2 - 1) / self.n * 10
        self.Q = (np.random.random((self.n, 3952 + 1)) * 2 - 1) / self.n * 10

        # Stores model history
        self.history = {'train_loss': [], 'val_loss': []}

    def fit(self, utilmat, iters, val_utilmat):
        '''
        Helper function:
            Implements SGD with learnable bias
        '''
        P = self.P
        Q = self.Q
        um = utilmat.um
        # gloabal average rating
        mu = utilmat.mu
        bx = np.random.random(P.shape[0]) * 2 - 1
        bi = np.random.random(Q.shape[1]) * 2 - 1
        # Error function:
        # exi = rxi - mu - bx - bi - px.T * qi
        for i in range(iters):
            for user in um:
                for movie in um[user]:
                    # Actual rating
                    rxi = um[user][movie]
                    px = P[user, :].reshape(-1, 1)
                    qi = Q[:, movie].reshape(-1, 1)
                    # Calculate error
                    exi = rxi - mu - bx[user] - bi[movie] - np.dot(px.T, qi)
                    # Update parameters
                    px = px + self.learning_rate * (exi * qi - self.lmbda * px)
                    qi = qi + self.learning_rate * (exi * px - self.lmbda * qi)
                    bx[user] += self.learning_rate * (exi - self.lmbda * bx[user])
                    bi[movie] += self.learning_rate * (exi - self.lmbda * bi[movie])
                    px = px.reshape(-1)
                    qi = qi.reshape(-1)
                    P[user, :] = px
                    Q[:, movie] = qi
            # Saving state after each iteration
            self.P = P
            self.Q = Q
            self.bx = bx
            self.bi = bi
            if self.verbose:
                print('Iteration {}'.format(i + 1))
                tloss = self.calc_loss(utilmat)
                print('Training Loss: ', tloss)
                self.history['train_loss'].append(tloss)
                if val_utilmat:
                    vloss = self.calc_loss(val_utilmat)
                    print('Validation Loss: ', vloss)
                    self.history['val_loss'].append(vloss)

    def predict(self, user, movie):
        '''
        Finds predicted rating for the user-movie pair
        '''
        mu = self.utilmat.mu
        bx = self.utilmat.bx
        bi = self.utilmat.bi
        # Baseline prediction
        bxi = mu + bx[user] + bi[movie]
        bxi += np.dot(self.P[user, :], self.Q[:, movie])
        return bxi

    def calc_loss(self, utilmat, get_mae=False):
        '''
        Finds the RMSE loss (optional MAE)
        '''
        um = utilmat.um
        mu = utilmat.mu
        bx = utilmat.bx
        bi = utilmat.bi
        cnt = 0
        rmse = 0
        mae = 0
        for user in um:
            for movie in um[user]:
                y = um[user][movie]
                yhat = mu + bx[user] + bi[movie] + np.dot(self.P[user, :], self.Q[:, movie])
                rmse += (y - yhat) ** 2
                mae += abs(y - yhat)
                cnt += 1
        rmse /= cnt
        mae /= cnt
        rmse = math.sqrt(rmse)
        if get_mae:
            return rmse, mae
        return rmse


class LatentFactorModel(object):
    def __init__(self, n=5, lamb=0.01, verbose=False):
        '''

        :param n: int, default is 5
            Number of latent factors
        :param lmbda: float, default is 0.01
            Regularization constant
        :param verbose: bool, default is False
            Controls the verbosity of the code
        '''

        self.n = n
        self.lamb = lamb
        self.verbose = verbose

        # # Initializing Latent factor matrices
        # self.P = (np.random.random((6040 + 1, self.n)) * 2 - 1) / self.n * 10
        # self.Q = (np.random.random((self.n, 3952 + 1)) * 2 - 1) / self.n * 10

        # Stores model history
        self.history = {'loss': [], 'val_loss': []}

    def _prepare_data(self, df):
        reviewsPerUser = df.groupby(self.user_id)[self.item_id].count()
        reviewsPerItem = df.groupby(self.item_id)[self.user_id].count()
        self.nUsers = len(reviewsPerUser)
        self.nItems = len(reviewsPerItem)
        self.users = list(reviewsPerUser.keys())
        self.items = list(reviewsPerItem.keys())

        # containers to store user and item biases (beta_u and beta_i)
        self.userBiases = defaultdict(float)
        self.itemBiases = defaultdict(float)
        return

    def fit(self, df, user_id='userid', item_id='movieid', rating_id='rating'):
        self.user_id = user_id
        self.item_id = item_id
        self.rating_id = rating_id
        self.y = df[rating_id]
        self._prepare_data(df)
        self.dataset = df.to_dict('records')

        #ToDo: provide better initializations
        self.alpha = np.mean(self.y)
        init = [self.alpha] + [0.0] * (self.nUsers + self.nItems)

        # optimize
        res = fmin_l_bfgs_b(self.cost, init, self.jac, args=[self.lamb], maxiter=2500)
        flag = res[2]['warnflag']
        if flag != 0:
            print('W: Optimization has not converged.')
        return

    def _predict(self, user, item):
        rec = self.alpha + self.userBiases[user] + self.itemBiases[item]
        return rec

    def unpack(self, theta):
        self.alpha = theta[0]
        self.userBiases = dict(zip(self.users, theta[1:self.nUsers + 1]))
        self.itemBiases = dict(zip(self.items, theta[1 + self.nUsers:]))
        return

    def cost(self, theta, lamb):
        self.unpack(theta)
        y_pred = [self._predict(d[self.user_id], d[self.item_id]) for d in self.dataset]
        cost = mean_squared_error(self.y, y_pred)
        if self.verbose:
            print('MSE = ' + str(cost))
        for u in self.userBiases:
            cost += lamb * self.userBiases[u] ** 2
        for i in self.itemBiases:
            cost += lamb * self.itemBiases[i] ** 2
        self.history['loss'].append(cost)
        return cost

    def jac(self, theta, lamb):
        self.unpack(theta)
        N = len(self.y)
        dalpha = 0
        dUserBiases = defaultdict(float)
        dItemBiases = defaultdict(float)
        for d in self.dataset:
            u, i = d[self.user_id], d[self.item_id]
            pred = self._predict(u, i)
            diff = pred - d[self.rating_id]
            dalpha += 2 / N * diff
            dUserBiases[u] += 2 / N * diff
            dItemBiases[i] += 2 / N * diff
        for u in self.userBiases:
            dUserBiases[u] += 2 * lamb * self.userBiases[u]
        for i in self.itemBiases:
            dItemBiases[i] += 2 * lamb * self.itemBiases[i]
        dtheta = [dalpha] + [dUserBiases[u] for u in self.users] + [dItemBiases[i] for i in self.items]
        dtheta = np.array(dtheta)
        return dtheta
