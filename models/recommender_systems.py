import numpy as np
from collections import defaultdict
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from .utilmat import UtilMat


class LatentFactor(object):
    def __init__(self, K=5, learning_rate=0.001, lamb=0.05, verbose=False):
        '''

        :param K: int, default is 5
            Number of latent factors
        :param learning_rate: float, default is 0.001
            Learning rate for Stochastic Gradient Descent
        :param lamb: float, default is 0.05
            Regularization constant
        :param verbose: bool, default is False
            Controls the verbosity of the code
        '''

        self.K = K
        self.learning_rate = learning_rate
        self.lamb = lamb
        self.verbose = verbose

        # Initializing Latent factor matrices
        self.P = (np.random.random((6040 + 1, self.K)) * 2 - 1) / self.K * 10
        self.Q = (np.random.random((self.K, 3952 + 1)) * 2 - 1) / self.K * 10

        # Stores model history
        self.history = {'loss': [], 'val_loss': []}

    def fit(self, df, df_val, user_id='userid', item_id='movieid', rating_id='rating', iters=10):
        '''
        Helper function:
            Implements SGD with learnable bias
        '''

        utilmat = UtilMat(df.reset_index(drop=True, inplace=True), user_id, item_id, rating_id)

        if df_val is not None:
            val_utilmat = UtilMat(df_val.reset_index(drop=True, inplace=True), user_id, item_id, rating_id)

        P = self.P
        Q = self.Q
        um = utilmat.um
        # global average rating
        mu = utilmat.mu
        bx = np.random.random(P.shape[0]) * 2 - 1
        bi = np.random.random(Q.shape[1]) * 2 - 1
        # Error function:
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
                    px = px + self.learning_rate * (exi * qi - self.lamb * px)
                    qi = qi + self.learning_rate * (exi * px - self.lamb * qi)
                    bx[user] += self.learning_rate * (exi - self.lamb * bx[user])
                    bi[movie] += self.learning_rate * (exi - self.lamb * bi[movie])
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
                self.history['loss'].append(tloss)
                if df_val is not None:
                    vloss = self.calc_loss(val_utilmat)
                    print('Validation Loss: ', vloss)
                    self.history['val_loss'].append(vloss)

    def predict(self, user, movie):
        mu = self.utilmat.mu
        bx = self.utilmat.bx
        bi = self.utilmat.bi
        # Baseline prediction
        bxi = mu + bx[user] + bi[movie]
        bxi += np.dot(self.P[user, :], self.Q[:, movie])
        return bxi

    def calc_loss(self, utilmat):
        um = utilmat.um
        mu = utilmat.mu
        bx = utilmat.bx
        bi = utilmat.bi
        cnt = 0
        mse = 0
        for user in um:
            for movie in um[user]:
                y = um[user][movie]
                yhat = mu + bx[user] + bi[movie] + np.dot(self.P[user, :], self.Q[:, movie])
                mse += (y - yhat) ** 2
                cnt += 1
        mse /= cnt
        return mse


class LatentFactorModel(object):
    def __init__(self, K=5, lamb=0.01, verbose=False):
        '''

        :param K: int, default is 5
            Number of latent factors
        :param lamb: float, default is 0.01
            Regularization constant
        :param verbose: bool, default is False
            Controls the verbosity of the code
        '''

        self.K = K
        self.lamb = lamb
        self.verbose = verbose

        # Stores model history
        self.history = {'loss': [], 'val_loss': []}

    def initialize_data(self, df, user_id='userid', item_id='movieid', rating_id='rating'):
        avgReviewsPerUser = df.groupby(user_id)[rating_id].mean()
        avgReviewsPerItem = df.groupby(item_id)[rating_id].mean()
        self.nUsers = len(avgReviewsPerUser)
        self.nItems = len(avgReviewsPerItem)
        self.users = list(avgReviewsPerUser.keys())
        self.items = list(avgReviewsPerItem.keys())

        # containers to store user and item biases (beta_u and beta_i)
        self.userBiases = defaultdict(float)
        self.itemBiases = defaultdict(float)

        # containers to store gamma vectors for users and items
        self.userGamma = {}
        self.itemGamma = {}

        # initializations
        alpha = np.mean(df[rating_id])
        beta_u = list((avgReviewsPerUser - alpha).values)
        beta_i = list((avgReviewsPerItem - alpha).values)
        gamma_u = list(np.random.random(self.K * self.nUsers) * 2 - 1)
        gamma_i = list(np.random.random(self.K * self.nItems) * 2 - 1)
        self.init = [alpha] + beta_u + beta_i + gamma_u + gamma_i

        self.num_params = 1 + self.nUsers + self.nItems + self.K * self.nUsers + self.K * self.nItems
        self.N = len(df)

        return

    def fit(self, df, user_id='userid', item_id='movieid', rating_id='rating'):
        self.user_id = user_id
        self.item_id = item_id
        self.rating_id = rating_id
        self.y = df[rating_id]
        self.dataset = df.to_dict('records')
        self.initialize_data(df, user_id, item_id, rating_id)

        if self.verbose:
            iprint = 1
        else:
            iprint = -1
        self.opt = minimize(self.cost, x0=self.init, jac=self.jac, args=(self.lamb), method='L-BFGS-B',
                            options={'maxiter': 2500, 'iprint': iprint})
        if self.opt['success'] is False:
            print('W: Optimization has not converged.')
        return

    @staticmethod
    def inner(x, y):
        return sum([a * b for a, b in zip(x, y)])

    def _predict(self, user, item):
        rec = self.alpha + self.userBiases[user] + self.itemBiases[item] + self.inner(self.userGamma[user],
                                                                                      self.itemGamma[item])
        return rec

    def unpack(self, theta):
        index = 0
        self.alpha = theta[0]
        index += 1
        self.userBiases = dict(zip(self.users, theta[index:index + self.nUsers]))
        index += self.nUsers
        self.itemBiases = dict(zip(self.items, theta[index:index + self.nItems]))
        index += self.nItems
        for u in self.users:
            self.userGamma[u] = theta[index:index + self.K]
            index += self.K
        for i in self.items:
            self.itemGamma[i] = theta[index:index + self.K]
            index += self.K
        return

    def cost(self, theta, lamb):
        self.unpack(theta)
        y_pred = [self._predict(d[self.user_id], d[self.item_id]) for d in self.dataset]
        # cost = mean_squared_error(y, y_pred)
        cost = np.sum((self.y-y_pred)**2)
        self.history['loss'].append(cost/self.N)
        if self.verbose:
            print('MSE = ' + str(cost/self.N))
        for u in self.userBiases:
            cost += lamb * self.userBiases[u] ** 2
            for k in range(self.K):
                cost += lamb * self.userGamma[u][k] ** 2
        for i in self.itemBiases:
            cost += lamb * self.itemBiases[i] ** 2
            for k in range(self.K):
                cost += lamb * self.itemGamma[i][k] ** 2
        return cost

    def jac(self, theta, lamb):
        self.unpack(theta)
        N = 1#len(self.y)
        dalpha = 0
        dUserBiases = defaultdict(float)
        dItemBiases = defaultdict(float)
        dUserGamma = {}
        dItemGamma = {}
        for u in self.users:
            dUserGamma[u] = [0.0] * self.K
        for i in self.items:
            dItemGamma[i] = [0.0] * self.K
        for d in self.dataset:
            u, i = d[self.user_id], d[self.item_id]
            pred = self._predict(u, i)
            diff = pred - d[self.rating_id]
            dalpha += 2 / N * diff
            dUserBiases[u] += 2 / N * diff
            dItemBiases[i] += 2 / N * diff
            for k in range(self.K):
                dUserGamma[u][k] += 2 / N * self.itemGamma[i][k] * diff
                dItemGamma[i][k] += 2 / N * self.userGamma[u][k] * diff
        for u in self.userBiases:
            dUserBiases[u] += 2 * lamb * self.userBiases[u]
            for k in range(self.K):
                dUserGamma[u][k] += 2 * lamb * self.userGamma[u][k]
        for i in self.itemBiases:
            dItemBiases[i] += 2 * lamb * self.itemBiases[i]
            for k in range(self.K):
                dItemGamma[i][k] += 2 * lamb * self.itemGamma[i][k]
        dtheta = [dalpha] + [dUserBiases[u] for u in self.users] + [dItemBiases[i] for i in self.items]
        for u in self.users:
            dtheta += dUserGamma[u]
        for i in self.items:
            dtheta += dItemGamma[i]
        dtheta = np.array(dtheta)
        return dtheta
