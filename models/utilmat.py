class UtilMat:
    '''
    Utility matrix abstraction for storing user - movie ratings
    '''

    def __init__(self, df, user_id='userid', item_id='movieid', rating_id='rating_id'):
        '''
        Accepts a dataframe and generates the utility matrix
        Arguments:
            df: pandas dataframe with ratings
        Format:
        userid - movieid - rating - timestamp
        '''
        # Utility matrix
        um = {}
        # Inverse utility matrix
        ium = {}
        # Bias for each user
        bx = {}
        cx = {}
        # Bias for each movie
        bi = {}
        ci = {}
        l = len(df)
        rating_sum = 0
        for i in range(l):
            user = df.loc[i, user_id]
            movie = df.loc[i, item_id]
            rating = df.loc[i, rating_id]
            rating_sum += rating
            if um.get(user):
                um[user][movie] = rating
            else:
                um[user] = {movie: rating}
            if ium.get(movie):
                ium[movie][user] = rating
            else:
                ium[movie] = {user: rating}
            if bx.get(user):
                bx[user] += rating
                cx[user] += 1
            else:
                bx[user] = rating
                cx[user] = 1
            if bi.get(movie):
                bi[movie] += rating
                ci[movie] += 1
            else:
                bi[movie] = rating
                ci[movie] = 1

        self.um = um
        self.ium = ium
        # Global average of ratings
        self.mu = rating_sum / l
        for user in bx:
            bx[user] = bx[user] / cx[user]
            bx[user] = bx[user] - self.mu
        for movie in bi:
            bi[movie] = bi[movie] / ci[movie]
            bi[movie] = bi[movie] - self.mu
        self.bx = bx
        self.bi = bi