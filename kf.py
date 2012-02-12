from numpy import array, diag, sqrt, Inf, zeros, dot, ones, outer, prod, isinf


"""
    Configuration:
      beta, discount factor

    Generative Model:
      Sigma_{t+1} = Sigma_t "+ Wishart",  "+" => conv. with Sing MV Beta 
      m_{t+1} = m_t + Omega_t,  Omega_t ~ N(0, W_t, Sigma_t)
      y_t = x_t m_t + epsilon_t,  epsilon_t ~ N(0, Sigma_t)
"""

"""
  double beta; // Coeffient rate of change adaptiveness 
  double k; // calculated from beta
  diagonal_matrix<double> Delta_root_inv; // Coefficient rate of change

  double yPred; // return value from previous timestep, for updating
  vector<double> Kprev; // K from previous timestep, for updating

  vector<double> m; // estimated regression coefficient vector
  matrix<double> P; // covariance matrix of (m_true - m)
  matrix<double> R; // covariance matrix of (m_true - m-1)
  vector<double> K; // kalman gain, optimal weight for new update
  double Q;         // forecast variance Var(y), scalar
  double S;         // Wishart mean covariance of Sigma => P
"""



class Kf:
    """
    beta is the discount factor
    delta is the discount factor for each input variable
    m is the initial guess as to the regression coefficients for each input variable
    """
    def __init__(self, beta=.99, delta = array([.99]), m=array([1.0])):
        p = len(delta)

        # Hyperparameters 
        self.beta = beta
        self.k = (beta*(1-p)+p)/(beta*(2-p)+p-1);
        self.delta_root_inv = diag(sqrt(delta))
   
        # Priors

        self.S = 1
        self.P = 1000*diag(ones(p))
        #self.m = zeros(p)
        self.m = m
        self.yPred = Inf
        
        #self.R = zeros([p,p])
        #self.K = zeros(p)

    def not_first_run(self):
        return not isinf(self.yPred)

    def predict(self, x):
        return dot(x,self.m)

    def confidence(self):
        #print self.Q
        #print self.S
        
        return self.Q*(1-self.beta)*self.S/(self.k*(3*self.beta-2))


    """
  Inputs:
  y from previous timestep, for updating
  x current observation
  yVar reference to second return value, predicted y variance

  Output:
  predicted y value
    """

    def __call__(self, y, x):
        self.estimate(x)
        if self.not_first_run():
            self.update(y)

        self.Kprev = self.K.copy()
        self.yPred = self.predict(x)
        self.yVar = self.confidence()

        return (self.yPred, self.yVar)
        
    def estimate(self, x):
        R = dot(self.delta_root_inv, dot(self.P, self.delta_root_inv))
        #print x
        #print R
        #print dot(x.T,R)
       
        self.Q = dot(dot(x.T,R), x) + 1
        #print self.Q
        self.K = dot(R, x) / self.Q
        #print self.K
        self.P = R - outer(self.K,self.K) * self.Q
        #print self.P

    def update(self, y):
        e = y - self.yPred
        #print self.k
        self.S = self.S / self.k + e*e/self.Q
        self.m += self.Kprev * e
    

