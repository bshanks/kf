from libcpp.vector cimport vector

cdef extern from "kf.cpp":
    cdef cppclass KF[size_t]:
        KF(double,  vector[double])
        void update(double)
        double call(double,  vector[double], double)
        double predict( vector[double])



cdef class PyKf:
    cdef KF[size_t] *kf

    def __cinit__(self, double beta = 0.99, delta = [.99]):
        cdef vector[double] delta_vec
        
        for i in range(len(delta)):
            delta_vec.push_back(delta[i])
        self.kf = new KF(beta, delta_vec)

    def __dealloc__(self):
        del self.kf

    def __call__(self, dependent_observation, independents_observations):
        cdef double& yVar
        cdef vector[double] independents_observations_vec

        for i in range(len(independents_observations)):
            independents_observations_vec.push_back(independents_observations[i])
        return self.kf.call(dependent_observation, independents_observations_vec, yVar)
