import numpy as np

def gradient_descent(x,y):
    m_cur=0
    b_cur=0
    itr=1000
    n = len(x)
    learning_rate = 0.001 #is tweaked for correct result
    for i in range(itr):
        y_predicted = m_cur * x + b_cur
        cost = (1/n) * sum(np.square(y-y_predicted))
        m_par_deriv = -(2/n) *  sum(x *(y-y_predicted))
        b_par_deriv = -(2/n) *  sum(y-y_predicted)
        m_cur = m_cur - learning_rate * m_par_deriv
        b_cur = b_cur - learning_rate * b_par_deriv
        print(f"m:{m_cur} b:{b_cur} i:{i} | cost:{cost}")
    pass


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
gradient_descent(x,y)