import numpy as np

def main():
    a = [np.array([1,2]),1]
    b = [np.array([3,4]),2]
    c = [np.array([5,6]),3]
    d = [np.array([7,8]),4]

    # a = [[1,2],1]
    # b = [[3,4],2]
    # c = [[5,6],3]
    # d = [[7,8],4]


    li = []
    li.append(a)
    li.append(b)
    li.append(c)
    li.append(d)

    liarr = np.array(li)

    costs_arr = liarr[:,1]
    cost_rhoth_q = np.quantile(costs_arr, q=0.5)
    indexesUq_cost = [index for index in range(4) if costs_arr[index] <= cost_rhoth_q]

    elite_samples = liarr[indexesUq_cost, 0]
    elite_samples1 = [elite_samples[i] for i in range(len(elite_samples)) ]

    aaa = 1
if __name__ == '__main__':
    main()