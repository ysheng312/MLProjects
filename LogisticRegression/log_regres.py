from numpy import *


def load_data_set():
    data_mat = []
    label_mat = []
    fr = open('test_set.txt')
    for line in fr.readlines():
        line_arr = line.strip().split()
        data_mat.append([1.0, float(line_arr[0]), float(line_arr[1])])
        label_mat.append(int(line_arr[2]))
    return data_mat, label_mat


def sigmoid(in_x):
    from numpy.ma import exp
    return 1.0 / (1 + exp(-in_x))


def grad_ascent(data_mat_in, class_labels):
    data_matrix = mat(data_mat_in)
    label_mat = mat(class_labels).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001
    max_cycles = 500
    weights = ones((n, 1))

    for k in range(max_cycles):
        h = sigmoid(data_matrix * weights)
        error = (label_mat - h)
        weights = weights + alpha * data_matrix.transpose() * error
    return weights


def stoc_grad_ascent_1(data_matrix, class_labels, num_iter=150):
    m, n = shape(data_matrix)
    weights = ones(n)
    for j in range(num_iter):
        data_index = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.0001
            rand_index = int(random.uniform(0, len(data_index)))
            h = sigmoid(sum(data_matrix[rand_index] * weights))
            error = class_labels[rand_index] - h
            weights = weights + alpha * error * data_matrix[rand_index]
            del(data_index[rand_index])
    return weights


def plot_best_fit(weights):
    import matplotlib.pyplot as plt
    data_mat, label_mat = load_data_set()
    data_arr = array(data_mat)
    n = shape(data_arr)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(label_mat[i]) == 1:
            xcord1.append(data_arr[i, 1])
            ycord1.append(data_arr[i, 2])
        else:
            xcord2.append(data_arr[i, 1])
            ycord2.append(data_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def classify_vector(in_x, weights):
    prob = sigmoid(sum(in_x * weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0


def colic_test():
    fr_train = open('horse_colic_training.txt')
    fr_test = open('horse_colic_test.txt')
    training_set = []
    training_labels = []
    for line in fr_train.readlines():
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        training_set.append(line_arr)
        training_labels.append(float(curr_line[21]))
    training_weights = stoc_grad_ascent_1(array(training_set), training_labels, 1000)
    error_count = 0
    num_test_vec = 0.0
    for line in fr_test.readlines():
        num_test_vec += 1.0
        curr_line = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(curr_line[i]))
        if int(classify_vector(array(line_arr), training_weights)) != int(curr_line[21]):
            error_count += 1
    error_rate = (float(error_count)/num_test_vec)
    print("the error rate of this test is: %f" % error_rate)
    return error_rate


def multi_test():
    num_tests = 10
    error_sum = 0.0
    for k in range(num_tests):
        error_sum += colic_test()
    print("after %d iterations the average error rate is: %f " % (num_tests, error_sum/float(num_tests)))


def main():
    # data_arr, label_mat = load_data_set()
    # weights = stoc_grad_ascent_1(array(data_arr), label_mat)
    # plot_best_fit(weights)
    multi_test()


if __name__ == '__main__':
    main()
