from numpy import *


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea',
                    'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him',
                    'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute',
                    'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how',
                    'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])
    # Remove duplicate words
    for document in data_set:
        vocab_set = vocab_set | set(document)
    return list(vocab_set)


def bag_of_words_2_vec_mn(vocab_list, input_set):
    return_vec = [0] * len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


def train_nb_0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category)/float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)
    p0_denom = 2.0
    p1_denom = 2.0

    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    p1_vect = log(p1_num / p1_denom)     # change to log()
    p0_vect = log(p0_num / p0_denom)     # change to log()
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec_2_classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec_2_classify * p1_vec) + log(p_class1)
    p0 = sum(vec_2_classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


def testing_nb():
    list_of_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_of_posts)
    train_mat = []
    for postin_doc in list_of_posts:
        train_mat.append(bag_of_words_2_vec_mn(my_vocab_list, postin_doc))
    p0_v, p1_v, p_ab = train_nb_0(array(train_mat), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(bag_of_words_2_vec_mn(my_vocab_list, test_entry))
    print(test_entry, 'classified as ', classify_nb(this_doc, p0_v, p1_v, p_ab))
    test_entry = ['stupid', 'garbage']
    this_doc = array(bag_of_words_2_vec_mn(my_vocab_list, test_entry))
    print(test_entry, 'classified as: ', classify_nb(this_doc, p0_v, p1_v, p_ab))


def main():
    testing_nb()


if __name__ == '__main__':
    main()
