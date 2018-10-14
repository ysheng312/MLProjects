import feedparser
import ssl
from NaiveBayes.text_classifier_nb import *


def calc_most_freq(vocab_list, full_test):
    import operator
    freq_dict = {}
    for token in vocab_list:
        freq_dict[token] = full_test.count(token)
    sorted_freq = sorted(freq_dict.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_freq[:30]


def local_words(feed1, feed0):
    doc_list = []
    class_list = []
    full_text = []
    min_len = min(len(feed1['entries']), len(feed0['entries']))
    for i in range(min_len):
        word_list = text_parse(feed1['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(feed0['entries'][i]['summary'])
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)

    vocab_list = create_vocab_list(doc_list)
    top_30_words = calc_most_freq(vocab_list, full_text)
    for pair_w in top_30_words:
        if pair_w[0] in vocab_list:
            vocab_list.remove(pair_w[0])
    training_set = list(range(2 * min_len))
    test_set = []
    for i in range(20):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:
        train_mat.append(bag_of_words_2_vec_mn(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb_0(array(train_mat), array(train_classes))
    error_count = 0
    for doc_index in test_set:
        word_vector = bag_of_words_2_vec_mn(vocab_list, doc_list[doc_index])
        if classify_nb(array(word_vector), p0_v, p1_v, p_spam) != class_list[doc_index]:
            error_count += 1
    print('the error rate is: ', float(error_count)/len(test_set))
    return vocab_list, p0_v, p1_v


def get_top_words(ny, bd):
    vocab_list, p0_v, p1_v = local_words(ny, bd)
    top_ny = []
    top_bd = []
    for i in range(len(p0_v)):
        if p0_v[i] > - 6.0:
            top_bd.append((vocab_list[i], p0_v[i]))
        if p1_v[i] > - 6.0:
            top_ny.append((vocab_list[i], p1_v[i]))
    sorted_bd = sorted(top_bd, key=lambda pair: pair[1], reverse=True)
    print("BD**BD**BD**BD**BD**BD**BD**BD**BD**BD**BD**BD**BD**BD**")
    for item in sorted_bd:
        print(item[0])
    sorted_ny = sorted(top_ny, key=lambda pair: pair[1], reverse=True)
    print("NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**")
    for item in sorted_ny:
        print(item[0])


def main():
    url0 = 'http://boulder.craigslist.org/search/act?format=rss'
    url1 = 'http://newyork.craigslist.org/search/act?format=rss'

    if hasattr(ssl, '_create_unverified_context'):
        ssl._create_default_https_context = ssl._create_unverified_context

    bd = feedparser.parse(url0)
    ny = feedparser.parse(url1)

    vocab_list, p_bd, p_ny = local_words(ny, bd)

    get_top_words(ny, bd)


if __name__ == '__main__':
    main()
