from scipy.special import expit as logistic_sigmoid
from utils.constants import interrogative_words
from utils.helper import jaccard_index, spacy_model, get_heads, get_objects, get_roots, get_subjects, \
    get_non_alphanumeric_characters, filter_words_with_minimum_idf, geometric_mean_of_unigram_idfs, \
    is_subject_verb_inversion, naive_normalization, number_of_children, document_pos, relative_levenshtein_distance, \
    compare_compressed_size, get_all_lemmas, get_cosine_similarity, simple_document_filter, \
    relative_size_similarity


def spacy_process(entry):
    entry['question1_document'] = spacy_model(entry['question1'])
    entry['question2_document'] = spacy_model(entry['question2'])
    return entry


def simple_similarity(entry):
    entry['spacy_similarity_feature'] = naive_normalization(
        entry['question1_document'].similarity(entry['question2_document'])
    )
    return entry


def entity_sets_similarity(entry):
    entry['entities_similarity_feature'] = jaccard_index(
        set([entity.text for entity in entry['question1_document'].ents]),
        set([entity.text for entity in entry['question2_document'].ents])
    )
    return entry


def numbers_sets_similarity(entry):
    entry['numbers_similarity_feature'] = jaccard_index(
        set([word.lemma for word in entry['question1_document'] if word.like_num]),
        set([word.lemma for word in entry['question2_document'] if word.like_num])
    )
    return entry


def url_sets_similarity(entry):
    entry['url_similarity_feature'] = jaccard_index(
        set([word.lemma for word in entry['question1_document'] if word.like_url]),
        set([word.lemma for word in entry['question2_document'] if word.like_url])
    )
    return entry


def email_sets_similarity(entry):
    entry['email_similarity_feature'] = jaccard_index(
        set([word.lemma for word in entry['question1_document'] if word.like_email]),
        set([word.lemma for word in entry['question2_document'] if word.like_email])
    )
    return entry


def subject_sets_similarity(entry):
    entry['subjects_similarity_feature'] = jaccard_index(
        get_subjects(entry['question1_document']),
        get_subjects(entry['question2_document'])
    )
    return entry


def parse_roots_sets_similarity(entry):
    entry['roots_similarity_feature'] = jaccard_index(
        get_roots(entry['question1_document']),
        get_roots(entry['question2_document'])
    )
    return entry


def parse_heads_sets_similarity(entry):
    entry['heads_similarity_feature'] = jaccard_index(
        get_heads(entry['question1_document']),
        get_heads(entry['question2_document'])
    )
    return entry


def object_sets_similarity(entry):
    entry['objects_similarity_feature'] = jaccard_index(
        get_objects(entry['question1_document']),
        get_objects(entry['question2_document'])
    )
    return entry


def first_interrogative_matching(entry):
    interrogatives1 = [word.lemma for word in entry['question1_document'] if word.lemma_ in interrogative_words]
    interrogatives2 = [word.lemma for word in entry['question2_document'] if word.lemma_ in interrogative_words]

    match = False
    if interrogatives1 and interrogatives2:
        match = interrogatives1[0] == interrogatives2[0]
    elif not interrogatives1 and not interrogatives2:
        match = True

    entry['interrogative_match_feature'] = float(match)
    return entry


def non_alphanumeric_sets_similarity(entry):
    entry['non_alphanumeric_similarity_feature'] = jaccard_index(
        set(get_non_alphanumeric_characters(entry['question1'])),
        set(get_non_alphanumeric_characters(entry['question2']))
    )
    return entry


def unigram_idf_cutoff_similarity(entry):
    for cutoff in [1, 5, 7.5, 10, 12.5, 15]:
        feature_name = 'unigram_idf_cutoff_similarity_{cutoff}_feature'.format(cutoff=cutoff)
        entry[feature_name] = jaccard_index(
            filter_words_with_minimum_idf(entry['question1_document'], cutoff),
            filter_words_with_minimum_idf(entry['question2_document'], cutoff)
        )
    return entry


def unigram_idf_mean_difference(entry):
    entry['unigram_idf_mean_difference_feature'] = logistic_sigmoid(
        abs(
            geometric_mean_of_unigram_idfs(entry['question1_document']) -
            geometric_mean_of_unigram_idfs(entry['question2_document'])
        )
    )
    return entry


def subject_verb_inversion_similarity(entry):
    entry['subject_verb_inversion_similarity_feature'] = float(
        is_subject_verb_inversion(entry['question1_document']) ==
        is_subject_verb_inversion(entry['question2_document'])
    )
    return entry


def number_of_children_similarity(entry):
    entry['number_of_children_similarity_5_feature'] = relative_levenshtein_distance(
        number_of_children(entry['question1_document'])[:5],
        number_of_children(entry['question2_document'])[:5]
    )
    return entry


def document_pos_cutoff_similarity(entry):
    for cutoff in [3, 5, 7, 10]:
        feature_name = 'document_pos_similarity_{cutoff}_feature'.format(cutoff=cutoff)
        entry[feature_name] = relative_levenshtein_distance(
            document_pos(entry['question1_document'])[:cutoff],
            document_pos(entry['question2_document'])[:cutoff]
        )
    return entry


def compression_size_reduction_ratio(entry):
    entry['compression_ratio_feature'] = naive_normalization(
        compare_compressed_size(
            entry['question1'],
            entry['question2']
        )
    )
    return entry


def lemma_edit_distance(entry):
    entry['lemma_edit_distance_feature'] = relative_levenshtein_distance(
        get_all_lemmas(entry['question1_document']),
        get_all_lemmas(entry['question2_document'])
    )
    return entry


def first_word_similarity(entry):
    try:
        entry['first_word_similarity_feature'] = naive_normalization(
            entry['question1_document'][0].similarity(entry['question2_document'][0])
        )
    except IndexError:
        entry['first_word_similarity_feature'] = 0.0
    return entry


def last_word_similarity(entry):
    try:
        entry['last_word_similarity_feature'] = naive_normalization(
            entry['question1_document'][-1].similarity(entry['question2_document'][-1])
        )
    except IndexError:
        entry['last_word_similarity_feature'] = 0.0
    return entry


def filtered_cosine_similarity(entry):
    entry['filtered_cosine_similarity_feature'] = naive_normalization(
        get_cosine_similarity(
            simple_document_filter(
                document=entry['question1_document'],
                use_out_of_vocabulary=False,
                use_stopwords=False,
                use_punctuation=False
            ),
            simple_document_filter(
                document=entry['question2_document'],
                use_out_of_vocabulary=False,
                use_stopwords=False,
                use_punctuation=False
            )
        )
    )
    return entry


def question_length_similarity(entry):
    entry['question_length_similarity_feature'] = relative_size_similarity(
        entry['question1_document'],
        entry['question2_document']
    )
    return entry
