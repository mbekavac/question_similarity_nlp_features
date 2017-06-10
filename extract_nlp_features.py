import sys

from pipey import apply_pipeline, modifier

from nlp_features import spacy_process, simple_similarity, entity_sets_similarity, \
    numbers_sets_similarity, subject_sets_similarity, parse_roots_sets_similarity, parse_heads_sets_similarity, \
    object_sets_similarity, first_interrogative_matching, non_alphanumeric_sets_similarity, \
    unigram_idf_cutoff_similarity, unigram_idf_mean_difference, subject_verb_inversion_similarity, \
    number_of_children_similarity, document_pos_cutoff_similarity, compression_size_reduction_ratio, \
    email_sets_similarity, filtered_cosine_similarity, url_sets_similarity, first_word_similarity, \
    last_word_similarity, lemma_edit_distance, question_length_similarity
from utils.dataset_utils import read_dataset_from_csv, process_line, assert_valid_input, feature_extraction, \
    write_results_to_csv

nlp_pipeline = [
    (process_line, modifier.map),
    (assert_valid_input, modifier.map),
    (spacy_process, modifier.map),
    (simple_similarity, modifier.map),
    (filtered_cosine_similarity, modifier.map),
    (entity_sets_similarity, modifier.map),
    (numbers_sets_similarity, modifier.map),
    (email_sets_similarity, modifier.map),
    (url_sets_similarity, modifier.map),
    (subject_sets_similarity, modifier.map),
    (parse_roots_sets_similarity, modifier.map),
    (parse_heads_sets_similarity, modifier.map),
    (object_sets_similarity, modifier.map),
    (first_interrogative_matching, modifier.map),
    (non_alphanumeric_sets_similarity, modifier.map),
    (unigram_idf_cutoff_similarity, modifier.map),
    (unigram_idf_mean_difference, modifier.map),
    (subject_verb_inversion_similarity, modifier.map),
    (number_of_children_similarity, modifier.map),
    (document_pos_cutoff_similarity, modifier.map),
    (compression_size_reduction_ratio, modifier.map),
    (first_word_similarity, modifier.map),
    (last_word_similarity, modifier.map),
    (lemma_edit_distance, modifier.map),
    (question_length_similarity, modifier.map),
    (feature_extraction, modifier.map)
]


def main(input_file_path, output_file_path, pipeline):
    dataset = read_dataset_from_csv(input_file_path)
    results = apply_pipeline(dataset, pipeline)
    write_results_to_csv(results, output_file_path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], nlp_pipeline)
