import csv


def read_dataset_from_csv(file_path):
    """
    Reads .csv dataset and yields line by line.
    :param file_path: Dataset file path.
    :return: Nothing.
    """
    with open(file_path, 'r') as input_file:
        reader = csv.DictReader(input_file)
        for line in reader:
            yield line


def process_line(line_dict):
    """
    Reads the specific format used in the Quora Question Pairs Kaggle competition
    (https://www.kaggle.com/c/quora-question-pairs)
    :param line_dict: Data point in the key-value format.
    :return: Normalized data point in the same format.
    """
    data = {
        'id': int(line_dict['test_id']) if 'test_id' in line_dict else int(line_dict['id']),
        'question1': line_dict['question1'],
        'question2': line_dict['question2']
    }
    if 'is_duplicate' in line_dict:
        data['is_duplicate'] = True if line_dict['is_duplicate'] == '1' else False
    return data


def assert_valid_input(entry):
    """
    Asserts that the input dataset is in correct format and has all the necessary attributes. Generally used at the
    beginning of the pipeline. Tightly coupled with the used dataset.
    :param entry: Data point.
    :return: Unchanged data point if it is valid.
    """
    assert isinstance(entry, dict)
    assert 'question1' in entry
    assert 'question2' in entry
    assert isinstance(entry['question1'], str)
    assert isinstance(entry['question2'], str)
    return entry


def feature_extraction(entry):
    """
    Extracts only the features, data point ids and the target classification (if there is one). Generally used at the
    end of the pipeline. Tightly coupled with the used dataset.
    :param entry: Data point with all the features extracted.
    :return: Filtered data point.
    """
    return {
        key: value for key, value in entry.items()
        if key.endswith('_feature') or key == 'id' or key == 'is_duplicate'
        }


def write_results_to_csv(results, output_file_path):
    """
    Writes extracted features to a new .csv file.
    :param results: Feature extraction results, either a list or an iterable.
    :param output_file_path: File path of the output file.
    :return: Nothing.
    """
    results_iterable = iter(results)
    try:
        first_row = next(results_iterable)
    except StopIteration:
        return

    with open(output_file_path) as output_file:
        field_names = sorted(list(first_row.keys()))

        csv_writer = csv.DictWriter(output_file, fieldnames=field_names)
        csv_writer.writeheader()
        csv_writer.writerow(first_row)

        field_names_set = set(field_names)

        for row in results_iterable:
            assert set(row.keys()) == field_names_set
            csv_writer.writerow(row)
