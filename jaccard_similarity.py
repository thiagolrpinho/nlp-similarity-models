from metaflow import FlowSpec, step, Parameter


def join_relative_folder_path(filename: str) -> str:
    """
    A convenience function to get the absolute path to a file in this
    data's directory. This allows the file to be launched from any
    directory.

    """
    import os

    path_to_data_folder = os.path.join(os.path.dirname(__file__), "data/")
    return os.path.join(path_to_data_folder, filename)


def calculate_jaccard_similarity(document_1: str, document_2: str) -> float:
    '''
    Jaccard similarity or intersection over union is defined as size of
    intersection divided by size of union of two sets. As we have already
    lemmatized the documents, we just have to transform those documents into
    sets and evaluate their intersections related to the compared sets.
    The greater the value, more words in common they have so they're more
    similar.
    '''
    words_set_1 = set(document_1.split())
    words_set_2 = set(document_2.split())
    words_in_common_set = words_set_1.intersection(words_set_2)

    count_words_in_common = len(words_in_common_set)
    count_different_words = len(words_set_1)+len(words_set_2)-len(words_in_common_set)
    if not count_different_words:
        # Let's avoid zero divisions
        jaccard_similarity = 1.0
    else:
        jaccard_similarity = float(count_words_in_common/count_different_words)
    return jaccard_similarity


class JaccardSimilarityFlow(FlowSpec):
    """
    Use Jaccard Similarity to Analyse Text Distances for ALEI Project

    1. Loading Data
        1.1 Sampling Data
    2. Treating Data
        2.1 Creating Text Pairs
    3. Jaccard Text Similarity
        3.1 Generating Heatmap for Text Similarities
    4. Evaluation
        4.1 Time Consumed
        4.2 Similarities Surface
        4.3 Using Similarities To Classify
    5. Bibliography
    """
    preprocessed_documents_data = Parameter(
        "preprocessed_documents_data",
        help="The path to preporcessed documents file.",
        default=join_relative_folder_path('data_preprocessed.parquet.gzip'))

    @step
    def start(self):
        """
        Load preprocessed documents

        """
        import pandas as pd
        from io import StringIO

        # Load the data set into a pandas dataframe.
        self.dataframe = pd.read_parquet(self.preprocessed_documents_data,
            columns=['text', 'process_id'])

        self.next(self.sampling)

    @step
    def sampling(self):
        """
        Samples a choosen number of rows

        """
        num_samples = 100
        sample_df = self.dataframe.sample(n=num_samples)
        self.sample_dataframe = sample_df.drop_duplicates(subset='process_id')
        self.next(self.creating_text_pairs)

    @step
    def creating_text_pairs(self):
        """
            Create a new dataframe with pair of documents. If the given
            input is N the output will be NxN long.
        """
        import pandas as pd

        text_pairs = []
        comparing_same_text = True
        for question_1_index, question_1_row in self.sample_dataframe.iterrows():
            question_1_text = question_1_row['text']
            question_1_id = question_1_row['process_id']
            for question_2_index, question_2_row in self.sample_dataframe.iterrows():
                is_same_index = question_1_index == question_2_index
                if not comparing_same_text and is_same_index:
                    continue
                question_2_text = question_2_row['text']
                question_2_id = question_2_row['process_id']
                text_pairs.append([
                    question_1_id,
                    question_1_text,
                    question_2_id,
                    question_2_text])

        self.pair_text_df = pd.DataFrame(
            text_pairs,
            columns=['question1_id', 'question1', 'question2_id', 'question2'])

        self.next(self.caculating_similarities)

    @step
    def caculating_similarities(self):
        """
        Measures jaccard similarity for all pair texts and sort their values
        for better visualisation.
        """
        import pandas as pd

        unique_ids = self.pair_text_df['question1_id'].unique()
        unique_ids_list = unique_ids.tolist()

        unique_ids_list.sort()

        distances_mapped = dict()
        for choosen_id in unique_ids_list:
            pair_text_ids = self.pair_text_df['question1_id'].values
            choosen_question_mask = pair_text_ids == choosen_id
            compared_df = self.pair_text_df[choosen_question_mask]

            compared_df.sort_values(by=['question2_id'], inplace=True)

            compared_ids_list = compared_df['question2_id'].to_list()
            if compared_ids_list != unique_ids_list:
                print("An error ocurred")
                break

            predictions_list = []
            for _, pair_text_row in compared_df.iterrows():
                row_distance = calculate_jaccard_similarity(
                    pair_text_row['question1'], pair_text_row['question2'])
                predictions_list.append(row_distance)
            distances_mapped[choosen_id] = predictions_list

        mapped_distances_df = pd.DataFrame.from_dict(
            distances_mapped, orient='index', columns=unique_ids_list)

        first_question_id = unique_ids_list[0]
        mapped_distances_df.sort_values(
            by=[first_question_id], ascending=False, inplace=True)
        self.mapped_distances_df = mapped_distances_df[mapped_distances_df.index]
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        Output data available on
            mapped_distances_df
        """
        pass


if __name__ == '__main__':
    JaccardSimilarityFlow()
