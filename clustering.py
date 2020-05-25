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


class ClusteringKMeans(FlowSpec):
    """
    Develop a clustering algorithm using k-means and represent those cluster
    distances for ALEI project

    Summary:
        1. Start
            1.1 Sampling Data
        2. Counting and Vectorizing
        3. Clustiring with k-means
            3.1 Finding the optimal number of clusters
            3.2 Evaluting the optimal number of clusters
        4. Evaluation
            4.1 Time Consumed
            4.2 Similarities Surface
            4.3 Using Similarities To Classify
    """
    preprocessed_documents_path = Parameter(
        "preprocessed_documents_path",
        help="The path to preporcessed documents file.",
        default=join_relative_folder_path('data_preprocessed.parquet.gzip'))

    num_samples = Parameter(
        "num_samples",
        help="""How many sample are used in the run. Default is 100""",
        default=100)

    @step
    def start(self):
        """
        Load preprocessed documents

        """
        import pandas as pd
        import spacy
        from spacy.lang.pt.stop_words import STOP_WORDS
        from io import StringIO
        from datetime import datetime
        self.start_time = datetime.now()

        VECTOR_MODEL_NAME = "pt_core_news_sm"
        self.NLP_SPACY = spacy.load(VECTOR_MODEL_NAME)
        self.stopwords_set = set(STOP_WORDS)

        # Load the data set into a pandas dataframe.
        self.dataframe = pd.read_parquet(
            self.preprocessed_documents_path,
            columns=['text', 'doc_id', 'process_class'])
        self.next(self.sampling)

    @step
    def sampling(self):
        """
        Samples a choosen number of rows

        """
        ailab_df = self.dataframe.drop_duplicates(subset='doc_id')
        self.ailab_df = ailab_df[:self.num_samples]
        self.next(self.counting_and_vectorizing)

    @step
    def counting_and_vectorizing(self):
        """
            Convert the documents to bags of tokens,
            counts them and stores them as frequency vectors
        """
        from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
        from sklearn.pipeline import Pipeline

        tokenizer = self.NLP_SPACY.Defaults.create_tokenizer(self.NLP_SPACY)
        raw_text = self.ailab_df['text'].to_list()

        tokenized_text = []
        for row in raw_text:
            doc = tokenizer(row)
            preprocessed_doc = [
                token for token in doc
                if token.norm_ not in self.stopwords_set]

            tokenized_text.append(
                " ".join([word.text for word in preprocessed_doc]))

        count_vectorizer = CountVectorizer()
        tfidf_transformer = TfidfTransformer()

        ''' Encapsuling components in pipeline '''
        pipeline = Pipeline([
            ('count_vectorizer', count_vectorizer),
            ('tfidf_transformer', tfidf_transformer)
        ])

        self.vectorized_docs = pipeline.fit_transform(tokenized_text)

        self.next(self.finding_optimal_number_of_clusters)

    @step
    def finding_optimal_number_of_clusters(self):
        """
        To find it, let's use the elbow curve analysis.
        In this analysis we try models from an interval of cluster numbers
        and using each model's cluster cohesion to score them. 
        """
        from sklearn.cluster import KMeans
        number_clusters = range(1, 7)
        self.number_clusters = number_clusters

        kmeans = [
            KMeans(n_clusters=i, max_iter=600) for i in number_clusters]
        self.kmeans_score = [
            kmeans[i].fit(self.vectorized_docs).score(self.vectorized_docs)
            for i in range(len(kmeans))]
        self.next(self.evaluating_optimal_number_of_clusters)

    @step
    def evaluating_optimal_number_of_clusters(self):
        """
        On the last step we saw how after 3 cluster the improving in score
        slows down how much it changes.
        We use 3 as default
        """
        from sklearn.cluster import KMeans
        from sklearn.decomposition import TruncatedSVD

        sklearn_SVD = TruncatedSVD(n_components=2)
        kmeans = KMeans(n_clusters=3, max_iter=600, algorithm='auto')
        svd_docs = sklearn_SVD.fit_transform(self.vectorized_docs)
        fitted = kmeans.fit(svd_docs)
        self.prediction = kmeans.predict(svd_docs)
        self.ailab_df['group_prediction'] = self.prediction
        self.ailab_df['x_component'] = [x for x, y in svd_docs]
        self.ailab_df['y_component'] = [y for x, y in svd_docs]
        self.svd_docs = svd_docs

        self.next(self.creating_text_pairs)

    @step
    def creating_text_pairs(self):
        """
            Create a new dataframe with pair of documents. If the given
            input is N the output will be NxN long.
        """
        import pandas as pd

        ailab_df = self.ailab_df
        text_pairs = []
        comparing_same_text = True
        for question_1_index, question_1_row in ailab_df.iterrows():
            question_1_prediction = question_1_row['group_prediction']
            question_1_components_coordinates = [
                question_1_row['x_component'], question_1_row['y_component']]
            question_1_id = question_1_row['doc_id']
            question_1_class = question_1_row['process_class']
            for question_2_index, question_2_row in ailab_df.iterrows():
                if not comparing_same_text and question_1_index == question_2_index:
                    continue
                question_2_prediction = question_2_row['group_prediction']
                question_2_components_coordinates = [
                    question_2_row['x_component'], question_2_row['y_component']]
                question_2_id = question_2_row['doc_id']
                question_2_class = question_2_row['process_class']

                is_same_class = question_1_class == question_2_class
                is_same_group = question_1_prediction == question_2_prediction
                x_question_1, y_question_1 = question_1_components_coordinates
                x_question_2, y_question_2 = question_2_components_coordinates

                manhattan_distance = abs(x_question_1 - x_question_2) + abs(y_question_1 - y_question_2)
                similarity = 1 - manhattan_distance

                text_pairs.append([
                    question_1_id, question_2_id,
                    is_same_group, is_same_class, similarity])
        
        self.pair_text_df = pd.DataFrame(
            text_pairs,
            columns=[
                'question1_id', 'question2_id', 'is_same_group',
                'is_same_class', 'similarity'])
        self.next(self.caculating_similarities)

    @step
    def caculating_similarities(self):
        """
        Measures kmeans similarities for all pair texts and sort their values
        for better visualisation.
        """
        import pandas as pd
        pair_text_df = self.pair_text_df
        unique_ids = pair_text_df['question1_id'].unique()
        unique_ids_list = unique_ids.tolist()

        unique_ids_list.sort()
        dimension = len(unique_ids_list)
        distances_mapped = dict()
        predicitions_mapped = dict()
        for choosen_id in unique_ids_list:
            choosen_question_mask = pair_text_df['question1_id'].values == choosen_id
            compared_df = pair_text_df[choosen_question_mask]
            
            compared_df.sort_values(by=['question2_id'], inplace=True)
            
            compared_ids_list = compared_df['question2_id'].to_list()
            if compared_ids_list != unique_ids_list:
                print("An error ocurred")
                break
            predictions_list = []
            predictions_same_class_list = []
            for pair_text_index, pair_text_row in compared_df.iterrows():
                row_similarity = pair_text_row['similarity']
                predictions_list.append(row_similarity)
                predictions_same_class_list.append((
                    pair_text_row['is_same_group'],
                    pair_text_row['is_same_class']))
            distances_mapped[choosen_id] = predictions_list
            predicitions_mapped[choosen_id] = predictions_same_class_list

        first_question_id = unique_ids_list[0]
        mapped_distances_df = pd.DataFrame.from_dict(
            distances_mapped, orient='index', columns=unique_ids_list)
        mapped_distances_df.sort_values(by=[first_question_id], ascending=False, inplace=True)
        self.mapped_distances_df = mapped_distances_df[
            mapped_distances_df.index]

        self.mapped_predictions_df = pd.DataFrame.from_dict(
            predicitions_mapped, orient='index', columns=unique_ids_list)
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        Output data available on
            mapped_distances_df
        """
        from datetime import datetime

        self.end_time = datetime.now()
        self.total_time = str(self.end_time - self.start_time)
        pass


if __name__ == '__main__':
    ClusteringKMeans()
