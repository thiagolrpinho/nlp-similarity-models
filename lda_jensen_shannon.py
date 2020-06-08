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

def compute_coherence_values(dictionary, corpus, texts, limit, start=2):
    """
    Creates a list of models with different number of topics and returns theirs topics average
    coherence_values. The greater the better.

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    from gensim.models import LdaModel

    coherence_values = []
    model_list = []
    chunksize = 2000
    passes = 20
    iterations = 400
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    
    for num_topics in range(start, limit+1):
        # Make a index to word dictionary.
        temp = dictionary[0]  # This is only to "load" the dictionary.
        id2word = dictionary.id2token

        model = LdaModel(
            corpus=corpus, id2word=id2word, chunksize=chunksize,
            alpha='auto', eta='auto', iterations=iterations,
            num_topics=num_topics, passes=passes,
            eval_every=eval_every, random_state=25)

        top_topics = model.top_topics(corpus)
        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        coherence_values.append(sum([t[1] for t in top_topics]) / num_topics)
        model_list.append(model)
    
    return model_list, coherence_values


class LDAJensenShannonFlow(FlowSpec):
    """
    Use an LDA components analysis for text documents and then evaluate their
    similarities using Jensen-Shannon Distance

    1. Loading data
    2. Treating Data
    3. Generating LDA Component Analysis
        3.1 Finding Optimal Number of Components
        3.2 Generating Topics
    4. Similarity Jensen-Shannon
        4.1 Creating Topics Similarities Pairs
        4.2 Generating Heatmap for Text Similarities
    5. Bibliography
    """

    preprocessed_documents_data = Parameter(
        "preprocessed_documents_data",
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
        from io import StringIO
        from datetime import datetime
        self.start_time = datetime.now()
        # Load the data set into a pandas dataframe.
        ailab_df = pd.read_parquet(
            self.preprocessed_documents_data,
            columns=['text', 'doc_id', 'process_class'])
        num_samples = self.num_samples
        filename = "data_preprocessed"

        ailab_df = ailab_df.drop_duplicates(subset='doc_id')
        ailab_df = ailab_df.head(num_samples)
        number_of_classes = ailab_df['process_class'].nunique()

        self.ailab_df = ailab_df
        self.number_of_classes = number_of_classes
        self.next(self.selecting_number_components)

    @step
    def selecting_number_components(self):
        """
            Using the elbow curve analysis.
            In this analysis we try models from an interval of topics numbers
            and using each model's topics coherence values to score them.
        """
        from gensim.corpora import Dictionary

        ailab_df = self.ailab_df
        ailab_df['tokenized_docs'] = [
            document_text.split()
            for document_text in ailab_df['text'].to_list()]

        tokenized_docs = ailab_df['tokenized_docs'].to_list()
        # Creating the term dictionary of our courpus, where every unique term
        # is assigned an index.
        dictionary = Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
        # Converting list of documents (corpus) into Document Term Matrix
        # using dictionary prepared above.
        doc_term_matrix = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        # Can take a long time to run.
        maximum_number_components = 12
        minimum_number_components = 3
        model_list, coherence_values = compute_coherence_values(
            dictionary=dictionary,
            corpus=doc_term_matrix,
            texts=tokenized_docs,
            start=minimum_number_components,
            limit=maximum_number_components)

        best_model_index = coherence_values.index(max(coherence_values))
        self.number_topics = coherence_values.index(
            max(coherence_values)) + minimum_number_components
        self.lda_model = model_list[best_model_index]
        self.doc_term_matrix = doc_term_matrix
        self.dictionary = dictionary
        self.coherence_values = coherence_values
        self.minimum_number_components = minimum_number_components
        self.maximum_number_components = maximum_number_components
        self.next(self.generating_topics)

    @step
    def generating_topics(self):
        """
            Generate topics and describe documents based
            on how much they fit in each topic.
        """
        NUM_TOPICS = self.number_topics
        best_model = self.lda_model
        doc_term_matrix = self.doc_term_matrix

        probabilities = best_model[doc_term_matrix]
        same_shape_probabilities = []
        for topics_probability_list in probabilities:
            topics_probability_dict = dict()
            for index, score in topics_probability_list:
                topics_probability_dict[index] = score
            if len(topics_probability_dict) < NUM_TOPICS:
                for i in range(NUM_TOPICS):
                    if i not in topics_probability_dict:
                        topics_probability_dict[i] = 0.0
            same_shape_probabilities.append([
                topics_probability_dict[key]
                for key in sorted(topics_probability_dict.keys())])

        self.ailab_df['topics_similarity'] = same_shape_probabilities

        self.next(self.generating_text_pairs)

    @step
    def generating_text_pairs(self):
        """
        Create a new dataframe with pair of documents. If the given
        input is N the output will be NxN long.
        """

        import pandas as pd

        ailab_df = self.ailab_df
        comparing_same_text = True
        text_pairs = []
        for question_1_index, question_1_row in ailab_df.iterrows():
            question_1_topics = question_1_row['topics_similarity']
            question_1_id = question_1_row['doc_id']
            question_1_class = question_1_row['process_class']
            for question_2_index, question_2_row in ailab_df.iterrows():
                is_same_id = question_1_index == question_2_index
                if not comparing_same_text and is_same_id:
                    continue
                question_2_topics = question_2_row['topics_similarity']
                question_2_id = question_2_row['doc_id']
                question_2_class = question_2_row['process_class']
                is_same_class = question_1_class == question_2_class
                text_pairs.append([
                    question_1_id, question_1_topics, question_2_id,
                    question_2_topics, is_same_class])

        self.pair_text_df = pd.DataFrame(
            text_pairs, columns=[
                'question1_id', 'question1', 'question2_id',
                'question2', 'is_same_class'])
        self.next(self.mapping_distances)
    
    @step
    def mapping_distances(self):
        """
            Measures jensen shannon similarities for all pair texts topics
            and sort their values.
        """

        from scipy.spatial.distance import jensenshannon
        import pandas as pd

        pair_text_df = self.pair_text_df
        number_of_classes = self.number_of_classes

        unique_ids_list = pair_text_df['question1_id'].unique().tolist()
        unique_ids_list.sort()

        dimension = len(unique_ids_list)
        distances_mapped = dict()
        predicitions_mapped = dict()
        for choosen_id in unique_ids_list:
            choosen_question_mask = pair_text_df[
                'question1_id'].values == choosen_id
            compared_df = pair_text_df[choosen_question_mask]
            
            compared_df.sort_values(by=['question2_id'], inplace=True)
            
            compared_ids_list = compared_df['question2_id'].to_list()
            if compared_ids_list != unique_ids_list:
                print("An error ocurred")
                break
            
            threshold = 1/number_of_classes
            predictions_list = []
            predictions_same_class_list = []
            for pair_text_index, pair_text_row in compared_df.iterrows():
                row_distance = 1 - jensenshannon(
                    pair_text_row['question1'], pair_text_row['question2'])
                same_class_prediction = row_distance > threshold
                predictions_list.append(row_distance)
                predictions_same_class_list.append(
                    (same_class_prediction, pair_text_row['is_same_class']))
            distances_mapped[choosen_id] = predictions_list
            predicitions_mapped[choosen_id] = predictions_same_class_list

        mapped_distances_df = pd.DataFrame.from_dict(
            distances_mapped, orient='index', columns=unique_ids_list)
        mapped_predictions_df = pd.DataFrame.from_dict(
            predicitions_mapped, orient='index', columns=unique_ids_list)

        # Ordering mapped distances
        first_question_id = unique_ids_list[0]
        mapped_distances_df.sort_values(
            by=[first_question_id], ascending=False, inplace=True)

        self.mapped_distances_df = mapped_distances_df[
            mapped_distances_df.index]
        self.mapped_predictions_df = mapped_predictions_df
        self.next(self.end)

    @step
    def end(self):
        """
        End the flow.
        Output data available on
            mapped_distances_df
            mapped_predictions_df
        """
        from datetime import datetime
        
        self.end_time = datetime.now()
        self.total_time = self.end_time - self.start_time
        pass


if __name__ == '__main__':
    LDAJensenShannonFlow()
