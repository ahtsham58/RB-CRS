from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats
import pandas as pd
import numpy as np
import nlp
import re
import os
import random
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize

class RetrieveCandidateResponses_TFIDF:
    def __init__(self):
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR_PATH = os.path.abspath(os.path.dirname(self.PATH))
        self.DATA_path = os.path.join(self.ROOT_DIR_PATH, 'data\\dialog_data\\')
        self.corpus_dataSW = []
        self.corpus_data = []
        self.original_corpus = []
        self.input_dialogs = []
        self.generated_dialogs = []
        self.corpus_tfidfSW =None
        self.corpus_tfidf =None
        self.stored_sentences = None
        self.stored_embeddings = None
        self.vectorizerSW = None
        self.vectorizer = None
        self.error =  None
        with open(self.DATA_path+'ParsedData_PLSW.txt',encoding='utf-8') as f:
            self.corpus_dataSW = f.read().splitlines()
        with open(self.DATA_path+'ParsedData_PL.txt',encoding='utf-8') as f:
            self.corpus_data = f.read().splitlines()
        with open(self.DATA_path+'TrainingDataParsed_Con.txt',encoding='utf-8') as f:
            self.original_corpus = f.read().splitlines()
        if len(self.input_dialogs) < 1:
             with open(self.DATA_path+'dialogue_data.txt',encoding='utf-8') as f:
                 self.input_dialogs = f.read().splitlines()
             self.input_dialogs_for_history = self.read_input_dialogs(self.DATA_path + 'dialogue_data.txt')
        self.vectorizerSW = TfidfVectorizer()
        self.vectorizer = TfidfVectorizer()
        self.corpus_tfidfSW = self.vectorizerSW.fit_transform(self.corpus_dataSW)
        self.corpus_tfidf = self.vectorizer.fit_transform(self.corpus_data)

    def get_tf_idf_query_similarity(self, query):
        """
        vectorizer: TfIdfVectorizer model
        docs_tfidf: tfidf vectors for all docs
        query: query doc
        return: cosine similarity between query and all docs
        """
        query_tfidf = self.vectorizerSW.transform([query])
        cosineSimilarities = cosine_similarity(query_tfidf, self.corpus_tfidfSW).flatten()
        return cosineSimilarities

    def seeker_sentences_parser(self, line):
        if line:
            p = re.compile("SEEKER:(.*)").search(str(line))
            temp_line = p.group(1)
            m = re.compile('<s>(.*?)</s>').search(temp_line)
            seeker_line = m.group(1)
            seeker_line = seeker_line.lower().strip()
            return seeker_line

    def read_input_dialogs(self, path_to_input_file):
            is_visited = False
            #print(path_to_input_file)
            dialogues = []
            dialog = []
            with open(path_to_input_file, 'r', encoding='utf-8') as input:
                for line in input:
                    if not line.strip(): continue
                    if 'CONVERSATION:' in line and is_visited:
                        dialogues.append(dialog)
                        dialog = []
                        dialog.append(line)
                        is_visited = False
                    else:
                        dialog.append(line)
                        is_visited = True
            dialogues.append(dialog)
            return dialogues
    def preprocess_sentences(self, line):
        gt_line = self.replace_movieIds_withPL(line)
        #gt_line = gt_line.split('~')[1].strip().lower()
        gt_line = self.convert_contractions(gt_line)
        gt_line = re.sub('[^A-Za-z0-9]+', ' ', gt_line)
        gt_line = gt_line.replace('im', 'i am').strip()
        return gt_line

    def convert_contractions(self, line):
        #line = "What's the best way to ensure this?"
        filename = os.path.join(self.DATA_path+'//contractions.txt')
        contraction_dict = {}
        with open(filename) as f:
            for key_line in f:
               (key, val) = key_line.split(':')
               contraction_dict[key] = val
            for word in line.split():
                if word.lower() in contraction_dict:
                    line = line.replace(word, contraction_dict[word.lower()])
        return line

    def remove_stopwords(self, line):
        text_tokens = word_tokenize(line)
        tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
        filtered_sentence = (" ").join(tokens_without_sw)
        #print(filtered_sentence)
        return filtered_sentence

    def retrieve_response_candidates(self, k):
        #Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = 500
        for query in self.input_dialogs:
            if query.__contains__('CONVERSATION:'):
                self.generated_dialogs.append(query)
                continue
            counter = 0
            GT_candidate_list = []
            SK_candidate_index_list=[]
            #self.generated_dialogs.append(query)
            if query.__contains__('SEEKER:'):
                self.generated_dialogs.append(query)
                parsed_sentence = self.seeker_sentences_parser(query)
                preproc_sentence = self.preprocess_sentences(parsed_sentence)
                seeker_sentence = self.remove_stopwords(preproc_sentence)
                tokens = word_tokenize(seeker_sentence)
                if len(tokens) > 2:
                    query_tfidf = self.vectorizerSW.transform([seeker_sentence])
                    cosine_matrix = cosine_similarity(query_tfidf, self.corpus_tfidfSW).flatten()
                #similarity check for tokens less than 2
                else:
                    seeker_sentence = preproc_sentence
                    query_tfidf = self.vectorizer.transform([seeker_sentence])
                    cosine_matrix = cosine_similarity(query_tfidf, self.corpus_tfidf).flatten()

                sim_sent_indices = cosine_matrix.argsort()[:-top_k:-1]
                sim_sent_scores = cosine_matrix[sim_sent_indices]

                for score, idx in zip(sim_sent_scores, sim_sent_indices):
                    rt_sentence = ''
                    rt_corres_sentence= ''
                    rt_corres_token_count =0
                    rt_sentence = self.original_corpus[idx]
                    if idx < len(self.original_corpus)-1:
                        try:
                            rt_corres_sentence = self.original_corpus[idx+1]
                            rt_corres_sentence = re.sub('[^A-Za-z0-9~]+', ' ', rt_corres_sentence)
                            text_tokens = word_tokenize(rt_corres_sentence.split('~')[1])
                            rt_corres_token_count = len(text_tokens)
                        except  IndexError as err:
                            continue
                    else:
                        rt_corres_sentence = rt_sentence
                    if not rt_corres_sentence.__contains__('GT~') or rt_corres_token_count < 2 or rt_corres_token_count > 12 :
                        continue
                    elif not rt_sentence.__contains__('SKR~'):
                        continue
                    else:
                        counter =counter+1
                        self.generated_dialogs.append(str(self.original_corpus[idx] + ' |Score: %.4f' %(score)))
                        GT_candidate_list.append(str(self.original_corpus[idx+1]))
                        SK_candidate_index_list.append(idx)
                        if counter == k:
                            ranked_responses_list = self.rank_gt_candidates(parsed_sentence, GT_candidate_list)
                            self.generated_dialogs.extend(ranked_responses_list)
                            break
            else:
                self.generated_dialogs.append(query)
        return self.generated_dialogs

    def rank_candidates(self,seeker, candidate_list,index_list):

        ranked_list=[]
        common_token_list_count =[]
        seeker = self.preprocess_sentences(seeker)
        seeker_tokens = word_tokenize(seeker)
        for idx, candidate in enumerate(candidate_list):
            if candidate.__contains__('~'):
                candidate_tokens= (word_tokenize(self.preprocess_sentences(candidate_list[idx].split('~')[1].lower())))
                if 'movieid' in seeker_tokens and 'movieid' in candidate_tokens:
                    common_token_list_count.append(list(set(seeker_tokens).intersection(candidate_tokens)).__len__()+1)
                else:
                    common_token_list_count.append(list(set(seeker_tokens).intersection(candidate_tokens)).__len__())
            else:
                print(candidate)

        for i in range(0,5):

            #get first most similar response into the list
            min_length_candidate_index = common_token_list_count.index(max(common_token_list_count))
            ranked_list.append(self.original_corpus[index_list[min_length_candidate_index]+1])
            common_token_list_count.pop(min_length_candidate_index)
            index_list.pop(min_length_candidate_index)
            if len(common_token_list_count) < 1:
                break

        return ranked_list

    def rank_gt_candidates(self, seeker_query, candidate_list):
        score_list=[]
        seeker_query = self.preprocess_sentences(seeker_query)
        for idx, candidate in enumerate(candidate_list):
            if candidate.__contains__('~'):
                candidate_tokens= (word_tokenize(self.preprocess_sentences(candidate_list[idx].split('~')[1].lower())))
                if 'movieid' in candidate_tokens:
                    score_list.append(1)
                else:
                    score_list.append(0)
            else:
                print(candidate)
        count_score = sum(score_list)
        indices1s=[]
        indices0s=[]
        ranked_list=[]
        candidate_set1s =[]
        candidate_set0s = []
        for i in range(len(score_list)):
            if score_list[i]== 1:
                indices1s.append(i)
            else:
                indices0s.append(i)

        candidate_set1s = [candidate_list[x] for x in indices1s]
        candidate_set0s = [candidate_list[x] for x in indices0s]
        if count_score > 1:
                candidate_set1s.sort(key=lambda s: len(s))
                ranked_list.extend(candidate_set1s)
                ranked_list.extend(candidate_set0s)
        else:
            #candidate_set0s.sort(key=lambda s: len(s))
            ranked_list.extend(candidate_set0s)
            ranked_list.extend(candidate_set1s)
        return ranked_list

    def gt_sentence_parser(self, line):
        try:
            if not line == '\n':
                p = re.compile("GROUND TRUTH:(.*)").search(str(line))
                temp_line = p.group(1)
                m = re.compile('<s>(.*?)</s>').search(temp_line)
                gt_line = m.group(1)
                gt_line = gt_line.lower().strip()
                # gt_line = re.sub('[^A-Za-z0-9]+', ' ', gt_line)
            else:
                gt_line = ""
        except AttributeError as err:
                print('exception accured while parsing ground truth.. \n')
                #print(line)
                print(err)
                return gt_line

    def replace_movieIds_withPL(self , line):
        try:
            if "@" in line:
                ids = re.findall(r'@\S+', line)
                for id in ids:
                    line = line.replace(id,'movieid')
                    #id = re.sub('[^0-9@]+', 'movieid', id)
        except:
            lines.append(line)
            print('exception occured here')
        return line

    def retrieve_sentences_history(self, k):
        #Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
        top_k = 500
        for dlg in self.input_dialogs_for_history:
            for i in range(len(dlg)):
                if dlg[i].__contains__('CONVERSATION:'):
                    self.generated_dialogs.append(dlg[i].replace('\n',''))
                    continue
                dialog_history = dlg[1:i+1]
                dialog_history = [self.replace_movieIds_withPL(line) for line in dialog_history]
                counter = 0
                GT_candidate_list =[]
                #self.generated_dialogs.append(query)
                final_query = ''
                if dialog_history[len(dialog_history)-1].__contains__('SEEKER:'):
                    query = ''
                    for q in dialog_history:
                        if q.__contains__('SEEKER:'):
                            final_query = final_query + str(self.seeker_sentences_parser(q))+','
                        else:
                            final_query = final_query + str(self.gt_sentence_parser(q))+','

                    self.generated_dialogs.append(dialog_history[len(dialog_history)-1].replace('\n',''))
                    final_query = final_query.replace('None,','')
                    query_tfidf = self.vectorizerSW.transform([final_query])
                    cosine_matrix = cosine_similarity(query_tfidf, self.corpus_tfidfSW).flatten()
                top_k = top_k +1
                sim_sent_indices = cosine_matrix.argsort()[:-top_k:-1]
                sim_sent_scores = cosine_matrix[sim_sent_indices]
                for score, idx in zip(sim_sent_scores, sim_sent_indices):
                    rt_sentence = ''
                    rt_corres_sentence= ''
                    rt_corres_token_count =0
                    rt_sentence = self.original_corpus[idx]
                    if idx < len(self.original_corpus)-1:
                        try:
                            rt_corres_sentence = self.original_corpus[idx+1]
                            rt_corres_sentence = re.sub('[^A-Za-z0-9~]+', ' ', rt_corres_sentence)
                            text_tokens = word_tokenize(rt_corres_sentence.split('~')[1])
                            rt_corres_token_count = len(text_tokens)
                        except  IndexError as err:
                            continue
                    else:
                        rt_corres_sentence = rt_sentence
                    if not rt_corres_sentence.__contains__('GT~') or rt_corres_token_count < 2 or rt_corres_token_count > 12 :
                        continue
                    elif not rt_sentence.__contains__('SKR~'):
                        continue
                    else:
                        counter =counter+1
                        self.generated_dialogs.append(str(self.original_corpus[idx] + ' |Score: %.4f' %(score)))
                        SK_candidate_list.append(str(self.original_corpus[idx+1]))
                        SK_candidate_index_list.append(idx)
                        if counter == k:
                            #for i , cand_ind in enumerate(SK_candidate_index_list):
                                #self.generated_dialogs.append(str(self.original_corpus[cand_ind+1]))
                            ranked_responses_list = self.rank_gt_candidates(parsed_sentence, SK_candidate_list,SK_candidate_index_list)
                            self.generated_dialogs.extend(ranked_responses_list)
                            break
            else:
                self.generated_dialogs.append(query)
        return self.generated_dialogs

if __name__ == '__main__':

    obj = RetrieveCandidateResponses_TFIDF()
    dialogs = obj.retrieve_response_candidates(5)
    #dialogs = obj.retrieve_sentences_history(2)
    dialogs = list(filter(None,dialogs))
    with open(obj.DATA_path+'retrived_dialogs_TFIDF_with_reranking.txt', 'w', encoding='utf-8') as filehandle:
        for line in dialogs:
            filehandle.writelines("%s\n" % line)
    print('execution finshed')
    exit()

