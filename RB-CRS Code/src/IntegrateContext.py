import pandas as pd
import re
import random
import numpy as np
import os
from src.Recommender_MF import Recommender_MF
from src.Recommender import Recommender
from src.Recommender_with_genre import Recommender_with_genre



class IntegrateContext:
    def __init__(self):
        self.PATH = os.path.dirname(os.path.abspath(__file__))
        self.ROOT_DIR_PATH = os.path.abspath(os.path.dirname(self.PATH))
        self.ITEM_DATA_path = os.path.join(self.ROOT_DIR_PATH, 'data\\recommenders_item_data\\')
        self.rec = None
        self.genre_rec= None
        self.rec_only_genre = None
        self.mentioned_movies= []
        self.gt_pref_keywords = []
        self.recommended_movies= []
        self.seeker_movies = []
        self.sk_pref_tokens = []
        self.movie_mentions_data =pd.DataFrame()
        self.movie_mentions_data = pd.read_csv(self.ITEM_DATA_path +'movies_with_mentions.csv', ',', encoding='utf-8', dtype=object, low_memory=False)
        self.movies_metadata = pd.read_csv(self.ITEM_DATA_path +'combined_metadata.csv')
        self.pref_keywords = ['scary','horror','pixar','graphic', 'classic','comedy', 'kids','funny', 'disney','comedies','action','family','adventure','crime','fantasy','thriller','scifi', 'documentary', 'science fiction', 'drama','romance','romances','romantic','mystery','mysteries','history','no preference','suspense' ]
        self.country_keywords = ['american','asian','european','british']
        self.asser_tags= ['good', 'great', 'nice', 'awesome', 'fine']



    #read and retrive dialogs from the input file
    def read_dialogs_input_file(self, file_name):
        is_visited = False
        redial_dialogues = []
        dialog = []
        with open(file_name, 'r') as input:
            for line in input:
                if not line.strip(): continue
                if 'CONVERSATION:' in line and is_visited:
                    redial_dialogues.append(dialog)
                    dialog = []
                    dialog.append(line)
                    is_visited = False
                else:
                    dialog.append(line)
                    is_visited = True
        if not dialog[0].__contains__('CONVERSATION:'):
            return
        redial_dialogues.append(dialog)
        return redial_dialogues
    ## parse seeeker utterance in a text line
    def seeker_sentences_parser(self, line):
        if line:
            p = re.compile("SEEKER:(.*)").search(str(line))
            temp_line = p.group(1)
            m = re.compile('<s>(.*?)</s>').search(temp_line)
            seeker_line = m.group(1)
            seeker_line = seeker_line.lower().strip()
            return seeker_line
    ## parse ground truth utterance in a text line
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

            return gt_line
        except AttributeError as err:
            print('exception accured while parsing ground truth.. \n')
            print(line)
            print(err)
    ##parse all the seeker movie IDs from the utterance into a list
    def check_if_movieid_mention(self, query):
        # check if there is a movie mention
        parsed_IDs = []
        if query.__contains__('@'):
            IDs = re.findall('@\\S+', query)
            parsed_IDs = [id.replace('@', '') for id in IDs]
            self.mentioned_movies.extend(parsed_IDs)
        else:
            parsed_IDs = []
        return parsed_IDs
    ##parse all the ground truth movie IDs from the utterance into a list
    def check_GT_movie_mention(self, query):
        # check if there is a movie mention
        parsed_IDs = []
        if query.__contains__('@'):
            IDs = re.findall('@(.*)', query)
            parsed_IDs = [id.split('(')[0].strip() for id in IDs]
        else:
            parsed_IDs = []
        return parsed_IDs
    ## input: a complete dialog with retrieved candidate responses from the previous retriveal process
    ## this function process the input dialogs and integerate new movie recommendations
    ## based on user dialog history context (e.g., movie IDs, preference keywords like genres)
    ##output: a complete dialog with new movie recommendations
    def integrate_recommendations(self, dialog):
        self.mentioned_movies = []
        completed_dialog_list = []
        self.seeker_movies = []
        groundtruth_movies = []
        pref_common_tokens =[]
        self.recommended_movies= []
        self.sk_pref_tokens= []
        self.gt_pref_keywords=[]
        for i, line in enumerate(dialog):
            if line.__contains__('CONVERSATION:'):
                completed_dialog_list.append(line)
                continue
            elif line.__contains__('SKR~'):
                continue
            elif line.__contains__('SEEKER:'):
                completed_dialog_list.append(line)
                is_GT_parsed = False
                seeker_query = self.seeker_sentences_parser(line)
                seeker_query = re.sub('[^A-Za-z0-9@]+', ' ', seeker_query)
                seekerk_query_list = seeker_query.split(' ')
                pref_common_tokens = list(set(seekerk_query_list).intersection(self.pref_keywords))
                if len(pref_common_tokens) > 0:
                    self.sk_pref_tokens.append(pref_common_tokens)
                movieids = self.check_if_movieid_mention(seeker_query)
                if len(movieids) > 0:
                    self.seeker_movies.extend(movieids)
            ## is_GT_parsed means flag for first candidate resoponse to be process while ignore the rest
            ## in case of multiple retrived response candidates, e.g., in this N=5
            elif line.__contains__('GT~') and not is_GT_parsed:
                try:
                    is_GT_parsed = True
                    orginal_response= line
                    recomendation_list = []
                    movieids = self.check_if_movieid_mention(line)
                    total_recommendations = len(movieids)
                    #check  if GT contains recommendation,
                    if total_recommendations > 0:
                        #get recommendations based on the user context i.e.,
                        # 1) last seeker movie id, 2) last ground_truth movie id, 3) preference keywords like genres.
                        if len(self.seeker_movies) > 0:
                            #pass last seeker movie to get recommendations and count of movies mentioned in a response
                            recomendation_list = self.get_recommendation(self.seeker_movies[len(self.seeker_movies)-1],total_recommendations, self.recommended_movies)
                            parsed_line = self.replace_recommendations(orginal_response, movieids, recomendation_list)

                        #retrieve new recommendations based on genre in the immidiate current seeker utterance.
                        elif len(pref_common_tokens) > 0 :
                            mentioned_genre= pref_common_tokens[len(pref_common_tokens)-1]
                            recomendation_list = self.get_recommendation_by_genre(mentioned_genre, total_recommendations, self.recommended_movies)
                            parsed_line = self.replace_recommendations(orginal_response, movieids, recomendation_list)
                        # if none of the above case is true, retrieve new recommendations based on ground_truth movies
                        elif len(groundtruth_movies) > 0:
                            #pass last ground truth movie to get recommendations and length of movies in a response
                            recomendation_list = self.get_recommendation(groundtruth_movies[len(groundtruth_movies)-1][0], total_recommendations, self.recommended_movies)
                            parsed_line = self.replace_recommendations(orginal_response, movieids, recomendation_list)
                        #retrieve new recommendations based on genre from the previous seeker utterances.
                        elif len(self.sk_pref_tokens) > 0:
                            mentioned_genre= self.sk_pref_tokens[len(self.sk_pref_tokens)-1][0]
                            recomendation_list = self.get_recommendation_by_genre(mentioned_genre, total_recommendations, self.recommended_movies)
                            parsed_line = self.replace_recommendations(orginal_response, movieids, recomendation_list)
                        #retrieve new recommendations based on genre from the previous ground_truth utterances.
                        elif len(self.gt_pref_keywords) > 0:
                            mentioned_genre= self.gt_pref_keywords[len(self.gt_pref_keywords)-1][0]
                            recomendation_list = self.get_recommendation_by_genre(mentioned_genre, total_recommendations, self.recommended_movies)
                            parsed_line = self.replace_recommendations(orginal_response, movieids, recomendation_list)
                        completed_dialog_list.append(parsed_line)
                    else:
                        completed_dialog_list.append(orginal_response)
                except:
                    completed_dialog_list.append(orginal_response)
                    continue
            ##process ground_truth response here and extract mentioned movie ids
            elif line.__contains__('GROUND TRUTH:'):
                completed_dialog_list.append(line)
                gt_response = self.gt_sentence_parser(line)
                gt_response = re.sub('[^A-Za-z0-9@]+', ' ', gt_response)
                gt_query_list = gt_response.split(' ')
                keywds = list(set(gt_query_list).intersection(self.pref_keywords))
                if len(keywds) > 0:
                    self.gt_pref_keywords.append(keywds)
                movie_mentions = self.check_if_movieid_mention(line)
                if movie_mentions:
                    groundtruth_movies.append(movie_mentions)

        return completed_dialog_list

    def replace_recommendations(self, line, movieids, rec_list):
        if len(rec_list) > 0:
            for i, id in enumerate(movieids):
                recomendation = str(rec_list[i])
                self.recommended_movies.append(recomendation)
                line = line.replace(id,recomendation+'"')
        return line

    def process_dialog(self,input_dialogs):
        final_dialogues = []
        for dlg in input_dialogs:
            print('new dialog is processing.......')
            recs_dialogue = self.integrate_recommendations(dlg)
            final_dialog = self.integrate_metadata(recs_dialogue)
            final_dialogues.append(final_dialog)
        ##after integeration of movie recommendations and related metadata, replace redial movie Ids with actual movie titles
        final_dialogues = self.replace_movieIDs_with_titles(final_dialogues)
        return final_dialogues

    #get recommendations based on movie id
    def get_recommendation(self, movie_id, nb_recommendations, prev_movie_mentions):
        prev_movie_mentions = self.get_seeker_movie_titles(prev_movie_mentions)
        try:
            if self.rec is None:
                self.rec = Recommender_MF()
            suggestion_list = self.rec.get_similar_movies_based_on_user_ratings(movie_id, nb_recommendations, prev_movie_mentions)

            # ##backup recommender in case movie is not present in movielens data, identifier means, recommendation request is actual
            if len(suggestion_list) == 0:
                if self.genre_rec is None:
                    self.genre_rec = Recommender()
                suggestion_list = self.genre_rec.get_similar_movies_based_on_content(movie_id, nb_recommendations,prev_movie_mentions)

        except RuntimeError as err:
            print(err)
            suggestion_list = []
        return suggestion_list


        #get recommendation by recommender system which take genre type as an input

    def get_seeker_movie_titles(self, prev_movie_mentions):
        for id in self.seeker_movies:
            temp = self.movie_mentions_data[self.movie_mentions_data['databaseId'] == str(id)]
            if len(temp) > 0:
                movieName = temp['title'].iloc[0]
                if movieName not in prev_movie_mentions:
                    prev_movie_mentions.append(movieName)
        return prev_movie_mentions
    #get recommendations based on movie genre
    def get_recommendation_by_genre(self, genre, nb_recom, recommendations):
        recommendations = self.get_seeker_movie_titles(recommendations)
        try:
            suggestion_list= []
            if self.rec_only_genre is None:
                self.rec_only_genre = Recommender_with_genre()
            suggestion_list = self.rec_only_genre.get_similar_movies_based_on_genre(genre, nb_recom,recommendations)
        except RuntimeError as err:
            print(err)
            suggestion_list =[]
        return suggestion_list

    def contains_word(self, input_str, word):
        return f' {word} ' in f' {input_str} '

    # integration of domain metadata after new recommendations are incorporated.
    # input: a complete dialog integrated with new recommendations for retrieved candidate response against each seeker utterance
    # output: a complete dialog with metadata integrations based on recommended movie title
    def integrate_metadata(self, dialog):
        completed_dialog_list = []
        seeker_pref_keywords =[]
        for i, line in enumerate(dialog):
            line = line.replace('\n','')
            if line.__contains__('CONVERSATION:'):
                completed_dialog_list.append(line)
                continue
            elif line.__contains__('SEEKER:'):
                completed_dialog_list.append(line)
                seekerk_query_list = line.lower().split(' ')
                pref_keywords = list(set(seekerk_query_list).intersection(self.pref_keywords))
                if len(pref_keywords) > 0:
                    seeker_pref_keywords.extend(pref_keywords)

            elif line.__contains__('GT~'):
                try:
                    original_response = line
                    movie_titles = self.check_GT_movie_mention(line)
                    line = re.sub('[^A-Za-z0-9@()]+', ' ', line)
                    total_recommendations = len(movie_titles)
                    line = line.lower()
                    GT_response_list = line.split(' ')
                    region_list = line.split(' ')
                    GT_pref_keywords = list(set(GT_response_list).intersection(self.pref_keywords))
                    common_region_list = list(set(region_list).intersection(self.country_keywords))
                    #check  if retrieved candidate response contains recommendation,
                    if len(GT_pref_keywords) > 0 and total_recommendations > 0:
                        first_movie_rec = movie_titles[0]
                        #fetch and replace genres context
                        original_response = self.retrieve_genres(original_response, GT_pref_keywords, first_movie_rec)
                    ## in case of no recommendation, replace genres with previous seeker preference keywords
                    elif len(GT_pref_keywords) > 0 and len(seeker_pref_keywords) > 0:
                        original_response = original_response.replace(GT_pref_keywords[len(GT_pref_keywords)-1],seeker_pref_keywords[len(seeker_pref_keywords)-1])

                    #integrate actors/director details
                    if total_recommendations > 0 and (self.contains_word(line, 'with') or self.contains_word(line, 'has') or self.contains_word(line, 'stars') or self.contains_word(line, 'starring') or self.contains_word(line, 'staring') or self.contains_word(line, 'by')):
                         original_response = self.retrieve_actors(original_response, line, movie_titles[0])
                    elif len(self.recommended_movies)> 0 and (self.contains_word(line, 'with') or self.contains_word(line, 'has') or self.contains_word(line, 'stars') or self.contains_word(line, 'starring') or self.contains_word(line, 'staring') or self.contains_word(line, 'by')):
                         original_response = self.retrieve_actors(original_response, line, self.recommended_movies[len(self.recommended_movies) - 1])
                    ## replace movie region keywords
                    if len(common_region_list) > 0 and total_recommendations > 0:
                        original_response = self.replace_country(original_response, movie_titles[0],common_region_list)

                    original_response = original_response.replace('comedy', 'funny').replace('laugh','movie').replace('romance','romantic').replace('@','"')
                    completed_dialog_list.append(original_response)

                except:
                    completed_dialog_list.append(original_response.replace('@','"'))
                    continue
            elif line.__contains__('GROUND TRUTH:'):
                completed_dialog_list.append(line)

        return completed_dialog_list

    def replace_country(self,original_response, movie_mention, common_keyword):
        if movie_mention.__contains__('('):
            movie_mention = movie_mention.split('(')[0].strip()
        df = self.movies_metadata.loc[self.movies_metadata['title'] == movie_mention]
        if len(df) <1:
            if movie_mention.__contains__(','):
                movie_mention =movie_mention.split(',')[0].strip()
                df = self.movies_metadata[self.movies_metadata['title'].str.contains(movie_mention)]
            elif movie_mention.lower().__contains__('the'):
                movie_mention= movie_mention.split('the')[0].strip()
                df = self.movies_metadata[self.movies_metadata['title'].str.contains(movie_mention)]

        region = df.head(1)['country'].tolist()
        if len(region) > 0:
            for i , key in enumerate(common_keyword):
                if region[0] == 'USA':
                    actual_keyword ='American'
                elif region[0] == 'UK':
                    actual_keyword ='British'
                elif region[0] == 'Russia':
                    actual_keyword = 'Russian'
                original_response = original_response.replace(key.title(),actual_keyword)
                original_response = original_response.replace(key,actual_keyword)
        return original_response


    def retrieve_actors(self, original_line, line, movie_mention):
        if movie_mention.__contains__('('):
            movie_mention = movie_mention.split('(')[0].strip()
        is_movie_keyword_identifier = False
        if 'with' in movie_mention or 'has' in movie_mention or 'by' in movie_mention or 'stars' in movie_mention:
            is_movie_keyword_identifier = True
        if line.__contains__('with') and not is_movie_keyword_identifier:
            names = line.split('with')[1].strip().split(' ')
            if len(names) > 0 and not 'you' in names and not 'your' in names and not 'that' in names:
                original_line = self.get_actors_metadata(original_line, names, movie_mention)
        elif line.__contains__('has') and not is_movie_keyword_identifier:
            names = line.split('has')[1].strip().split(' ')
            if len(names) >0:
                original_line = self.get_actors_metadata(original_line, names, movie_mention)
        elif line.__contains__('starring') and not is_movie_keyword_identifier:
            names = line.split('starring')[1].strip().split(' ')
            if len(names) >0:
                original_line = self.get_actors_metadata(original_line, names, movie_mention)
        elif line.__contains__('staring') and not is_movie_keyword_identifier:
            names = line.split('staring')[1].strip().split(' ')
            if len(names) >0:
                original_line = self.get_actors_metadata(original_line, names, movie_mention)

        elif line.__contains__('by') and not is_movie_keyword_identifier :
            names = line.split('by')[1].strip().split(' ')
            if len(names) >0:
                original_line = self.get_actors_metadata(original_line, names, movie_mention)
        elif line.__contains__('stars') and not is_movie_keyword_identifier :
            names = line.split('stars')[1].strip().split(' ')
            if len(names) >0:
                original_line = self.get_actors_metadata(original_line, names, movie_mention)

        return original_line

    def get_actors_metadata(self, line, names , movie_mention):
        df = self.movies_metadata.loc[self.movies_metadata['title'] == movie_mention]
        if len(df) <1:
            if movie_mention.__contains__(','):
                movie_mention =movie_mention.split(',')[0].strip()
                df = self.movies_metadata[self.movies_metadata['title'].str.contains(movie_mention)]
            elif movie_mention.lower().__contains__('the'):
                movie_mention= movie_mention.split('the')[0].strip()
                df = self.movies_metadata[self.movies_metadata['title'].str.contains(movie_mention)]
        actors = df.head(1)['actors'].tolist()
        if len(actors) > 0:
            actors_list = actors[0].split(',')
            for i, name in enumerate(names):
                actor = actors_list[i]
                if i == len(names)-1:
                    line = line.replace(name, actor.strip())
                    line = line.replace(name.title(), actor.strip())
                else:
                    line = line.replace(name, actor.strip()+',')
                    line = line.replace(name.title(), actor.strip()+',')
                #max two actors replacement to avoid indexing issues
                if i == 1:
                    break
        return line

    def retrieve_genres(self, line, prefer_keywords, movie_mention):
        if line.lower().__contains__('not a') and len(prefer_keywords) > 0:
            return line
        if movie_mention.__contains__('('):
            movie_mention = movie_mention.split('(')[0].strip()
        df = self.movies_metadata.loc[self.movies_metadata['title'] == movie_mention]
        if len(df) < 1:
                if movie_mention.__contains__(','):
                    movie_mention =movie_mention.split(',')[0].strip()
                    df = self.movies_metadata[self.movies_metadata['title'].str.contains(movie_mention)]
                elif movie_mention.lower().__contains__('the'):
                    movie_mention= movie_mention.split('the')[0].strip()
                    df = self.movies_metadata[self.movies_metadata['title'].str.contains(movie_mention)]
        genres = df.head(1)['genres'].tolist()
        if len(genres) >0:
            genres_list = genres[0].lower().split('|')
            for i, pref_key in enumerate(prefer_keywords):
                if i > len(genres_list):
                    line= line.replace(pref_key,'')
                else:
                    line = line.replace(pref_key, genres_list[i])
                    line = line.replace(pref_key.title(), genres_list[i])
                    line = line.replace('comedy', 'funny').replace('romance','romantic')
                    #remove consective repeated genres
                    temp_line = re.sub('[^A-Za-z0-9@().]+', ' ', line)
                    unique_words = dict.fromkeys(temp_line.split())
                    line = ' '.join(unique_words)
        else:
            line = line.replace(prefer_keywords[0], self.asser_tags[random.randint(0,len(self.asser_tags))])
        return line
    def replace_movieIDs_with_titles(self, input_dialogue_data):
        final_dialogues = []
        for dlg in input_dialogue_data:
            lines = []
            for row in dlg:
                try:
                    # replace movie IDs  of the format, e.g. @1234, with movie title
                    if "@" in row:
                        ids = re.findall(r'@\S+', row)
                        for id in ids:
                            id = re.sub('[^0-9@]+', '', id)
                            onlyid = id.split('@')[1]
                            temp = self.movie_mentions_data[self.movie_mentions_data['databaseId'] == str(onlyid)]
                            if len(temp) > 0:
                                movieName = temp['title'].iloc[0]
                                # m = movieName.index('(')
                                # movieName = movieName[:m]
                                row = row.replace(id, '"' +movieName+'"')

                        lines.append(row)
                    # replace movie IDs  of the format, e.g. "1234, with movie title
                    elif '"' in row:
                        ids = re.findall(r'"\S+', row)
                        for id in ids:
                            id = re.sub('[^0-9"]+', '', id)
                            onlyid = id.split('"')[1]
                            temp = self.movie_mentions_data[self.movie_mentions_data['databaseId'] == str(onlyid)]
                            if len(temp) > 0:
                                movieName = temp['title'].iloc[0]
                                # m = movieName.index('(')
                                # movieName = movieName[:m]
                                row = row.replace(id, '"' +movieName+'"')

                        lines.append(row)
                    else:
                        lines.append(row)
                except:
                    lines.append(row)
                    continue
            final_dialogues.append(lines)
        print('execution ends here')
        return final_dialogues

if __name__ == '__main__':
     obj = IntegrateContext()
     data_path = obj.ROOT_DIR_PATH+'\\data\\dialog_data\\'
     input_dialogs = obj.read_dialogs_input_file(data_path + 'retrived_dialogs_TFIDF_with_reranking.txt')
     if len(input_dialogs) > 0:
         retrived_dialogs_with_rec = obj.process_dialog(input_dialogs)

         #write genenerated dialogs in the text file
         with open(data_path + 'generated_dialogs_TFIDF.txt', 'w') as filehandle:
            for dia in retrived_dialogs_with_rec:
                filehandle.writelines("%s\n" % line for line in dia)
         print('Dialogs have been generated successfully.')

     else:
         print('error occurs while reading the file')
     exit()





