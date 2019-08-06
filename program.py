import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score
import pickle as pkl
import gzip
from enum import Enum

class InputType(Enum):
    BUSINESS = "business"
    OCCUPATION = "occupation"
    BAGOFWORDS = "bagofwords"
    TFIDF = "tfidf"

class ResponseType(Enum):
    BUSINESS = "bus"
    OCCUPATION = "occu"
    BAGOFWORDS = "bow"
    TFIDF = "tfidf"

class ModelType(Enum):
    LGR = "lgr"
    MNB = "mnb"
    GNB = "gnb"
    KNN = "knn"
    RF = "rf"

class Models():    # all models'  parent class
    def __init__(self, target, converter, modeltype):
        self.target = target
        self.converter = converter
        self.modeltype = modeltype
        if (self.converter in [InputType.BAGOFWORDS.value, InputType.TFIDF.value]) and \
            (self.target in [InputType.BUSINESS.value, InputType.OCCUPATION.value]):
            self.setModel()

    def setModel(self):
        if (self.target in [InputType.BUSINESS.value, InputType.OCCUPATION.value]) and \
            (self.converter in [InputType.BAGOFWORDS.value, InputType.TFIDF.value]):
            with gzip.open("models/%s.%s.%s.model.pkl.gzip" % \
                (self.modeltype, \
                self.inputToOutputConverter(self.converter), \
                self.inputToOutputConverter(self.target)), "rb") as md:
                self.__model = pkl.load(md)

    @classmethod
    def listAllModels(cls):
        return [i.__name__ for i in Models.__subclasses__()]
    
    def inputToOutputConverter(self, input):
        if input == InputType.BUSINESS.value: return ResponseType.BUSINESS.value
        elif input == InputType.OCCUPATION.value: return ResponseType.OCCUPATION.value
        elif input == InputType.BAGOFWORDS.value: return ResponseType.BAGOFWORDS.value
        elif input == InputType.TFIDF.value: return ResponseType.TFIDF.value
        else: return None

    def getVocab(self):
        if self.target in [InputType.BUSINESS.value, InputType.OCCUPATION.value]:
            
            response = self.inputToOutputConverter(self.target)
            
            with open("new_%s.txt" % response, encoding="utf-8") as f:
                vocab = f.read()
            return eval(vocab)
        else:
            print("You enter the wrong target.")

    def convertToVector(self, X):
        # load in the dictionary
        vocab = self.getVocab()  # 取得之前訓練時的字典!!!!

        if self.converter == InputType.BAGOFWORDS.value:
            X_bow = CountVectorizer(vocabulary=vocab).fit_transform(X).toarray()
            return X_bow
        elif self.converter == InputType.TFIDF.value:
            X_tfidf = TfidfVectorizer(vocabulary=vocab).fit_transform(X).toarray()
            return X_tfidf
        else:
            print("You enter a wrong converter")
            return None


    def splitWords(self, excel_file):  # converter 為bow or tfidf !!!
        jieba.set_dictionary("trad_dict.txt")
        excel_df = pd.read_excel(excel_file, encoding="utf-8")  # excel_file為user 輸入的檔案!

        # load in stopwords
        stopword = []
        with open('stopwords_chn.txt', 'r', encoding='UTF-8') as file:
            for data in file.readlines():
                data = data.strip()
                stopword.append(data)
        stopword += [".", "丶", "(", ")", "-"]
        
        if self.target == InputType.BUSINESS.value:  # target為要預測的目標
            # to predict business code
            try:
                zip_ind_x = zip(excel_df["k_a08a_1"].values, excel_df["k_a08a_2"].values)

                ind_X = np.array([" ".join(jieba.cut(a.split(",")[0], cut_all=False)) + " " + \
                                  " ".join(jieba.cut(b.split(",")[0], cut_all=False)) for a, b in zip_ind_x]).tolist()

                # split original sentence to words and delete stopwords
                ind_X = [content.split(" ") for content in ind_X]
                ind_X = [[i for i in each if i not in stopword] for each in ind_X]
                ind_X = [" ".join(each) for each in ind_X]
                return self.convertToVector(ind_X)

            except Exception as e:
                print(e)
                print("You load in the wrong format of excel file.")
            
        elif self.target == InputType.OCCUPATION.value:
            try:
                
                zip_occu_x = zip(excel_df["k_a08a_3"].values, excel_df["k_a08a_4"].values, excel_df["k_a08a_5"].values)
                occu_X = np.array([" ".join(jieba.cut(a.split(",")[0], cut_all=False)) + " " + \
                    " ".join(jieba.cut(b.split(",")[0], cut_all=False)) + " " + \
                    " ".join(jieba.cut(c.split(",")[0], cut_all=False)) for a, b, c in zip_occu_x]).tolist()
                
                occu_X = [content.split(" ") for content in occu_X]
                occu_X = [[i for i in each if i not in stopword] for each in occu_X]
                occu_X = [" ".join(each) for each in occu_X]
                return self.convertToVector(occu_X)

            except Exception as e:
                print("You load in the wrong format of excel file.")
        else:
            print("You enter the wrong predict target.")


    def __str__(self):
        return """
        This is a '%s' , 
        converting words using '%s' method, 
        predict for '%s code'.
        """ % (self.__class__.__name__, self.converter, self.target)

    def predict(self, excel_file):
        # load the dictionary, convert csv to word vector
        # then do sklearn.<model>.predict() method to output results
        if (self.converter in [InputType.BAGOFWORDS.value, InputType.TFIDF.value]) and \
            (self.target in [InputType.BUSINESS.value, InputType.OCCUPATION.value]):
            wd_vec = self.splitWords(excel_file)  # only python3 will do successfully
            self._result = self.__model.predict(wd_vec)
            return self.generateOutput(excel_file, self._result)
        else:
            return "You enter the wrong target or wrong converter."

    def generateOutput(self, excel_file, result):
        output = pd.read_excel(excel_file, encoding="utf-8")
        output['Prediction'] = result
        if self.target == InputType.BUSINESS.value:
            self._accuracy = accuracy_score(output["a08a01"], output["Prediction"])
        elif self.target == InputType.OCCUPATION.value:
            self._accuracy = accuracy_score(output["a08a02"], output["Prediction"])
        print(output[:10])
        print("\n............................\n")
        return "Accuracy = %f" % self._accuracy

class LogisticRegressionModel(Models):
    # target: to predict business code or occupation code
    def __init__(self, target, converter):
        super().__init__(target, converter, ModelType.LGR.value) 


class MultinomialNaiveBayesModel(Models):
    def __init__(self, target, converter):
        super().__init__(target, converter, ModelType.MNB.value)

class GaussianNaiveBayesModel(Models):
    def __init__(self, target, converter):
        super().__init__(target, converter, ModelType.GNB.value)


class KNearestNeighborModel(Models):
    def __init__(self, target, converter):
        super().__init__(target, converter, ModelType.KNN.value)


class RandomForestModel(Models):
    def __init__(self, target, converter):
        super().__init__(target, converter, ModelType.RF.value)
