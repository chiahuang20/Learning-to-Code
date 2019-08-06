from program import LogisticRegressionModel, MultinomialNaiveBayesModel, GaussianNaiveBayesModel, KNearestNeighborModel, RandomForestModel
from program import Models

print(Models.listAllModels())

a = LogisticRegressionModel(target="business", converter="tfidf")
print(a)
print(a.predict("sample_dataset.xlsx"))

b = MultinomialNaiveBayesModel(target="occupation", converter="bagofwords")
print(b)
print(b.predict("sample_dataset.xlsx"))

c = GaussianNaiveBayesModel(target="occupation", converter="bagofwords")
print(c)
print(c.predict("sample_dataset.xlsx"))

d = KNearestNeighborModel(target="occupation", converter="bagofwords")
print(d)
print(d.predict("sample_dataset.xlsx"))

e = RandomForestModel(target="occupation", converter="bagofwords")
print(e)
print(u.predict("sample_dataset.xlsx"))