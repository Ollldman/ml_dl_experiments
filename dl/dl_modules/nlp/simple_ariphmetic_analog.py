from gensim.models import KeyedVectors
# загрузите модель fastText из файла "/models/cc.ru.300.vec.gz"
model_ft = KeyedVectors.load_word2vec_format(
    "/models/cc.ru.300.vec.gz", 
    binary=False,
    limit=50000
)

res = model_ft.most_similar(
    positive=["женщина", "король"],  # векторы складываем
    negative=["мужчина"]                # векторы вычитаем
)
print("король – мужчина + женщина →", res[0][0])  # первое слово из результата