import newspaper

url = "https://www.programiz.com/python-programming"
article = newspaper.article(url)

print(article.text)
