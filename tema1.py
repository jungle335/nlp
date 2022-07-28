import bs4
import json
import requests
import threading
import time

movies = []
semaphore = threading.Semaphore(10)


def htmlParser(link):
    return bs4.BeautifulSoup(requests.get(link).content, "html.parser")


def buildInfoReview(movie, container):
    reviewTitle = container.select_one("a.title").text
    reviewText = container.select_one("div.text").text
    try:
        reviewRating = container.select_one("span.rating-other-user-rating span:first-of-type").text
    except Exception:
        reviewRating = ''
    reviewsDate = container.select_one("span.review-date").text
    reviewersName = container.select_one("span.display-name-link").text
    return {'movieName': movie,
            'title': reviewTitle,
            'text': reviewText,
            'rating': reviewRating,
            'date': reviewsDate,
            'reviewer': reviewersName}


def getInfoReviews(link, limit=1):
    semaphore.acquire()
    htmlParsed = htmlParser(link)
    reviewsDiv = htmlParsed.select("div.review-container")
    movie = htmlParsed.select_one("div.parent a").text
    print(f"Aici se gaseste pagina de recenzii pentru {movie: <50}{link : ^100}")
    loadDataBtn = htmlParsed.select_one("div.load-more-data")["data-key"]
    for i in range(limit + 1):
        if i != 0:
            requestUrl = link + f"_ajax?ref_=undefined&paginationKey={loadDataBtn}"
            htmlParsed = htmlParser(requestUrl)
            reviewsDiv = htmlParsed.select("div.review-container")
            loadDataBtn = htmlParsed.select_one("div.load-more-data")["data-key"]
        for j in range(len(reviewsDiv)):
            movies.append(buildInfoReview(movie, reviewsDiv[j]))
    semaphore.release()


if __name__ == '__main__':
    start = time.time()
    threads, htmlParsed = [], htmlParser("https://www.imdb.com/chart/top")
    lines = htmlParsed.select("table.chart tbody tr td.titleColumn a")
    moviesList = ["https://imdb.com" + line["href"] + "reviews/" for line in lines]
    for i in range(250):
        t = threading.Thread(target=getInfoReviews, args=(moviesList[i],))
        threads.append(t)
        t.start()

    for thread in threads:
        thread.join()

    with open("movies.json", "w") as f:
        json.dump(movies, f, indent=4)

    finish = time.time()
    print(f'Total time taken by the program: {round(finish - start)} seconds')
