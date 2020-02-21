import requests

r = requests.post("https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_string.php", data={'netid': 'skcheun2', 'name': 'sam'})

print(r.status_code, r.reason)

#print(r.text)


#news = "x" * 400  # gives you "xxxxxxxxxx"

news = (r.text[498*i] for i in range(400))

print(news)
print(*(char for char in news), sep='')
