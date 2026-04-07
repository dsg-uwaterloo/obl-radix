# Based on https://zenodo.org/records/15169905

movie_id = set()

f3 = open("title.basics.tsv", "r")
f4 = open("imdb_title.txt", "w")
d = f3.readlines()
for i in range(1, len(d)):
    lst = d[i].split('\t')
    if (lst[5][0] != '1') and (lst[5][0] != '2'):
        continue
    year = int(lst[5])
    tid = int(lst[0][2:])
    movie_id.add(tid)
    f4.write(str(tid)+" "+str(year)+"\n")
f3.close()
f4.close()


f1 = open("name.basics.tsv", "r")
f2 = open("imdb_name.txt", "w")
d = f1.readlines()
for i in range(1, len(d)):
    #print(str(i))
    lst = d[i].split('\t')
    if (lst[-1][0] != 't'):
        continue
    lst2 = lst[-1].split(',')
    lst2[-1] = lst2[-1][:-1]
    for o in range(len(lst2)):
        tid = int(lst2[o][2:])
        if tid in movie_id:
            f2.write(str(tid)+" "+str(int(lst[0][2:]))+"\n")
f1.close()
f2.close()
