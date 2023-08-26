import csv

with open('/data/private/zhangzhen/dir3/RanMASK/dataset/agnews/test.csv', newline='') as csvfile:
    with open("test.tsv", "w", newline="", encoding="utf-8") as file:
    
        writer = csv.writer(file, delimiter="\t", quotechar="", quoting=csv.QUOTE_NONE)
    

        
        reader = csv.reader(csvfile, delimiter=',')
        for i,row in enumerate(reader):
            if i==0:
                continue
            class_index = str(int(row[0])-1)
            title = row[1]
            description = row[2]

            sentence = title+' '+description
            label = class_index
            writer.writerow((sentence,label))