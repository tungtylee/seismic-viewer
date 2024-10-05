import os

def split_smallcsv_train_and_val(fn, prefix):
    arr = []
    head = None
    count = 0
    with open(fn) as f:
        for line in f:
            row = line.rstrip()
            if count == 0:
                head = row
            else:
                arr.append(row)
            count += 1

    nline = len(arr)
    half = int(nline / 2)
    # traingt.txt, trainlist.txt
    # valgt.txt, vallist.txt
    print(prefix + "traingt.txt")
    traingt = open(prefix + "traingt.txt", "w")
    trainlist = open(prefix + "trainlist.txt", "w")
    valgt = open(prefix + "valgt.txt", "w")
    vallist = open(prefix + "vallist.txt", "w")

    traingt.write(head + "\n")
    valgt.write(head + "\n")
    for i in range(nline - 1):
        if i < half:
            traingt.write(arr[i] + "\n")
            fn = arr[i].split(",")[0]
            trainlist.write(fn)
        else:
            valgt.write(arr[i] + "\n")
            fn = arr[i].split(",")[0]
            vallist.write(fn)
    traingt.close()
    trainlist.close()
    valgt.close()
    vallist.close()

if __name__ == "__main__":
    os.makedirs("algdev", exist_ok=True)
    fn = "space_apps_2024_seismic_detection/data/lunar/training/catalogs/apollo12_catalog_GradeA_final.csv"
    prefix = "algdev/lunar_"
    split_smallcsv_train_and_val(fn, prefix)
    fn = "space_apps_2024_seismic_detection/data/mars/training/catalogs/Mars_InSight_training_catalog_final.csv"
    prefix = "algdev/mars_"
    split_smallcsv_train_and_val(fn, prefix)
