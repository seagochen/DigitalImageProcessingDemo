import csv


def save_pts(x_pts, y_pts, filename="Data/data.csv"):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for item in zip(x_pts, y_pts):
            csv_writer.writerow(item)


def load_pts(filename="Data/data.csv"):
    x_pts = []
    y_pts = []
    with open(filename, newline='') as csvfile:
        csv_reader = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in csv_reader:
            x_pts.append(float(row[0]))
            y_pts.append(float(row[1]))
    return x_pts, y_pts
