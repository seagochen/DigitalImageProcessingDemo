import matplotlib.pyplot as plt
import numpy as np
from siki.basics import Exceptions


class ImageDescriptionToken(object):

    def __init__(self, image, title):
        self.img = image
        self.title = title

    def process(self):
        plt.imshow(self.img, cmap="gray")
        plt.title(self.title)
        plt.axis('off')


class HistDescriptionToken(object):

    def __init__(self, chart, title):
        self.chart = chart
        self.bins = 256
        self.title = title

    def process(self):
        plt.hist(self.chart, bins=self.bins, histtype='bar')
        plt.title(self.title)


class BarDescriptionToken(object):
    
    def __init__(self, chart, title):
        self.chart = chart
        self.bins = np.arange(len(chart))
        self.title = title

    def process(self):
        plt.bar(x=self.bins, height=self.chart)
        plt.title(self.title)


class PointsDescriptionToken(object):
    
    def __init__(self, x, y, title):
        self.x = x
        self.y = y
        self.title = title

    def process(self):
        plt.plot(self.x, self.y)
        plt.title(self.title)


class DiagramPlotter(object):

    def __init__(self):
        self.tokens = []

    def append_image(self, image, title="image"):
        img = ImageDescriptionToken(image, title)
        self.tokens.append(img)
        return self

    def append_hist(self, histogram, title="hist"):
        hist = HistDescriptionToken(histogram, title)
        self.tokens.append(hist)
        return self

    def append_pts(self, x, y, title="pts"):
        pts = PointsDescriptionToken(x, y, title)
        self.tokens.append(pts)
        return self

    def append_bars(self, bars, title="bar"):
        bars = BarDescriptionToken(bars, title)
        self.tokens.append(bars)
        return self

    def clean(self):
        del self.tokens
        self.tokens = []

    def show(self, nrows=0, ncols=0):
        if len(self.tokens) == 1:
            self.tokens[0].process()

        if len(self.tokens) > 1:
            if nrows * ncols != len(self.tokens):
                raise Exceptions.ArrayIndexOutOfBoundsException("Dimensions does not match the size of images")

            for i in range(nrows * ncols):
                plt.subplot(nrows, ncols, i + 1)
                self.tokens[i].process()

        # finally show the images
        plt.show()

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, key):
        if 0 <= key < len(self.tokens):
            return self.tokens[key]
        return 0

    def __iter__(self):
        return self.tokens.__iter__()
