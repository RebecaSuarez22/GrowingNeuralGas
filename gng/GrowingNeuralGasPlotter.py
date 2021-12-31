import matplotlib.pyplot

import tensorflow as tf

class GrowingNeuralGasPlotter(object):
    @staticmethod
    def plotGraphConnectedComponent(pathFigure, nameFigure, A, N):
        figure, axis = matplotlib.pyplot.subplots()

        x = [A[index][0].numpy() for index in tf.range(A.shape[0])]
        y = [A[index][1].numpy() for index in tf.range(A.shape[0])]

        graphZero = axis.scatter(x, y, color='black', marker='.')

        #matplotlib.pyplot.show()
        figure.savefig(pathFigure + '//' + nameFigure + '.png', transparent=False, dpi=80, bbox_inches="tight")
        figure.savefig(pathFigure + '//' + nameFigure + '.svg')
        #matplotlib.pyplot.close(figure)