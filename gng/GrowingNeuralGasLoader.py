class GrowingNeuralGasLoader(object):

    def loader(self):

        with open('graph.txt') as f:
            N = [[str(x) for x in line.split()] for line in f]
            print(N)

        with open('points.txt') as f:
            A = [[str(x) for x in line.split()] for line in f]
            print(A)



