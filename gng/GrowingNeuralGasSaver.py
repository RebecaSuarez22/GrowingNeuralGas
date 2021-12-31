import GrowingNeuralGas

class GrowingNeuralGasSaver(object):


    def saver(self, x = GrowingNeuralGas()):
        N = str(x.N)
        A = str(x.A)

        nF = open("graph.txt", "w+")
        aF = open("points.txt", "w+")

        for _ in A:
            aF.write(A + "\n")

        for _ in N:
            nF.write(N + "\n")

        aF.close()
        nF.close()

