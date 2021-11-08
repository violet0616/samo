import types

from optimiser.annealing import SimulatedAnnealing
import backend.fpgaconvnet.parser as parser

if __name__ == "__main__":

    # parse the example network
    graph = parser.parse("models/simple.onnx")

    # platform
    platform = {
        "LUT" : 437200,
        "DSP" : 900,
        "BRAM" : 1090,
        "FF" : 218600
    }

    # perform optimisation on the computation graph
    opt = SimulatedAnnealing(graph)

    opt.network.platform = platform

    # turn off inter layer folding matching
    # opt.network.constraints["inter_layer_matching"] = False

    print("latency (before): ", opt.network.eval_latency())
    opt.optimise()

    print("latency (after): ", opt.network.eval_latency())
