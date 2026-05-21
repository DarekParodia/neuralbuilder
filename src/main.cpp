#include "logger.h"
#include "neuron.h"

#include <vector>

int main() {
    constexpr int                        neuronN = 6;
    constexpr int                        dataN   = 4;
    std::vector<float>                   input   = { 2.0f, 1.0f, 0.5f, 0.0f };

    Eigen::Matrix<float, neuronN, dataN> weights1;
    weights1 << 0.2f, 0.8f, -0.3f, 1.1f,
    -0.5f, 0.1f, 0.9f, -0.7f,
    1.3f, -0.4f, 0.6f, 0.2f,
    -0.1f, 1.5f, -0.8f, 0.3f,
    0.7f, -0.9f, 0.4f, -1.2f,
    -0.6f, 0.3f, 1.0f, -0.5f;

    Eigen::Matrix<float, 2, 6> weights2;
    weights2 << 0.3f, -0.7f, 0.5f, -0.2f, 0.9f,
    -0.4f, 0.3f, -0.7f, 10.5f, -0.2f, 0.9f, -0.4f;

    neural::layer lay1(neuronN, dataN, input.data(), weights1, neural::activations::relu);
    neural::layer lay2(2, neuronN, lay1.outputs.data(), weights2, neural::activations::relu);

    lay1.process();
    lay2.process();

    logger->info("Output: {:.4f}, {:.4f}", lay2.outputs[0], lay2.outputs[1]);
}