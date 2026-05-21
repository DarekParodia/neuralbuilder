#pragma once

#include <Eigen/Dense>

namespace neural {

    namespace activations {
        void relu(float *input, float *output, int n) {
            for(int i = 0; i < n; i++)
                output[i] = std::max(0.0f, input[i]);
        }
    }; // namespace activations

    template <int NeuronN, int InputN>
    struct layer {
            float *inputs;
            float  sums[NeuronN];
            float  outputs[NeuronN];

            // n column = wieights of n neuron
            Eigen::Matrix<float, NeuronN, InputN> *weights;

            void (*activation_func)(float *input, float *output, int n);

            void sum(const float *input, float *output, const Eigen::Matrix<float, NeuronN, InputN> &w) {
                Eigen::Map<const Eigen::Vector<float, InputN>> in_vec(input);
                Eigen::Map<Eigen::Vector<float, NeuronN>>      out_vec(output);

                out_vec = w * in_vec; // (NeuronN x InputN) * (InputN) = (NeuronN)
            }

            void process() {
                // Firstly add weights to neurons
                sum(inputs, sums, *weights);
                activation_func(sums, outputs, NeuronN);
            }

            layer(float *inputs, Eigen::Matrix<float, NeuronN, InputN> *weights, void (*activation_func)(float *input, float *output, int n)) {
                this->inputs          = inputs;
                this->weights         = weights;
                this->activation_func = activation_func;
            }
    };
} // namespace neural