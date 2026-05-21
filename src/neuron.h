#pragma once

#include <algorithm>
#include <vector>

#include <Eigen/Dense>

namespace neural {

    namespace activations {
        inline void relu(float *input, float *output, int n) {
            for(int i = 0; i < n; i++)
                output[i] = std::max(0.0f, input[i]);
        }
    } // namespace activations

    struct layer {
            using activation_func_t        = void (*)(float *input, float *output, int n);

            int               neuron_count = 0;
            int               input_count  = 0;
            const float      *inputs       = nullptr;
            Eigen::VectorXf   sums;
            Eigen::VectorXf   outputs;
            Eigen::MatrixXf   weights;
            activation_func_t activation_func = nullptr;

            template <typename Derived>
            layer(int                         neuron_count,
            int                               input_count,
            const float                      *inputs,
            const Eigen::MatrixBase<Derived> &weights,
            activation_func_t                 activation_func) :
                neuron_count(neuron_count),
                input_count(input_count),
                inputs(inputs),
                sums(neuron_count),
                outputs(neuron_count),
                weights(weights),
                activation_func(activation_func) {
            }

            void sum(const float *input, Eigen::VectorXf &output, const Eigen::MatrixXf &w) {
                Eigen::Map<const Eigen::VectorXf> in_vec(input, input_count);
                output = w * in_vec;
            }

            void process() {
                sum(inputs, sums, weights);
                activation_func(sums.data(), outputs.data(), neuron_count);
            }
    };

    struct network {
            std::vector<layer> layers;

            network(int layers) {
            }
    };

} // namespace neural