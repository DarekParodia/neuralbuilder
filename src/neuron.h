#pragma once

#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include "logger.h"

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
            float             *inputs;
            int                input_count;

            float             *outputs;
            int                output_count;

            network() {
            }

            void print() {
                logger->info("Network info:");
                logger->info("Inputs: {}", input_count);
                logger->info("Outputs: {}", output_count);
                logger->info("Layers: {}", layers.size());
            }
            void load_json(std::string filename) {
                std::ifstream  f(filename);
                nlohmann::json data = nlohmann::json::parse(f);

                // network info
                int layer_count     = data["layers"].size();
                input_count         = data["layers"][0]["weights"][0].size();
                output_count        = data["layers"][layer_count - 1]["biases"].size();
                inputs              = new float[input_count];

                for(size_t i = 0; i < layer_count; i++) {
                    auto         c_lay          = data["layers"][i];
                    int          c_neuron_count = c_lay["biases"].size();
                    int          c_input_count  = 0;
                    const float *c_inputs       = nullptr;


                    if(i == 0) {
                        // first layer
                        c_input_count = input_count;
                        c_inputs      = inputs;
                    } else {
                        c_input_count = layers.at(i - 1).neuron_count;
                        c_inputs      = layers.at(i - 1).outputs.data();
                    }
                    if(i == layer_count - 1) {
                        // last layer
                    }

                    Eigen::MatrixXf c_weights;
                    c_weights.resize(c_neuron_count, c_input_count);

                    for(size_t j = 0; j < c_neuron_count; j++) {
                        auto row_weights = c_lay["weights"][j].get<std::vector<float>>();
                        for(size_t k = 0; k < c_input_count; k++)
                            c_weights(j, k) = row_weights[k];
                    }

                    layers.emplace_back(
                    c_neuron_count,
                    c_input_count,
                    c_inputs,
                    c_weights,
                    neural::activations::relu);
                }
            }
            void create() {
            }
    };


} // namespace neural