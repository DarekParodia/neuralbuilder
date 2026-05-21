#pragma once

namespace neural {

    template <int InputNeuronN>
    struct neuron {
            float *inputs;
            float  output;

            float (*activation_func)(float x);
            float (*summing_func)();

            void process() {
                this->output = activation_func(summing_func());
            }
    };
} // namespace neural