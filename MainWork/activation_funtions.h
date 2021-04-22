#pragma once

float ReLu(float num) {
    return fmax(0.0f, num);
}

float Identity(float num) {
    return num;
}

float Sigmoid(float num) {
    return 1.0f / (1.0f + exp(-num));
}

float Tanh(float num) {
    return (exp(num) - exp(-num)) /
           (exp(num) + exp(-num));
}

float BinaryStep(float num) {
    if(num < 0.0f) {
        return 0.0f;
    }
    return 1.0f;
}
