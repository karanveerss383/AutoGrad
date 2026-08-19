#include <stdio.h>
#include <math.h>
#include "backward.h"
#include "autograd.h"
#include "utils.h"
#include "activations.h"
#include "weight_init.h"

#define EPSILON 1e-4
#define TOLERANCE 1e-2   // relative error tolerance, loosen if needed for float precision

// Runs a full forward pass through two layers and returns a single scalar loss
// (sum of outputs) so we have one number to differentiate.
float forward_loss(Layer* L1, Layer* L2, float* raw_input, int input_size){
    Value** input = init_Value(raw_input, input_size);

    Value** l1_out = matmul(L1, input);
    Value** l1_relu = relu(l1_out, L1->fan_out);
    Value** l2_out = matmul(L2, l1_relu);

    float loss = 0.0f;
    for(int i = 0; i < L2->fan_out; i++){
        loss += l2_out[i]->data;
    }
    return loss;
}

// Numerically checks one specific weight in a layer against the analytically
// computed gradient sitting in weight->grad after a real backward() call.
void check_weight(const char* label, Value* weight, Layer* L1, Layer* L2,
                   float* raw_input, int input_size){

    float original = weight->data;

    weight->data = original + EPSILON;
    float loss_plus = forward_loss(L1, L2, raw_input, input_size);

    weight->data = original - EPSILON;
    float loss_minus = forward_loss(L1, L2, raw_input, input_size);

    weight->data = original;  // restore

    float numerical_grad = (loss_plus - loss_minus) / (2.0f * EPSILON);
    float analytical_grad = weight->grad;

    float denom = fabsf(numerical_grad) + fabsf(analytical_grad) + 1e-8f;
    float relative_error = fabsf(numerical_grad - analytical_grad) / denom;

    printf("%-12s  numerical: %10.6f   analytical: %10.6f   rel_error: %8.6f   %s\n",
           label, numerical_grad, analytical_grad, relative_error,
           (relative_error < TOLERANCE) ? "PASS" : "FAIL");
}

int main(){
    int input_size = 6;
    float raw_input[] = {1, 2, 3, 4, 5, 6};

    Layer L1, L2;
    createLayer(&L1, 1, 1, input_size, 9);   // 6 -> 9
    createLayer(&L2, 1, 1, 9, 6);            // 9 -> 6

    // --- real forward + backward pass, seeding grad from the scalar loss ---
    Value** input = init_Value(raw_input, input_size);
    Value** l1_out = matmul(&L1, input);
    Value** l1_relu = relu(l1_out, L1.fan_out);
    Value** l2_out = matmul(&L2, l1_relu);

    // build a single scalar loss = sum of L2 outputs, so backward() has one root
    Value* loss = l2_out[0];
    for(int i = 1; i < L2.fan_out; i++){
        loss = add(loss, l2_out[i]);
    }
    loss->grad = 1.0f;
    do_backward(loss);

    printf("=== Gradient Check ===\n");
    printf("Comparing analytical grad (from backward()) vs numerical grad (finite differences)\n\n");

    // check a handful of weights across both layers
    check_weight("L1 n0 w0", &L1.neurons[0].weights[0], &L1, &L2, raw_input, input_size);
    check_weight("L1 n0 w3", &L1.neurons[0].weights[3], &L1, &L2, raw_input, input_size);
    check_weight("L1 n4 w1", &L1.neurons[4].weights[1], &L1, &L2, raw_input, input_size);
    check_weight("L1 n8 w5", &L1.neurons[8].weights[5], &L1, &L2, raw_input, input_size);
    check_weight("L2 n0 w0", &L2.neurons[0].weights[0], &L1, &L2, raw_input, input_size);
    check_weight("L2 n2 w4", &L2.neurons[2].weights[4], &L1, &L2, raw_input, input_size);
    check_weight("L2 n5 w8", &L2.neurons[5].weights[8], &L1, &L2, raw_input, input_size);

    printf("\nIf all PASS, backward() is producing mathematically correct gradients.\n");
    printf("If any FAIL, the bug is in mult_back, add_back, relu backward, or the chain/accumulation logic.\n");

    return 0;
}
