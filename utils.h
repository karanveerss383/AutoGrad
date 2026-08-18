#ifndef UTILS_H
#define UTILS_H

#include "stdlib.h"
#include "struct.h"
#include "weight_init.h"

void createLayer(Layer* L1, int weights_dist, int weights_type, int in_features, int out_features);

Value* init_Value(float* input, int input_size);

void createLayer(Layer* L1, int weights_dist, int weights_type, int in_features, int out_features){
  
  L1->neurons = malloc(sizeof(Neurons)*out_features);
  L1->fan_out = out_features;
  L1->fan_in = in_features;

  L1->neurons[0].weights = init_weights(weights_type, weights_dist, L1->fan_in, L1->fan_out);

  for (int i = 0; i < out_features; i++){

    L1->neurons[i].weights = L1->neurons[0].weights + (L1->fan_in * i);

    L1->neurons[i].bias.data = 0.0001; 
  }

}

Value* init_Value(float* input, int input_size){

  Value* result = malloc(input_size* sizeof(Value));

  for (int i = 0; i < input_size; i++){

    result[i].data = input[i];
    result[i].grad = 0.0f;
    result[i].left = NULL;
    result[i].right = NULL;
    result[i].op = ' ';
    result[i].backward = NULL;
    result[i].visited = 0;

  }
  return result;
}

#endif
