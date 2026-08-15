#ifndef UTILS_H
#define UTILS_H

#include "stdlib.h"
#include "struct.h"
#include "weight_init.h"

void createLayer(Layer* L1, int weights_dist, int weights_type, int in_features, int out_features);

void createLayer(Layer* L1, int weights_dist, int weights_type, int in_features, int out_features){
  
  L1->n = malloc(sizeof(Neurons)*out_features);
  L1->fan_out = out_features;
  L1->fan_in = in_features;

  L1->n[0].weights = init_weights(weights_type, weights_dist, L1->fan_in, L1->fan_out);

  for (int i = 0; i < out_features; i++){

    L1->n[i].weights = L1->n[0].weights + (L1->fan_in * i);

    L1->n[i].bias.data = 0.0001; 
  }

}

#endif
