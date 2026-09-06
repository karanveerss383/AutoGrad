#ifndef WEIGHT_INIT_H
#define WEIGHT_INIT_H

#include "math.h"
#include "time.h"
#include "struct.h"
#include "stdio.h"
#include "stdlib.h"

#define DEFAULT (1.0f/2.0f)
#define XAVIER 1
#define KAIMING 3
#define UNIFORM 2
#define NORMAL 1

Tensor* init_weights(size_t type, size_t dist, int fan_in, int fan_out){
  
  int scale = type * 2;
  double var = (double)scale / (double)(fan_in + fan_out);
  double std_dev = sqrt(var);
  
  float* weights;

  for (int i = 0; i < fan_out; i++){
  
    for (int j = 0; j < fan_in; j++){
    
      weights[i * fan_in + j] = ((float)rand() / RAND_MAX) * (dist * std_dev) - ((dist - 1) * std_dev);
    } 
  }

  Tensor* return_tensor = create_tensor(2, (int[]){fan_out, fan_in}, weights);

  return return_tensor;
}

#endif

