#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

#include "struct.h"

Value* softmax(Value** input, int catagories){

  Value* result = malloc(sizeof(Value) * catagories);

  double exp_sum = 0;
  float max = input[0]->data;

  for (int i = 1; i < catagories; i++){

    if (input[i]->data > max){
      max = input[i]->data;
    }

  }

  for (int i = 0; i < catagories; i++){

    exp_sum += expf(input[i]->data - max);

  }

  printf("exp_sum : %f\n", exp_sum);

  for (int i = 0; i < catagories; i ++){

    result[i].data = (float)(expf(input[i]->data - max) / exp_sum);
    result[i].op = 'e';
    result[i].left = (input[i]);
    result[i].right = NULL;
    result[i].backward = NULL;
    result[i].visited = 0;
  
  }
  return result;
}

Value* relu(Value** input, int size){

  Value* result = malloc(sizeof(Value) * size);

  for (int i = 0; i < size; i++){

    result[i].data = (input[i]->data < 0) ? 0 : input[i]->data;
    result[i].op = 'r';
    result[i].left = input[i];
    result[i].right = NULL;
    result[i].backward = NULL;
    result[i].visited = 0;
  
  }

  return result;
}

#endif
