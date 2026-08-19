#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "stdlib.h"
#include "struct.h"
#include "utils.h"
#include "weight_init.h"
#include "activations.h"
#include "backward.h"

Value* add(Value* a, Value* b){

  Value* result = malloc(sizeof(Value));

  result->data = a->data + b->data;
  result->op = '+';
  result->left = a;
  result->right = b;
  result->backward = add_backward;
  result->visited = 0;

  return result;

}

Value* mul(Value* a, Value* b){

  Value* result = malloc(sizeof(Value));

  result->data = a->data * b->data;
  result->op = '*';
  result->left = a;
  result->right = b;
  result->backward = mul_backward;
  result->visited = 0;

  return result;

}

Value** matmul(Layer* L, Value** input){

  Value** output = malloc(sizeof(Value*) * L->fan_out);

  for (int i = 0; i < L->fan_out; i++){

    Value* zero = malloc(sizeof(Value));

    zero->data = 0.0f;
    zero->grad = 0.0f;
    zero->left = NULL;
    zero->right = NULL;
    zero->op = ' ';
    zero->backward = noop_backward;
    zero->visited = 0;

    output[i] = zero;

    for (int j = 0; j < L->fan_in; j++){

      Value* multprod = mul(&L->neurons[i].weights[j], input[j]);

      output[i] = add(output[i], multprod);

    }

    output[i] = add(output[i], &L->neurons[i].bias);
  }

  return output;

}

Value*** batch_matmul(Layer* L, Value**input, size_t batch_size){

  Value*** batch_output = malloc(sizeof(Value**) * batch_size);

  for(int i = 0; i < batch_size; i++){
    batch_output[i] = matmul(L, input + (L->fan_in * i));
  }

  return batch_output;
}

#endif
