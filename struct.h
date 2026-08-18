#ifndef STRUCT_H
#define STRUCT_H

typedef struct Value Value;

struct Value {
  float data;
  float grad;
  Value* left;
  Value* right;
  void (*backward) (Value*);
  int visited;
  char op;
};

typedef struct {
  Value* weights;
  Value bias;
} Neurons;

typedef struct {
  Neurons* neurons;
  size_t fan_in;
  size_t fan_out;
} Layer;


#endif
