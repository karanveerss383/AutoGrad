#ifndef BACKWARD_H
#define BACKWARD_H

void add_backward(Value* self){

  self->left->grad += self->grad;
  self->right->grad += self->grad;

}

void mul_backward(Value* self){

  self->left-grad += self->grad * self->right->data;
  self->right->grad += self->grad * self->left->data;

}

void build_topo(Value* root, Value** topo, size_t* size){
  
  if (!root || root->visited) return;
  
  root->visited = 1;

  if (root->left) build_topo(root->left, topo, size);
  if (root->right) build_topo(root->right);

  topo[(*size)++] = root;
}

void backward(Value* root){
  
  Value* topo[10000];
  size_t size = 0;

  build_topo(root, topo, &size);

  for (int i = size-1; i >= 0; i--){
    topo[i].backward(topo[i]);
  }
}

void softmax_crossentropy_backward(Value* softmax_out, float* one_hot, int size){
  
  for(int i = 0; i < size; i++){
      Value* input = softmax_out[i].left;
      input->grad += softmax_out[i].data - one_hot[i];
      input->backward(input);
  }
}

#endif 
