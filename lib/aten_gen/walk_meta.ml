(* Hand-crafted meta augmenting native_functions.yaml for the random-walk
   generator (Aten_walk_gen). See walk_meta_entry.ml for what [t]'s fields
   mean; the op entries themselves live in walk_meta_conv.ml, _pool.ml,
   _reduce.ml, _pointwise.ml, _shape.ml, _linalg.ml, _norm.ml, and
   _attention.ml. This file is now just the registry.. *)

type t = Walk_meta_entry.t

open Walk_meta_entry
open Walk_meta_conv
open Walk_meta_pool
open Walk_meta_reduce
open Walk_meta_pointwise
open Walk_meta_shape
open Walk_meta_linalg
open Walk_meta_norm
open Walk_meta_attention

let entries =
  [
    conv2d;
    convolution;
    conv2d_padding;
    linear;
    rms_norm;
    layer_norm;
    native_layer_norm;
    max_pool2d;
    max_pool2d_with_indices;
    avg_pool2d;
    adaptive_avg_pool2d;
    adaptive_max_pool2d;
    mean_dim;
    amax;
    sum_dim_int_list;
    softmax_int;
    pow_tensor_scalar;
    linalg_vector_norm;
    clamp;
    clamp_min;
    hardtanh;
    unbind_int;
    pad;
    slice_tensor;
    sub_tensor;
    add_tensor;
    add_inplace_tensor;
    mul_tensor;
    div_tensor;
    mul_scalar;
    view_default;
    unsafe_view;
    expand;
    transpose_int;
    permute;
    addmm;
    bmm;
    matmul;
    native_batch_norm;
    sdpa;
  ]

let find target = List.find_opt (fun e -> e.target = target) entries
