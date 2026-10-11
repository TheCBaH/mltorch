(** The groups a tensor metric is evaluated over.

    [Whole_tensor] is one group of every element. [Reduce_axes axes] reduces
    those axes within each group: the complementary axes (ascending) index the
    groups in row-major order, and a group's elements run row-major over the
    reduced axes (ascending, whatever order they were listed in). Reducing the
    feature axis of [batch; token; feature] therefore scores each token on its
    own. *)

type t

val make :
  shape:int64 list ->
  Inference_compare_schema.Tensor_metric.partition ->
  (t, Metric_config_error.t) Err.t
(** Rejects an empty axis list, duplicate or out-of-range axes, and a payload
    too large to index. *)

val groups : t -> int
val group_size : t -> int

val coords : t -> int -> int64 list
(** The complementary-axis indices of group [g]; empty for one whole-tensor
    group. *)

val indices : t -> int -> int array
(** Flat row-major element indices of group [g], in the group's order. *)
