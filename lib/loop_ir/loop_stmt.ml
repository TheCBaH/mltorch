type t =
  | Alloc of Loop_array.t * Slot.count Slot.t
      (** Hoisted, allocated once and reused across keys. *)
  | Array_set of Loop_array.t * Loop_index.t * float Loop_expr.t
  | Assign : 'a Loop_carrier.t * Loop_temp.t * 'a Loop_expr.t -> t
  | Assign_index of Loop_temp.t * Loop_index.t
  | Assign_index_of_i64 of Loop_temp.t * int64 Loop_expr.t
      (** An int64 already inside the index domain (a gather's normalized index,
          after its range check) becomes an index. *)
  | Charge_scan_update
      (** One lane update against the scan meter, BEFORE the update body runs:
          exactly [max_updates] charges succeed and the next fails. *)
  | Fail_if of Loop_expr.pred * Loop_failure.t
  | For of {
      var : Loop_var.t;
      lo : Loop_index.t;
      hi : Loop_index.t;
      body : t list;
    }  (** Half-open: [lo] inclusive, [hi] exclusive. *)
  | If of Loop_expr.pred * t list * t list
  | Mark of Loop_mark.t
  | Reduce_sum of {
      var : Loop_var.t;
      lo : Loop_index.t;
      hi : Loop_index.t;
      acc : Loop_temp.t;
          (** the float temporary the sum is built in, and read after the node
              as its value *)
      seed : float;  (** the value [acc] starts from: [+0.] for a sum *)
      body : t list;
          (** the statements that evaluate one term, run before it is read *)
      term : float Loop_expr.t;  (** one term, a function of [var] *)
      at : Tensor_id.t option;
          (** the value whose lowering produced it; [None] for a sum recovered
              from an optimized program ({!Loop_sum.recover}) *)
    }
      (** A sum kept as one operation until the performance planner has seen
          it: the ordered left fold [acc <- seed; for var in [lo, hi): body;
          acc <- acc + term], with one [Reduction] mark per iteration. Only the
          planner and {!Loop_sum} read it; every other consumer is handed
          {!Loop_sum.expand}'s program, which is the loop it stands for. *)
  | Release_scan_state of int
      (** Gives back the [2 * width] live state a [Reserve_scan_state] took. *)
  | Reserve_scan_state of int
      (** [width] lanes of an inline scan's two rolling rows, against the
          nesting peak of live state ([max_state]). *)
  | Reset_meter
      (** A fresh meter: full update budget, no live state. One per Region key,
          and one per cell of a Pixel value that scans. *)
  | Store of {
      buffer : Loop_buffer.t;
      coord : Loop_index.coord;
      value : Loop_stored.t;
    }
  | Store_flat of {
      buffer : Loop_buffer.t;
      offset : Loop_index.t;
      value : Loop_stored.t;
    }
      (** A store at a dense row-major offset ([Vec6.offset]'s), in place of a
          per-axis coordinate: what loop collapsing leaves. Never on a
          per-channel quantized buffer, whose decode needs the C component. *)
