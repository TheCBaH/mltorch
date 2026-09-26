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
