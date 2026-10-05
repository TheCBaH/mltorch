(* The runtime failures a program can raise, each an explicit [Fail_if] (or, for
   the meter, its own statement). A site names WHERE and carries the expressions
   the interpreter evaluates only if the check fires, so the reported row holds
   the concrete values [Kernel_eval] reports without the site being prose. A
   backend never discovers a failure through a host exception. *)
type t =
  | Gather_out_of_range of { raw : int64 Loop_expr.t; extent : int }
  | I64_division_by_zero
  | I64_division_overflow
  | I64_from_float of { value : float Loop_expr.t }
  | Index_overflow of { index : Loop_index.t }
  | Load_out_of_range of { buffer : Loop_buffer.t; coord : Loop_index.coord }
  | Local_out_of_range of {
      local : Expr.Local_var.t;
      index : Loop_index.t;
      extent : int;
    }
  | Scan_lane_out_of_range of {
      local : Expr.Local_var.t option;
      row : Loop_index.t;
      lane : Loop_index.t;
      extent : int;
    }
  | Scan_row_out_of_range of {
      local : Expr.Local_var.t option;
      row : Loop_index.t;
      lane : Loop_index.t;
      extent : int;
    }

(* Two failures name the same static part of a record: the same kind of failure
   at the same local variable. A record carries only the index of its site, and
   a decoder reads from the entry nothing but that, so any entry that agrees
   decodes it. *)
let same_site a b =
  let var_equal x y =
    match (x, y) with
    | Some x, Some y -> Expr.Local_var.equal x y
    | None, None -> true
    | Some _, None | None, Some _ -> false
  in
  match (a, b) with
  | Local_out_of_range a, Local_out_of_range b ->
      Expr.Local_var.equal a.local b.local
  | Scan_lane_out_of_range a, Scan_lane_out_of_range b ->
      var_equal a.local b.local
  | Scan_row_out_of_range a, Scan_row_out_of_range b ->
      var_equal a.local b.local
  | ( ( Gather_out_of_range _ | I64_division_by_zero | I64_division_overflow
      | I64_from_float _ | Index_overflow _ | Load_out_of_range _
      | Local_out_of_range _ | Scan_lane_out_of_range _
      | Scan_row_out_of_range _ ),
      _ ) ->
      false
