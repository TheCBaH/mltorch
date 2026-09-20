(* [int64]-segment counterpart of [Region_slots.t]: lays out scalar/vector
   [Region_local_i64.t] declarations into flat offsets within one [int64
   array], the same (offset, count) scheme [Region_slots] uses for float --
   no [scans] table, since [Region_local_i64] has no [Scan] variant. *)
type t = { slots : (int * int) Expr.Local_var.Map.t; total : int }

let of_locals locals =
  let slots, total =
    List.fold_left
      (fun (slots, offset) (local : Region_local_i64.t) ->
        let count =
          Region_local_i64.Rhs.slot_count local.Region_local_i64.rhs
        in
        ( Expr.Local_var.Map.add local.Region_local_i64.id (offset, count) slots,
          offset + count ))
      (Expr.Local_var.Map.empty, 0)
      locals
  in
  { slots; total }

let total t = t.total
let offset t id = Expr.Local_var.Map.find_opt id t.slots

let reader t (values : int64 array) =
  let local id =
    match offset t id with
    | Some (offset, _) when offset < Array.length values -> Some values.(offset)
    | _ -> None
  in
  let local_at id pos =
    match offset t id with
    | Some (offset, count) when pos >= 0 && pos < count ->
        let i = offset + pos in
        if i < Array.length values then Some values.(i) else None
    | _ -> None
  in
  (local, local_at)

(* Fills one [int64 array] segment, one Region key at a time -- the int64
   twin of [Region_eval.evaluate_locals], minus the [Scan]/[scan_meter]
   machinery neither [Region_local_i64.Rhs] nor this segment carries. Each
   local's stored body is resolved through [Expr.Eval.value_i64], the EXACT
   int64 entry point (added alongside [Expr.Eval.value] in this same slice):
   routing an I64 body through [Expr.Eval.value] instead would mean wrapping
   it in [Value.i64_to_float] first, silently losing precision above 2^53 --
   exactly what this whole carrier exists to rule out. [~local_i64]/
   [~local_at_i64] are this segment's own [reader], already filled up to the
   current local -- so a later local's body may read an earlier one, the
   same left-to-right visibility [Region_eval.evaluate_locals] gives float
   locals. *)
let fill (locals : Region_local_i64.t list) t ~env ~output =
  let values = Array.make t.total 0L in
  let local_i64, local_at_i64 = reader t values in
  List.iter
    (fun (decl : Region_local_i64.t) ->
      let offset, count = Option.get (offset t decl.Region_local_i64.id) in
      match decl.Region_local_i64.rhs with
      | Region_local_i64.Rhs.Scalar value ->
          values.(offset) <-
            Err.or_raise ~pp_error:Expr.Eval.pp_error
              (Expr.Eval.value_i64 ~local_i64 ~local_at_i64 env ~output value)
      | Region_local_i64.Rhs.Vector { var; body; _ } ->
          for p = 0 to count - 1 do
            values.(offset + p) <-
              Err.or_raise ~pp_error:Expr.Eval.pp_error
                (Expr.Eval.value_i64 ~local_i64 ~local_at_i64
                   ~reducer:[ (var, p) ]
                   env ~output body)
          done)
    locals;
  values
