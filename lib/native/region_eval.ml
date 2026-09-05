type error = [ Expr.Eval.error | Region_partition.error ]

let pp_error fmt : [< error ] -> unit = function
  | #Expr.Eval.error as error -> Expr.Eval.pp_error fmt error
  | #Region_partition.error as error -> Region_partition.pp_error fmt error

let expr_coord coord = Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int coord)

let widen_expr result =
  Err.map_error (fun (error : Expr.Eval.error) -> (error :> error)) result

let widen_partition result =
  Err.map_error
    (fun (error : Region_partition.error) -> (error :> error))
    result

(* [(offset, count)] per local, mirroring [Region_execution.lower_region]:
   a scalar occupies one slot, a vector [Region_local.Rhs.slot_count]-many.
   The running [offset] needs no bounds-checked fold here for the same reason
   it needs none there -- [Region_program.check] already proved the total
   fits, as a precondition of [program]'s own existence. *)
let local_slots program =
  List.fold_left
    (fun (slots, offset) local ->
      let count = Region_local.Rhs.slot_count local.Region_local.rhs in
      ( Expr.Local_var.Map.add local.Region_local.id (offset, count) slots,
        offset + count ))
    (Expr.Local_var.Map.empty, 0)
    (Region_program.locals program)

let total_slots slots =
  Expr.Local_var.Map.fold
    (fun _ (offset, count) m -> max m (offset + count))
    slots 0

let slot_reader slots values =
  let local id =
    match Expr.Local_var.Map.find_opt id slots with
    | Some (offset, _) when offset < Array.length values -> Some values.(offset)
    | _ -> None
  in
  let local_at id pos =
    match Expr.Local_var.Map.find_opt id slots with
    | Some (offset, count) when pos >= 0 && pos < count ->
        let i = offset + pos in
        if i < Array.length values then Some values.(i) else None
    | _ -> None
  in
  (local, local_at)

let evaluate_locals program ~env ~slots ~key =
  let values = Array.make (total_slots slots) 0. in
  let local, local_at = slot_reader slots values in
  let rec fill = function
    | [] -> Err.return values
    | binding :: rest -> (
        let open Err.Syntax in
        let offset, count =
          Expr.Local_var.Map.find binding.Region_local.id slots
        in
        match binding.Region_local.rhs with
        | Region_local.Rhs.Scalar value ->
            let* value =
              widen_expr
                (Expr.Eval.value ~local ~local_at env ~output:(expr_coord key)
                   value)
            in
            values.(offset) <- value;
            fill rest
        | Region_local.Rhs.Vector { var; body; _ } ->
            let rec each p =
              if p >= count then Err.return ()
              else
                let* value =
                  widen_expr
                    (Expr.Eval.value ~local ~local_at ~reducer:(var, p) env
                       ~output:(expr_coord key) body)
                in
                values.(offset + p) <- value;
                each (p + 1)
            in
            let* () = each 0 in
            fill rest)
  in
  fill (Region_program.locals program)

let emit program ~env ~slots ~values ~output =
  let local, local_at = slot_reader slots values in
  widen_expr
    (Expr.Eval.value ~local ~local_at env ~output:(expr_coord output)
       (Region_program.output program))

let value_at program ~output_shape ~env ~output =
  let open Err.Syntax in
  let* key =
    widen_partition
      (Region_partition.key_of_output ~output_shape
         (Region_program.partition program)
         output)
  in
  let slots, _ = local_slots program in
  let* values = evaluate_locals program ~env ~slots ~key in
  emit program ~env ~slots ~values ~output

let key_id key =
  ( Dim.to_int key.Vec6.n,
    Dim.to_int key.Vec6.t,
    Dim.to_int key.Vec6.d,
    Dim.to_int key.Vec6.h,
    Dim.to_int key.Vec6.w,
    Dim.to_int key.Vec6.c )

let materialize program ~output_shape ~env =
  Err.Escape.with_escape @@ fun esc ->
  let slots, _ = local_slots program in
  let cache = Hashtbl.create 16 in
  let value output =
    let key =
      match
        Region_partition.key_of_output ~output_shape
          (Region_program.partition program)
          output
      with
      | Ok key -> key
      | Error error -> Err.Escape.throw_error esc (error :> error Err.Error.t)
    in
    let values =
      match Hashtbl.find_opt cache (key_id key) with
      | Some values -> values
      | None ->
          let values =
            Err.Escape.or_throw esc (evaluate_locals program ~env ~slots ~key)
          in
          Hashtbl.add cache (key_id key) values;
          values
    in
    Err.Escape.or_throw esc (emit program ~env ~slots ~values ~output)
  in
  Tensor.materialize output_shape value
