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

let local_slots program =
  List.mapi
    (fun slot local -> (local.Region_local.id, slot))
    (Region_program.locals program)
  |> List.to_seq |> Expr.Local_var.Map.of_seq

let evaluate_locals program ~env ~slots ~key =
  let locals = Region_program.locals program in
  let values = Array.make (List.length locals) 0. in
  let local id =
    Option.map
      (fun slot -> values.(slot))
      (Expr.Local_var.Map.find_opt id slots)
  in
  let rec fill slot = function
    | [] -> Err.return values
    | binding :: rest ->
        let open Err.Syntax in
        let* value =
          widen_expr
            (Expr.Eval.value ~local env ~output:(expr_coord key)
               binding.Region_local.value)
        in
        values.(slot) <- value;
        fill (slot + 1) rest
  in
  fill 0 locals

let emit program ~env ~slots ~values ~output =
  let local id =
    Option.bind (Expr.Local_var.Map.find_opt id slots) (fun slot ->
        if slot < Array.length values then Some values.(slot) else None)
  in
  widen_expr
    (Expr.Eval.value ~local env ~output:(expr_coord output)
       (Region_program.output program))

let value_at program ~output_shape ~env ~output =
  let open Err.Syntax in
  let* key =
    widen_partition
      (Region_partition.key_of_output ~output_shape
         (Region_program.partition program)
         output)
  in
  let slots = local_slots program in
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
  let slots = local_slots program in
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
