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

let local_slots program = Region_slots.of_locals (Region_program.locals program)

let evaluate_locals program ~env ~slots ~key =
  let values = Array.make (Region_slots.total slots) 0. in
  let local, local_at = Region_slots.reader slots values in
  let rec fill = function
    | [] -> Err.return values
    | binding :: rest -> (
        let open Err.Syntax in
        let offset, count =
          Option.get (Region_slots.offset slots binding.Region_local.id)
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
  let local, local_at = Region_slots.reader slots values in
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
  let slots = local_slots program in
  let* values = evaluate_locals program ~env ~slots ~key in
  emit program ~env ~slots ~values ~output

(* Streams one key's [values] array at a time, exactly the traversal shape
   [Region_execution.materialize] already uses -- this is the reference
   path, so matching it changes peak scratch, not results. The previous
   [Hashtbl] retained every visited key's array for the whole call, so peak
   scratch was [keys * total_slots] rather than one array at a time. *)
let materialize program ~output_shape ~env =
  Err.Escape.with_escape @@ fun esc ->
  let slots = local_slots program in
  let tensor = Tensor.create output_shape in
  let partition = Region_program.partition program in
  Region_partition.fold_keys ~output_shape ~init:()
    ~f:(fun () key ->
      let values =
        Err.Escape.or_throw esc (evaluate_locals program ~env ~slots ~key)
      in
      Region_partition.fold_outputs ~output_shape ~key ~init:()
        ~f:(fun () output ->
          let value =
            Err.Escape.or_throw esc (emit program ~env ~slots ~values ~output)
          in
          Tensor.set_float tensor output value)
        partition)
    partition;
  tensor
