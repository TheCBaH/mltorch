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

let widen_meter result =
  Err.map_error
    (fun (e : Expr.Scan_meter.error) -> (Expr.Eval.scan_meter_error e :> error))
    result

let local_slots program = Region_slots.of_locals (Region_program.locals program)

(* Runs each lane of one scan row through [f] in order (lane order carries no
   semantics of its own -- every lane at a step reads the same COMPLETED prior
   row -- but running them in a fixed order keeps this deterministic and
   simple). *)
let rec each_lane f l width =
  if l >= width then Err.return ()
  else
    let open Err.Syntax in
    let* () = f l in
    each_lane f (l + 1) width

let evaluate_locals program ~env ~slots ~key ~scan_meter =
  let values = Array.make (Region_slots.total slots) 0. in
  let local, local_at = Region_slots.reader slots values in
  let scan = Region_slots.scan_reader slots values in
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
                (Expr.Eval.value ~local ~local_at ~scan ~scan_meter env
                   ~output:(expr_coord key) value)
            in
            values.(offset) <- value;
            fill rest
        | Region_local.Rhs.Vector { var; body; _ } ->
            let rec each p =
              if p >= count then Err.return ()
              else
                let* value =
                  widen_expr
                    (Expr.Eval.value ~local ~local_at ~scan ~scan_meter
                       ~reducer:[ (var, p) ]
                       env ~output:(expr_coord key) body)
                in
                values.(offset + p) <- value;
                each (p + 1)
            in
            let* () = each 0 in
            fill rest
        | Region_local.Rhs.Scan s ->
            (* Row 0 is the initializer, one evaluation per lane with [lane]
               bound; no update charge. Row [r] (1<=r<=steps) is the update,
               charged once per lane BEFORE evaluating its body, with [lane]
               and [step := r-1] both bound and [prev] rebound to read row
               [r-1] from the slots just written -- an ordinary [Local_at]
               resolver override, not [Eval.value]'s own inline scan
               machinery, since a trace local writes every row directly into
               its preflighted slot range rather than a rolling two-row
               buffer. *)
            let width = s.Expr.Scan.width in
            let row_offset r = offset + (r * width) in
            let init_lane l =
              let+ value =
                widen_expr
                  (Expr.Eval.value ~local ~local_at ~scan ~scan_meter
                     ~reducer:[ (s.Expr.Scan.lane, l) ]
                     env ~output:(expr_coord key) s.Expr.Scan.init)
              in
              values.(row_offset 0 + l) <- value
            in
            let update_lane ~step l =
              let local_at' v pos =
                if Expr.Local_var.equal v s.Expr.Scan.prev then
                  if pos >= 0 && pos < width then
                    Some values.(row_offset step + pos)
                  else None
                else local_at v pos
              in
              let* () =
                widen_meter (Expr.Scan_meter.charge_update scan_meter)
              in
              let+ value =
                widen_expr
                  (Expr.Eval.value ~local ~local_at:local_at' ~scan ~scan_meter
                     ~reducer:
                       [ (s.Expr.Scan.lane, l); (s.Expr.Scan.step, step) ]
                     env ~output:(expr_coord key) s.Expr.Scan.update)
              in
              values.(row_offset (step + 1) + l) <- value
            in
            let rec each_row r =
              if r > s.Expr.Scan.steps then Err.return ()
              else
                let* () =
                  each_lane
                    (if r = 0 then init_lane else update_lane ~step:(r - 1))
                    0 width
                in
                each_row (r + 1)
            in
            let* () = each_row 0 in
            fill rest)
  in
  fill (Region_program.locals program)

let emit program ~env ~slots ~values ~output ~scan_meter =
  let local, local_at = Region_slots.reader slots values in
  let scan = Region_slots.scan_reader slots values in
  widen_expr
    (Expr.Eval.value ~local ~local_at ~scan ~scan_meter env
       ~output:(expr_coord output)
       (Region_program.output program))

let value_at ?(scan_limits = Expr.Scan_limits.default) program ~output_shape
    ~env ~output =
  let open Err.Syntax in
  let* key =
    widen_partition
      (Region_partition.key_of_output ~output_shape
         (Region_program.partition program)
         output)
  in
  let slots = local_slots program in
  let scan_meter = Expr.Scan_meter.create ~limits:scan_limits in
  let* values = evaluate_locals program ~env ~slots ~key ~scan_meter in
  emit program ~env ~slots ~values ~output ~scan_meter

(* Streams one key's [values] array at a time, exactly the traversal shape
   [Region_execution.materialize] already uses -- this is the reference
   path, so matching it changes peak scratch, not results. The previous
   [Hashtbl] retained every visited key's array for the whole call, so peak
   scratch was [keys * total_slots] rather than one array at a time. A fresh
   [scan_meter] is created per key, shared by every local (including a scan's
   own trace fill) and the emitter for that key -- the reset scope the scan
   design record specifies for Region materialization. *)
let materialize ?(scan_limits = Expr.Scan_limits.default) program ~output_shape
    ~env =
  Err.Escape.with_escape @@ fun esc ->
  let slots = local_slots program in
  let tensor = Tensor.create output_shape in
  let partition = Region_program.partition program in
  Region_partition.fold_keys ~output_shape ~init:()
    ~f:(fun () key ->
      let scan_meter = Expr.Scan_meter.create ~limits:scan_limits in
      let values =
        Err.Escape.or_throw esc
          (evaluate_locals program ~env ~slots ~key ~scan_meter)
      in
      Region_partition.fold_outputs ~output_shape ~key ~init:()
        ~f:(fun () output ->
          let value =
            Err.Escape.or_throw esc
              (emit program ~env ~slots ~values ~output ~scan_meter)
          in
          Tensor.set_float tensor output value)
        partition)
    partition;
  tensor
