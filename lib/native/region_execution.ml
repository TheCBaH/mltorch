(* These are the two deliberate mutation exceptions in this executor.

   - [values] below is a fixed, per-key scalar-slot array. Dependent locals
     must become visible to later locals before the emitter runs; an immutable
     map/list would add allocation and lookup on the path whose purpose is to
     remove per-output work.
   - [counters] are optional test instrumentation. Ordinary execution passes
     none, so measurements do not introduce mutable state into the hot path.

   Tensor storage is separately mutable by definition: materialization writes
   each owned output once. *)
type counters = {
  mutable keys : int;
  mutable locals : int;
  mutable emitters : int;
  mutable loads : int;
  mutable reductions : int;
}

type lowered = {
  program : Region_program.t;
  slots : int Expr.Local_var.Map.t;
  local_count : int;
}

type t = Pixel_loop of Expr.Value.t | Region_loop of lowered

let counters () =
  { keys = 0; locals = 0; emitters = 0; loads = 0; reductions = 0 }

let lower program =
  match Region_program.pixel_expression program with
  | Some expression -> Pixel_loop expression
  | None ->
      let locals = Region_program.locals program in
      let slots =
        List.mapi (fun slot local -> (local.Region_local.id, slot)) locals
        |> List.to_seq |> Expr.Local_var.Map.of_seq
      in
      Region_loop { program; slots; local_count = List.length locals }

let widened r =
  Err.map_error (fun (e : Expr.Eval.error) -> (e :> Region_eval.error)) r

let expr_coord coord = Expr_bridge.coord_of_vec6 (Vec6.map Dim.to_int coord)

let instrument ?counters (env : Expr.Eval.Env.t) =
  match counters with
  | None -> env
  | Some counters ->
      {
        Expr.Eval.Env.load =
          (fun source coord ->
            counters.loads <- counters.loads + 1;
            env.Expr.Eval.Env.load source coord);
        load_index = env.Expr.Eval.Env.load_index;
      }

let evaluate_locals ?counters lowered ~env ~key =
  let values = Array.make lowered.local_count 0. in
  let local id =
    Option.map
      (fun slot -> values.(slot))
      (Expr.Local_var.Map.find_opt id lowered.slots)
  in
  let env = instrument ?counters env in
  let on_reduction () =
    Option.iter
      (fun counters -> counters.reductions <- counters.reductions + 1)
      counters
  in
  let rec fill slot = function
    | [] -> Err.return values
    | binding :: rest ->
        let open Err.Syntax in
        let* value =
          widened
            (Expr.Eval.value ~local ~on_reduction env ~output:(expr_coord key)
               binding.Region_local.value)
        in
        Option.iter
          (fun counters -> counters.locals <- counters.locals + 1)
          counters;
        values.(slot) <- value;
        fill (slot + 1) rest
  in
  fill 0 (Region_program.locals lowered.program)

let emit ?counters lowered ~env ~values ~output =
  let local id =
    Option.bind (Expr.Local_var.Map.find_opt id lowered.slots) (fun slot ->
        if slot < Array.length values then Some values.(slot) else None)
  in
  let env = instrument ?counters env in
  let on_reduction () =
    Option.iter
      (fun counters -> counters.reductions <- counters.reductions + 1)
      counters
  in
  Option.iter
    (fun counters -> counters.emitters <- counters.emitters + 1)
    counters;
  widened
    (Expr.Eval.value ~local ~on_reduction env ~output:(expr_coord output)
       (Region_program.output lowered.program))

let materialize ?counters lowered ~output_shape ~env =
  Err.Escape.with_escape @@ fun esc ->
  let tensor = Tensor.create output_shape in
  let partition = Region_program.partition lowered.program in
  Region_partition.fold_keys ~output_shape ~init:()
    ~f:(fun () key ->
      Option.iter (fun counters -> counters.keys <- counters.keys + 1) counters;
      let values =
        Err.Escape.or_throw esc (evaluate_locals ?counters lowered ~env ~key)
      in
      let () =
        Region_partition.fold_outputs ~output_shape ~key ~init:()
          ~f:(fun () output ->
            let value =
              Err.Escape.or_throw esc
                (emit ?counters lowered ~env ~values ~output)
            in
            Tensor.set_float tensor output value)
          partition
      in
      ())
    partition;
  tensor

let value_at lowered ~output_shape ~env ~output =
  Region_eval.value_at lowered.program ~output_shape ~env ~output
