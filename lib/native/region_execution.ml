(* These are the two deliberate mutation exceptions in this executor.

   - [values] below is a fixed, per-key SLOT array. Dependent locals must
     become visible to later locals before the emitter runs; an immutable
     map/list would add allocation and lookup on the path whose purpose is to
     remove per-output work. A scalar local occupies one slot; a vector local
     occupies [extent] consecutive slots, one per element -- [slots] below
     maps each local's id to its own [(offset, count)] range within the one
     flat array, so a vector read is a plain offset+index lookup, not a
     second data structure.
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

type lowered = { program : Region_program.t; slots : Region_slots.t }
type t = Pixel_loop of Expr.Value.t | Region_loop of lowered

let counters () =
  { keys = 0; locals = 0; emitters = 0; loads = 0; reductions = 0 }

(* For a caller that already knows, structurally, that [program] is not a
   plain pixel expression -- e.g. it just matched [pixel_expression = None] --
   so it need not re-discover that fact by lowering and matching on [t].

   [Region_slots.of_locals]'s running offset is plain [int] arithmetic, not
   [Int64]-checked: [Region_program.check] (run once, at [create]) already
   bounds the SUM of every local's slot count against [max_size] on [Int64]
   before any [Region_program.t] can exist, so by the time a program reaches
   here the total is already proven to fit -- this only has to use that
   proof, not re-derive it. *)
let lower_region program =
  { program; slots = Region_slots.of_locals (Region_program.locals program) }

let lower program =
  match Region_program.pixel_expression program with
  | Some expression -> Pixel_loop expression
  | None -> Region_loop (lower_region program)

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
  let values = Array.make (Region_slots.total lowered.slots) 0. in
  let local, local_at = Region_slots.reader lowered.slots values in
  let env = instrument ?counters env in
  let on_reduction () =
    Option.iter
      (fun counters -> counters.reductions <- counters.reductions + 1)
      counters
  in
  let count_local () =
    Option.iter
      (fun counters -> counters.locals <- counters.locals + 1)
      counters
  in
  let rec fill = function
    | [] -> Err.return values
    | binding :: rest -> (
        let open Err.Syntax in
        (* Dispatch on the local's DECLARED right-hand side, never on its
           numeric slot count -- a [Vector] local's own extent can be 1 (e.g.
           SDPA's [Wk = 1]), which [lower_region] gives the identical
           [(offset, 1)] slot range a [Scalar] local gets. Matching on the
           count alone silently took the scalar branch for that vector, which
           evaluates the body with no [~reducer] bound: any free occurrence of
           the vector's own per-element binder (present by construction -- see
           [Region_program.Builder.vector]) then raised [Unbound_reducer]. *)
        let offset, count =
          Option.get (Region_slots.offset lowered.slots binding.Region_local.id)
        in
        match binding.Region_local.rhs with
        | Region_local.Rhs.Scalar value ->
            let* value =
              widened
                (Expr.Eval.value ~local ~local_at ~on_reduction env
                   ~output:(expr_coord key) value)
            in
            count_local ();
            values.(offset) <- value;
            fill rest
        | Region_local.Rhs.Vector { var; body; _ } ->
            (* A vector local's body is evaluated once PER POSITION, exactly
               the loop shape [Reduce]'s own fold uses -- the body mentions
               its binder FREE (never under a nested [Reduce]), so each
               iteration seeds it with [~reducer], the same role
               [Reduce]'s internal per-iteration [bound] closure plays. *)
            let rec each p =
              if p >= count then Err.return ()
              else
                let* value =
                  widened
                    (Expr.Eval.value ~local ~local_at ~reducer:(var, p)
                       ~on_reduction env ~output:(expr_coord key) body)
                in
                count_local ();
                values.(offset + p) <- value;
                each (p + 1)
            in
            let* () = each 0 in
            fill rest)
  in
  fill (Region_program.locals lowered.program)

let emit ?counters lowered ~env ~values ~output =
  let local, local_at = Region_slots.reader lowered.slots values in
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
    (Expr.Eval.value ~local ~local_at ~on_reduction env
       ~output:(expr_coord output)
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
