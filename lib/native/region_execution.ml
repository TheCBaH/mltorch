(* These are the two deliberate mutation exceptions in this executor.

   - [values] below is a fixed, per-key SLOT array. Dependent locals must
     become visible to later locals before the emitter runs; an immutable
     map/list would add allocation and lookup on the path whose purpose is to
     remove per-output work. A scalar local occupies one slot; a vector local
     occupies [extent] consecutive slots, one per element; a scan (trace)
     local occupies its whole materialized trace, [(steps+1)*width] slots,
     row-major -- [slots] below maps each local's id to its own
     [(offset, count)] range within the one flat array, so a vector/trace read
     is a plain offset+index lookup, not a second data structure.
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
  mutable scans : int;
  mutable scan_updates : int;
}

type lowered = {
  program : Region_program.t;
  slots : Region_slots.t;
  output_shape : Vec6.shape;
  scan_limits : Expr.Scan_limits.t;
}

type t = Pixel_loop of Expr.Value.t | Region_loop of lowered

let counters () =
  {
    keys = 0;
    locals = 0;
    emitters = 0;
    loads = 0;
    reductions = 0;
    scans = 0;
    scan_updates = 0;
  }

(* Re-running [Region_program.check]/[Region_program.preflight] here is what
   makes the claim "an already-validated lowered Region program" true
   regardless of how [program] was produced: [Region_program.t] is only ever
   built through [create], which already checks it once, but
   [Region_program.with_output] is a raw record update with no check of its
   own -- [Kernel_eval.converted] uses exactly that to splice in a result
   conversion after construction, so its rewritten emitter has never been
   checked before it reaches here. [preflight] needs [output_shape] to derive
   one Region key's [outputs_per_key] from the program's own partition, which
   is also why [lowered] retains it: [materialize]/[value_at] can then stop
   taking a shape parameter at all, since it can no longer disagree with what
   was validated. [lowered] retains [scan_limits] for the same reason: it is
   what a fresh per-key/per-invocation [Expr.Scan_meter.t] is built from, so
   [materialize]/[value_at] need not take (or silently default) one either.

   [Region_slots.of_locals]'s running offset is plain [int] arithmetic, not
   [Int64]-checked: the [check] just above already bounds the SUM of every
   local's slot count against [max_local_slots] on [Int64], so by the time a
   program is lowered the total is already proven to fit -- this only has to
   use that proof, not re-derive it. *)
let validate ~max_size ~max_depth ~max_local_slots ~scan_limits ~output_shape
    program =
  let open Err.Syntax in
  let* () = Region_program.check ~max_size ~max_depth program in
  Region_program.preflight ~max_local_slots
    ~max_scan_state:(Expr.Scan_limits.max_state scan_limits)
    ~max_scan_updates:(Expr.Scan_limits.max_updates scan_limits)
    ~output_shape program

(* For a caller that already knows, structurally, that [program] is not a
   plain pixel expression -- e.g. it just matched [pixel_expression = None] --
   so it need not re-discover that fact by lowering and matching on [t]. *)
let lower_region ~max_size ~max_depth ~max_local_slots ~scan_limits
    ~output_shape program =
  let open Err.Syntax in
  let+ () =
    validate ~max_size ~max_depth ~max_local_slots ~scan_limits ~output_shape
      program
  in
  {
    program;
    slots = Region_slots.of_locals (Region_program.locals program);
    output_shape;
    scan_limits;
  }

let lower ~max_size ~max_depth ~max_local_slots ~scan_limits ~output_shape
    program =
  match Region_program.pixel_expression program with
  | Some expression ->
      (* A pixel program previously reached here with NO validation at all --
         not even [max_size]/[max_depth] -- since only the [None] branch
         called [check]. [Region_program.preflight] applies unchanged here
         too: [pixel_expression = Some] implies an empty local list and a
         singleton partition, so [outputs_per_key] is 1 and [per_key]
         collapses to exactly [U(t.output)] -- the same one-evaluation cost
         the per-output-coordinate runtime meter (see the scan design
         record's "Runtime metering") independently bounds. *)
      let open Err.Syntax in
      let+ () =
        validate ~max_size ~max_depth ~max_local_slots ~scan_limits
          ~output_shape program
      in
      Pixel_loop expression
  | None ->
      let open Err.Syntax in
      let+ lowered =
        lower_region ~max_size ~max_depth ~max_local_slots ~scan_limits
          ~output_shape program
      in
      Region_loop lowered

let widened r =
  Err.map_error (fun (e : Expr.Eval.error) -> (e :> Region_eval.error)) r

let widen_meter r =
  Err.map_error
    (fun (e : Expr.Scan_meter.error) ->
      (Expr.Eval.scan_meter_error e :> Region_eval.error))
    r

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

let rec each_lane f l width =
  if l >= width then Err.return ()
  else
    let open Err.Syntax in
    let* () = f l in
    each_lane f (l + 1) width

let evaluate_locals ?counters lowered ~env ~key ~scan_meter =
  let values = Array.make (Region_slots.total lowered.slots) 0. in
  let local, local_at = Region_slots.reader lowered.slots values in
  let scan = Region_slots.scan_reader lowered.slots values in
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
  let count_scan () =
    Option.iter (fun counters -> counters.scans <- counters.scans + 1) counters
  in
  let count_scan_update () =
    Option.iter
      (fun counters -> counters.scan_updates <- counters.scan_updates + 1)
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
                (Expr.Eval.value ~local ~local_at ~scan ~scan_meter
                   ~on_reduction env ~output:(expr_coord key) value)
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
                    (Expr.Eval.value ~local ~local_at ~scan ~scan_meter
                       ~reducer:[ (var, p) ]
                       ~on_reduction env ~output:(expr_coord key) body)
                in
                count_local ();
                values.(offset + p) <- value;
                each (p + 1)
            in
            let* () = each 0 in
            fill rest
        | Region_local.Rhs.Scan s ->
            (* Row 0 is the initializer, one evaluation per lane with [lane]
               bound, no update charge. Row [r] (1<=r<=steps) is the update,
               charged once per lane BEFORE evaluating its body, with [lane]
               and [step := r-1] both bound and [prev] rebound to read row
               [r-1] straight from the slots just written -- an ordinary
               [Local_at] resolver override, not [Eval.value]'s own inline
               scan machinery, since a trace local writes every row directly
               into its preflighted slot range rather than a rolling two-row
               buffer. *)
            count_scan ();
            let width = s.Expr.Scan.width in
            let row_offset r = offset + (r * width) in
            let init_lane l =
              let+ value =
                widened
                  (Expr.Eval.value ~local ~local_at ~scan ~scan_meter
                     ~reducer:[ (s.Expr.Scan.lane, l) ]
                     ~on_reduction env ~output:(expr_coord key) s.Expr.Scan.init)
              in
              count_local ();
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
              count_scan_update ();
              let+ value =
                widened
                  (Expr.Eval.value ~local ~local_at:local_at' ~scan ~scan_meter
                     ~reducer:
                       [ (s.Expr.Scan.lane, l); (s.Expr.Scan.step, step) ]
                     ~on_reduction env ~output:(expr_coord key)
                     s.Expr.Scan.update)
              in
              count_local ();
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
  fill (Region_program.locals lowered.program)

let emit ?counters lowered ~env ~values ~output ~scan_meter =
  let local, local_at = Region_slots.reader lowered.slots values in
  let scan = Region_slots.scan_reader lowered.slots values in
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
    (Expr.Eval.value ~local ~local_at ~scan ~scan_meter ~on_reduction env
       ~output:(expr_coord output)
       (Region_program.output lowered.program))

let materialize ?counters lowered ~env =
  Err.Escape.with_escape @@ fun esc ->
  let output_shape = lowered.output_shape in
  let tensor = Tensor.create output_shape in
  let partition = Region_program.partition lowered.program in
  Region_partition.fold_keys ~output_shape ~init:()
    ~f:(fun () key ->
      Option.iter (fun counters -> counters.keys <- counters.keys + 1) counters;
      (* One fresh meter per KEY, shared by every local (including a scan's
         own trace fill) and every emitter visiting that key -- the reset
         scope the scan design record specifies for Region materialization. *)
      let scan_meter = Expr.Scan_meter.create ~limits:lowered.scan_limits in
      let values =
        Err.Escape.or_throw esc
          (evaluate_locals ?counters lowered ~env ~key ~scan_meter)
      in
      let () =
        Region_partition.fold_outputs ~output_shape ~key ~init:()
          ~f:(fun () output ->
            let value =
              Err.Escape.or_throw esc
                (emit ?counters lowered ~env ~values ~output ~scan_meter)
            in
            Tensor.set_float tensor output value)
          partition
      in
      ())
    partition;
  tensor

let value_at lowered ~env ~output =
  Region_eval.value_at ~scan_limits:lowered.scan_limits lowered.program
    ~output_shape:lowered.output_shape ~env ~output
