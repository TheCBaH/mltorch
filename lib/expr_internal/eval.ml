(* The reference evaluator. Its shared plumbing (index evaluation, the
   [error]/[Env] types, [vchk]) moved verbatim to [eval_common.ml] in the
   tail-call conversion's Stage 2; see .ai/. *)

open Eval_common

let value ?(local = fun _ -> None) ?(local_at = fun _ _ -> None) ?scan
    ?scan_meter ?(reducer = []) ?(on_reduction = fun () -> ()) (env : Env.t)
    ~output e =
  Err.Escape.with_escape @@ fun esc ->
  let vchk r = vchk esc r in
  (* A LIST, not a single pair: a scan row's [update] has TWO simultaneously
     bound reducers ([lane] and [step]), unlike a vector local's body, which
     mentions only its own binder free -- [Region_execution]/[Region_eval]
     supply both as [[ (lane, l); (step, r) ]] when filling one trace row. *)
  let init_reducers w =
    List.find_map
      (fun (v, p) -> if Reduce_var.equal w v then Some p else None)
      reducer
  in
  (* Missing entirely -- no default trace table -- fails with the same
     [Unknown_local] a real reader would report for an unrecognized id. *)
  let scan : scan_reader =
    match scan with
    | Some reader -> reader
    | None -> fun id ~row:_ ~lane:_ -> Err.fail (Unknown_local id)
  in
  (* [eval_index] is polymorphic in the caller's error row, and [env.load_index]
     already sits at exactly this frame's own [error] row -- so [esc] is
     passed directly, with no narrowing view needed (unlike before [Data]
     existed, when the index evaluator's row was strictly narrower than this
     frame's). *)
  let idx reducers i =
    eval_index esc
      ~widen:(fun (e : index_error) -> (e :> error))
      ~output ~reducers ~resolve_data:env.Env.load_index i
  in
  (* [local_at] is a REF, not a plain closed-over value or a threaded
     argument: an inline [Scan_at]'s [update] evaluates under a temporarily
     REBOUND resolver that answers its own [prev] from the previous row's
     buffer, restored on every exit (success, error, or an [Err.Escape]
     unwind) via [Fun.protect]. Every other occurrence still falls through to
     the caller's original resolver. A ref keeps [go]'s calling convention,
     and so its stack frame, identical to before scan existed -- threading
     [local]/[local_at] as ordinary extra arguments measurably deepened
     [go]'s frame and regressed [Hard.eval_depth]'s node frontier under
     node. *)
  let local_at_ref = ref local_at in
  (* [eval_scan_at] stays an ordinary [and]-bound sibling of [go]/[guard]/
     [intrinsic], never called through a ref: js_of_ocaml's tail-call
     trampoline covers a statically-known mutually-recursive group, but a
     call through a ref cell is an "unknown function" it cannot fold into
     that analysis -- see
     https://ocsigen.org/js_of_ocaml/latest/js_of_ocaml/tailcall.html. An
     earlier attempt to route [Scan_at] through a forward-reference cell
     (to keep [eval_scan_at]'s bulkier body out of this group) made the
     regression below WORSE, not better. *)
  (* [@tailcall] below marks the genuine tail edges converted for JS stack
     safety; see .ai/. A missing tail call there is a build error (warning
     51), not a silent regression. *)
  let rec go reducers (e : Value.t) : float =
    match e with
    | Value.Binary (op, a, b) ->
        Value.apply_binary op (go reducers a) (go reducers b)
    | Value.Const x -> x
    | Value.Intrinsic i -> (intrinsic [@tailcall]) reducers i
    | Value.Local v -> (
        match local v with
        | Some x -> x
        | None -> Err.Escape.throw esc (`Unbound_local v))
    | Value.Local_at (v, i) -> (
        match !local_at_ref v (idx reducers i) with
        | Some x -> x
        | None -> Err.Escape.throw esc (`Unbound_local v))
    | Value.Local_scan_at (v, row_i, lane_i) ->
        let row = idx reducers row_i and lane = idx reducers lane_i in
        vchk (Err.map_error scan_error (scan v ~row ~lane))
    | Value.Load (s, c) -> vchk (env.Env.load s (Coord.map (idx reducers) c))
    | Value.Reduce r ->
        let lo = idx reducers r.Reduction.lo
        and hi = idx reducers r.Reduction.hi in
        let combine, init =
          match r.Reduction.kind with
          | Reduction.Max -> (Max_op.apply Max_op.Float_max, Float.neg_infinity)
          | Reduction.Sum -> (( +. ), 0.)
        in
        (* The ordered half-open left fold the denotation specifies. Same seed
             and same association as the engine's own reduction -- a rewrite that
             reassociated this would change the answer, not just its shape. *)
        let rec fold i acc =
          if i >= hi then acc
          else
            let bound v =
              if Reduce_var.equal v r.Reduction.var then Some i else reducers v
            in
            on_reduction ();
            (fold [@tailcall]) (i + 1) (combine acc (go bound r.Reduction.body))
        in
        fold lo init
    | Value.Round_f32 a ->
        (* Convert to binary32 and widen back. The one value expression that
             changes a value without being arithmetic. *)
        Int32.float_of_bits (Int32.bits_of_float (go reducers a))
    | Value.Scan_at (s, row_i, lane_i) ->
        (eval_scan_at [@tailcall]) reducers s row_i lane_i
    (* Only the SELECTED branch is evaluated -- the other may divide by zero
         or read out of bounds, and guarding is what the caller built it for. *)
    | Value.Select (c, a, b) ->
        if guard reducers c then (go [@tailcall]) reducers a
        else (go [@tailcall]) reducers b
    | Value.Unary (op, a) -> Value.apply_unary op (go reducers a)
    | Value.Value_of_index i -> vchk (float_of_index (idx reducers i))
  and guard reducers = function
    | Bool.Index_eq (a, b) -> Int.equal (idx reducers a) (idx reducers b)
    | Bool.Value_lt (a, b) -> go reducers a < go reducers b
  and intrinsic reducers (Intrinsic.Max_pool d as i) =
    let open Intrinsic.Max_pool in
    let at a = idx reducers (Coord.get d.out a) in
    let w = vchk (Intrinsic.window i ~out_h:(at Axis.H) ~out_w:(at Axis.W)) in
    let read ih iw =
      vchk
        (env.Env.load d.source
           (Coord.of_fn (fun a ->
                if a = Axis.H then ih else if a = Axis.W then iw else at a)))
    in
    (* Value and index advance TOGETHER under one predicate. Updating them
         separately is how they fell out of step originally, which is why
         [Max_op.pool_better] is shared rather than open-coded. An ordinary tie
         keeps the incumbent; a NaN re-triggers, so the LAST NaN wins. *)
    let rec rows ih best best_ix =
      if ih >= w.Intrinsic.Window.hhi then (best, best_ix)
      else (cols [@tailcall]) ih w.Intrinsic.Window.wlo best best_ix
    and cols ih iw best best_ix =
      if iw >= w.Intrinsic.Window.whi then
        (rows [@tailcall]) (ih + 1) best best_ix
      else
        let v = read ih iw in
        let best, best_ix =
          if Max_op.pool_better ~best ~value:v then
            (v, vchk (Intrinsic.flat_index i ~ih ~iw))
          else (best, best_ix)
        in
        (cols [@tailcall]) ih (iw + 1) best best_ix
    in
    let best, best_ix = rows w.Intrinsic.Window.hlo Float.neg_infinity 0 in
    match d.result with Value -> best | Index -> vchk (float_of_index best_ix)
  (* Bounds-checks row then lane (row wins on a simultaneous failure),
     reserves [2 * width] live state for the nesting peak (released on every
     exit path, including an [Err.Escape] unwind, via [Fun.protect]), then
     runs exactly [row] steps over two row buffers: [prev_row] the last
     completed row, [cur_row] the one being filled. [update] reads [prev]
     through a temporarily rebound [local_at_ref] that answers from
     [prev_row]; every other local reference in [update] still resolves
     through the caller's own [local]/[local_at], since a Region scan's
     update legitimately reads earlier Region locals. *)
  and eval_scan_at reducers s row_i lane_i =
    let row = idx reducers row_i and lane = idx reducers lane_i in
    let projection = { Scan_projection.local = None; row; lane } in
    if row < 0 || row > s.Scan.steps then
      Err.Escape.throw esc
        (`Scan_projection
           (Row_out_of_range
              { Scan_bounds.projection; extent = s.Scan.steps + 1 }))
    else if lane < 0 || lane >= s.Scan.width then
      Err.Escape.throw esc
        (`Scan_projection
           (Lane_out_of_range { Scan_bounds.projection; extent = s.Scan.width }))
    else
      let meter =
        match scan_meter with
        | Some m -> m
        | None -> Err.Escape.throw esc `Scan_meter_required
      in
      vchk
        (Err.map_error scan_meter_error
           (Scan_meter.reserve meter ~width:s.Scan.width));
      let saved_local_at = !local_at_ref in
      Fun.protect
        ~finally:(fun () ->
          local_at_ref := saved_local_at;
          Scan_meter.release meter ~width:s.Scan.width)
        (fun () ->
          let init_row () =
            Array.init s.Scan.width (fun l ->
                let bound v =
                  if Reduce_var.equal v s.Scan.lane then Some l else reducers v
                in
                go bound s.Scan.init)
          in
          let next_row ~step prev_row =
            (local_at_ref :=
               fun v pos ->
                 if Local_var.equal v s.Scan.prev then Some prev_row.(pos)
                 else saved_local_at v pos);
            Array.init s.Scan.width (fun l ->
                vchk
                  (Err.map_error scan_meter_error
                     (Scan_meter.charge_update meter));
                let bound v =
                  if Reduce_var.equal v s.Scan.lane then Some l
                  else if Reduce_var.equal v s.Scan.step then Some step
                  else reducers v
                in
                go bound s.Scan.update)
          in
          let rec run r prev_row =
            if r = row then prev_row.(lane)
            else (run [@tailcall]) (r + 1) (next_row ~step:r prev_row)
          in
          run 0 (init_row ()))
  in
  go init_reducers e
