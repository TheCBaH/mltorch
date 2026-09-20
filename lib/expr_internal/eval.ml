(* The reference evaluator. Its shared plumbing (index evaluation, the
   [error]/[Env] types, [vchk]) moved verbatim to [eval_common.ml] in the
   tail-call conversion's Stage 2; see .ai/. *)

open Eval_common

(* Tail-call conversion Stage 3: [value] is duplicated on both sides of this
   conditional -- no longer verbatim, since Stage 6 installed the selected
   stack-safe driver (see .ai/'s design record) on the JS side, but the two
   branches still start from the same denotation and this file remains the
   one place both live, so a reader can compare them directly. cppo, not an
   ordinary [if], because the two branches must be reachable as separate
   compiled artifacts from separate library stanzas (native's [expr_internal]
   takes no [-D]; [expr_internal_js]/[expr_internal_mel] define
   [JS_BACKEND]): a runtime branch would still force one shared preprocessing
   pass, and the JS side needs different cppo/dune wiring than native keeps.
   See .ai/. *)
#if defined JS_BACKEND

(* [Max_pool]'s row/column sweep used to be two mutually tail-recursive
   functions ([rows]/[cols]). js_of_ocaml already trampolines that shape
   safely, but Melange raises at depth on a mutual tail pair -- see .ai/.
   Collapsing both into one self-recursive [loop] over a nullary state tag
   keeps the row/column transition in tail position on every JS backend, at
   the cost of one extra branch jsoo alone did not need. *)
type pool_tag = Cols_tag | Rows_tag

(* Melange has no [caml_restore_raw_backtrace]: calling
   [Printexc.raise_with_backtrace] there throws "not polyfilled by Melange
   yet" at RUNTIME, not build time. Found by the tail-call conversion's
   Stage 5 cleanup-protocol negative controls; see .ai/. *)
#if defined MELANGE_BACKEND

type captured_backtrace = unit

let capture_backtrace () : captured_backtrace = ()
let reraise exn (_ : captured_backtrace) = raise exn

#else

type captured_backtrace = Printexc.raw_backtrace

let capture_backtrace () : captured_backtrace = Printexc.get_raw_backtrace ()
let reraise exn (bt : captured_backtrace) = Printexc.raise_with_backtrace exn bt

#endif

(* [cutoff]: below this recursion depth, [go]/[guard] recurse directly, like
   native's own [go]/[guard] below; at [cutoff], they hand the remaining
   subtree to [Eval_js_machine.run], the stack-safe array/mutation-based
   frame machine selected by the tail-call conversion's Stage 5 benchmarking
   (see .ai/'s design record for the full evaluation across all four
   candidates and both backends). 50 is not a measured maximum-safe
   recursion depth -- that per-backend frontier still needs Stage 7's
   stack-fault tooling, not yet built -- it is the SAME value Stage 5 M9-M12
   measured this composition at: proven safe to a 20,000-deep smoke case on
   both backends (js/probe's own [eval_hybrid ~cutoff:50] candidate, run
   through [make expr_bench.runtest]/[.js-benchmark]) and the config that won
   on nearly every shallow, production-shaped corpus case on both backends,
   trading only a modest, secondary above-frontier deep-depth penalty
   relative to the pure-machine candidate. *)
let cutoff = 50

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
     buffer. A ref keeps [go]'s calling convention, and so its stack frame,
     identical to before scan existed -- threading [local]/[local_at] as
     ordinary extra arguments measurably deepened [go]'s frame and regressed
     [Hard.eval_depth]'s node frontier under node. *)
  let local_at_ref = ref local_at in
  (* Cleanup-protocol state (tail-call conversion Stage 5; see .ai/'s design
     record), replacing native's [Fun.protect]: a trampoline bounce or a
     machine handoff is a host RETURN, not a stack unwind, so [Fun.protect]'s
     [~finally] would fire immediately on a bounce rather than staying
     pending across it. Every [Scan_at] reservation pushes its own release
     closure here instead; the top-level [try]/[with] below sweeps every
     pending closure, LIFO, on ANY exception (including a machine handoff's),
     and [run_top_cleanup] runs the OLDEST live reservation's release the
     instant its own scan completes normally. *)
  let cleanups : (unit -> unit) list ref = ref [] in
  let run_top_cleanup () =
    match !cleanups with
    | f :: rest ->
        cleanups := rest;
        f ()
    | [] -> assert false
  in
  let machine_run seed =
    Eval_js_machine.run ~esc ~env ~output ~scan ~scan_meter ~local
      ~local_at_ref ~cleanups ~run_top_cleanup ~on_reduction seed
  in
  (* [@tailcall] below marks the genuine tail edges converted for JS stack
     safety; see .ai/. A missing tail call there is a build error (warning
     51), not a silent regression. Below [cutoff], [go]/[guard] recurse
     exactly like native's own below -- real, unbounded-looking OCaml
     recursion, deliberately, since every real backend trampolines a bounded
     amount of it safely regardless of provable tail position (see .ai/'s
     design record). At [cutoff], they hand off to [machine_run] instead,
     sharing THIS call's own [esc]/[cleanups]/[local_at_ref] rather than
     starting fresh ones. Once handed off, a subtree stays in the machine for
     the rest of its own evaluation -- [Eval_js_machine.run] is a complete
     evaluator of the full grammar, so there is never a reason to hand a
     partially-machine-evaluated subtree back to direct recursion. *)
  let rec go depth reducers (e : Value.t) : float =
    if depth >= cutoff then
      match machine_run (Eval_js_machine.Eval_state (e, reducers)) with
      | Eval_js_machine.Float_result v -> v
      | _ -> assert false
    else
      let depth = depth + 1 in
      match e with
      | Value.Binary (op, a, b) ->
          Value.apply_binary op (go depth reducers a) (go depth reducers b)
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
          (match r.Reduction.kind with
          | Reduction.Max | Reduction.Sum ->
              let combine, init =
                match r.Reduction.kind with
                | Reduction.Max ->
                    (Max_op.apply Max_op.Float_max, Float.neg_infinity)
                | Reduction.Sum -> (( +. ), 0.)
                | Reduction.Argmax_index | Reduction.Argmax_value ->
                    assert false
              in
              (* The ordered half-open left fold the denotation specifies. Same
                   seed and same association as the engine's own reduction --
                   a rewrite that reassociated this would change the answer,
                   not just its shape. *)
              let rec fold i acc =
                if i >= hi then acc
                else
                  let bound v =
                    if Reduce_var.equal v r.Reduction.var then Some i
                    else reducers v
                  in
                  on_reduction ();
                  (fold [@tailcall]) (i + 1)
                    (combine acc (go depth bound r.Reduction.body))
              in
              fold lo init
          | Reduction.Argmax_index | Reduction.Argmax_value ->
              (* One predicate advances value and index together
                 ([Max_op.pool_better], the same convention
                 [Intrinsic.Max_pool]'s own paired value/index output uses),
                 so the two outputs cannot fall out of step: an ordinary tie
                 keeps the incumbent (first index wins) and a NaN retriggers. *)
              let rec fold i best best_i =
                if i >= hi then (best, best_i)
                else
                  let bound v =
                    if Reduce_var.equal v r.Reduction.var then Some i
                    else reducers v
                  in
                  let value = go depth bound r.Reduction.body in
                  on_reduction ();
                  let best, best_i =
                    if Max_op.pool_better ~best ~value then (value, i)
                    else (best, best_i)
                  in
                  (fold [@tailcall]) (i + 1) best best_i
              in
              let best, best_i = fold lo Float.neg_infinity lo in
              (match r.Reduction.kind with
              | Reduction.Argmax_value -> best
              | Reduction.Argmax_index -> vchk (float_of_index best_i)
              | Reduction.Max | Reduction.Sum -> assert false))
      | Value.Round_f32 a ->
          (* Convert to binary32 and widen back. The one value expression
               that changes a value without being arithmetic. *)
          Int32.float_of_bits (Int32.bits_of_float (go depth reducers a))
      | Value.Scan_at (s, row_i, lane_i) ->
          (eval_scan_at [@tailcall]) depth reducers s row_i lane_i
      (* Only the SELECTED branch is evaluated -- the other may divide by
           zero or read out of bounds, and guarding is what the caller built
           it for. *)
      | Value.Select (c, a, b) ->
          if guard depth reducers c then (go [@tailcall]) depth reducers a
          else (go [@tailcall]) depth reducers b
      | Value.Unary (op, a) -> Value.apply_unary op (go depth reducers a)
      | Value.Value_of_index i -> vchk (float_of_index (idx reducers i))
  and guard depth reducers (b : Bool.t) : bool =
    if depth >= cutoff then
      match machine_run (Eval_js_machine.Guard_state (b, reducers)) with
      | Eval_js_machine.Bool_result r -> r
      | _ -> assert false
    else
      let depth = depth + 1 in
      match b with
      | Bool.Index_eq (a, b) -> Int.equal (idx reducers a) (idx reducers b)
      | Bool.Value_lt (a, b) -> go depth reducers a < go depth reducers b
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
         keeps the incumbent; a NaN re-triggers, so the LAST NaN wins. [rows]/
         [cols] are one self-recursive [loop] over [pool_tag], not a mutual
         pair -- see .ai/. *)
    let rec loop tag ih iw best best_ix =
      match tag with
      | Rows_tag ->
          if ih >= w.Intrinsic.Window.hhi then (best, best_ix)
          else
            (loop [@tailcall]) Cols_tag ih w.Intrinsic.Window.wlo best best_ix
      | Cols_tag ->
          if iw >= w.Intrinsic.Window.whi then
            (loop [@tailcall]) Rows_tag (ih + 1) iw best best_ix
          else
            let v = read ih iw in
            let best, best_ix =
              if Max_op.pool_better ~best ~value:v then
                (v, vchk (Intrinsic.flat_index i ~ih ~iw))
              else (best, best_ix)
            in
            (loop [@tailcall]) Cols_tag ih (iw + 1) best best_ix
    in
    let best, best_ix =
      loop Rows_tag w.Intrinsic.Window.hlo 0 Float.neg_infinity 0
    in
    match d.result with Value -> best | Index -> vchk (float_of_index best_ix)
  (* Bounds-checks row then lane (row wins on a simultaneous failure),
     reserves [2 * width] live state for the nesting peak (released via the
     cleanup-protocol list above, not [Fun.protect] -- see this function's
     own doc comment), then runs exactly [row] steps over two row buffers:
     [prev_row] the last completed row, [cur_row] the one being filled.
     [update] reads [prev] through a temporarily rebound [local_at_ref] that
     answers from [prev_row]; every other local reference in [update] still
     resolves through the caller's own [local]/[local_at], since a Region
     scan's update legitimately reads earlier Region locals. *)
  and eval_scan_at depth reducers s row_i lane_i =
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
      cleanups :=
        (fun () ->
          local_at_ref := saved_local_at;
          Scan_meter.release meter ~width:s.Scan.width)
        :: !cleanups;
      let init_row () =
        Array.init s.Scan.width (fun l ->
            let bound v =
              if Reduce_var.equal v s.Scan.lane then Some l else reducers v
            in
            go depth bound s.Scan.init)
      in
      let next_row ~step prev_row =
        (local_at_ref :=
           fun v pos ->
             if Local_var.equal v s.Scan.prev then Some prev_row.(pos)
             else saved_local_at v pos);
        Array.init s.Scan.width (fun l ->
            vchk
              (Err.map_error scan_meter_error (Scan_meter.charge_update meter));
            let bound v =
              if Reduce_var.equal v s.Scan.lane then Some l
              else if Reduce_var.equal v s.Scan.step then Some step
              else reducers v
            in
            go depth bound s.Scan.update)
      in
      let rec run r prev_row =
        if r = row then prev_row.(lane)
        else (run [@tailcall]) (r + 1) (next_row ~step:r prev_row)
      in
      let result = run 0 (init_row ()) in
      run_top_cleanup ();
      result
  in
  try go 0 init_reducers e
  with exn ->
    let bt = capture_backtrace () in
    List.iter (fun f -> f ()) !cleanups;
    cleanups := [];
    reraise exn bt

#else

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
        (match r.Reduction.kind with
        | Reduction.Max | Reduction.Sum ->
            let combine, init =
              match r.Reduction.kind with
              | Reduction.Max ->
                  (Max_op.apply Max_op.Float_max, Float.neg_infinity)
              | Reduction.Sum -> (( +. ), 0.)
              | Reduction.Argmax_index | Reduction.Argmax_value -> assert false
            in
            (* The ordered half-open left fold the denotation specifies. Same
                 seed and same association as the engine's own reduction -- a
                 rewrite that reassociated this would change the answer, not
                 just its shape. *)
            let rec fold i acc =
              if i >= hi then acc
              else
                let bound v =
                  if Reduce_var.equal v r.Reduction.var then Some i
                  else reducers v
                in
                on_reduction ();
                (fold [@tailcall]) (i + 1)
                  (combine acc (go bound r.Reduction.body))
            in
            fold lo init
        | Reduction.Argmax_index | Reduction.Argmax_value ->
            (* One predicate advances value and index together
               ([Max_op.pool_better], the same convention [Intrinsic.Max_pool]'s
               own paired value/index output uses), so the two outputs cannot
               fall out of step: an ordinary tie keeps the incumbent (first
               index wins) and a NaN retriggers. *)
            let rec fold i best best_i =
              if i >= hi then (best, best_i)
              else
                let bound v =
                  if Reduce_var.equal v r.Reduction.var then Some i
                  else reducers v
                in
                let value = go bound r.Reduction.body in
                on_reduction ();
                let best, best_i =
                  if Max_op.pool_better ~best ~value then (value, i)
                  else (best, best_i)
                in
                (fold [@tailcall]) (i + 1) best best_i
            in
            let best, best_i = fold lo Float.neg_infinity lo in
            (match r.Reduction.kind with
            | Reduction.Argmax_value -> best
            | Reduction.Argmax_index -> vchk (float_of_index best_i)
            | Reduction.Max | Reduction.Sum -> assert false))
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

#endif
