(* Resolves an index tree to a compile-time constant, or [None] if it depends
   on anything that varies per evaluation ([Output], [Data], a [Reduce]
   reference). Used only to decide whether an ENCLOSING reduction's extent is
   statically known -- not a general constant-folder, and deliberately not
   shared with one: this is the one place a partial, best-effort answer
   (rather than a full evaluation) is the correct behavior, since "not
   provably constant" and "extremely expensive to prove constant" are both
   answered as [None]. *)
let rec index_const : type r. r Index.t -> int option = function
  | Index.Add (a, b) ->
      Option.bind (index_const a) (fun x ->
          Option.map (( + ) x) (index_const b))
  | Index.Assume_position a -> index_const a
  | Index.Ceil_div_pos (a, d) ->
      Option.map (fun x -> (x + d - 1) / d) (index_const a)
  | Index.Clamp_low a -> Option.map (Stdlib.max 0) (index_const a)
  | Index.Const n -> Some n
  | Index.Data _ -> None
  | Index.Floor_div_pos (a, d) -> Option.map (fun x -> x / d) (index_const a)
  | Index.Max (a, b) ->
      Option.bind (index_const a) (fun x ->
          Option.map (Stdlib.max x) (index_const b))
  | Index.Min (a, b) ->
      Option.bind (index_const a) (fun x ->
          Option.map (Stdlib.min x) (index_const b))
  | Index.Of_position a -> index_const a
  | Index.Output _ -> None
  | Index.Reduce _ -> None
  | Index.Scale (k, a) -> Option.map (( * ) k) (index_const a)
  | Index.Zero -> Some 0

(* [None] once anywhere beneath a statically unbounded reduction; [Some m]
   beneath a chain of statically-constant reductions whose combined extent
   product is [m], starting at [Some 1] at top level. *)
let static_extent lo hi =
  match (index_const lo, index_const hi) with
  | Some l, Some h -> Some (Stdlib.max 0 (h - l))
  | _ -> None

let combine multiplier extent =
  match (multiplier, extent) with Some m, Some e -> Some (m * e) | _ -> None

(* Over a whole raw [Value.t] -- a checked scan can still be composed under
   another reduction, inserted by a raw rewrite, or passed to the evaluator
   directly, none of which [Builder.scan]'s own construction-time check can
   see. A scan beneath a statically unbounded reduction is rejected outright;
   one beneath a chain of constant-extent reductions has its worst-case
   update count multiplied by their combined extent and re-checked against
   [limits], since LSTM's own scans sit at Region-local top level and this
   costs that target nothing. *)
let check ~limits value =
  let open Err.Syntax in
  let rec go multiplier (e : Value.t) =
    match e with
    | Value.Const _ | Value.Local _ | Value.Local_at _ | Value.Local_scan_at _
    | Value.Load _ | Value.Value_of_index _ | Value.Intrinsic _ ->
        Err.return ()
    | Value.Binary (_, a, b) ->
        let* () = go multiplier a in
        go multiplier b
    | Value.Unary (_, a) | Value.Round_f32 a -> go multiplier a
    | Value.Select (c, a, b) ->
        let* () =
          match c with
          | Bool.Value_lt (x, y) ->
              let* () = go multiplier x in
              go multiplier y
          | Bool.Index_eq _ -> Err.return ()
        in
        let* () = go multiplier a in
        go multiplier b
    | Value.Reduce r ->
        let extent = static_extent r.Reduction.lo r.Reduction.hi in
        go (combine multiplier extent) r.Reduction.body
    | Value.Scan_at (s, _, _) ->
        let* () =
          match multiplier with
          | None -> Err.fail Scan.Unbounded_reduction_context
          | Some m ->
              let worst =
                Int64.mul (Int64.of_int m)
                  (Int64.mul
                     (Int64.of_int s.Scan.steps)
                     (Int64.of_int s.Scan.width))
              in
              if Int64.compare worst (Scan_limits.max_updates limits) > 0 then
                Err.fail
                  (Scan.Updates_over_limit
                     { limit = Scan_limits.max_updates limits })
              else Err.return ()
        in
        let* () = go multiplier s.Scan.init in
        go multiplier s.Scan.update
  in
  go (Some 1) value
