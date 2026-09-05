(* Deliberately narrow. Division parameters, load roles and intrinsic
     dimensions are NOT rechecked here: the smart constructors and the [Index]
     GADT make each of those unconstructable through the public API, so a rule
     for them could never be turned red, and CLAUDE.md is explicit that a check
     which has never failed is not evidence. What remains is exactly what
     COMPOSITION can still violate. *)
type error =
  [ `Duplicate_local_binder of Local_var.t
  | `Duplicate_reducer_binder of Reduce_var.t
  | `Free_reducer of Reduce_var.t
  | `Too_deep of int
  | `Too_large of int
  | `Unbound_local of Local_var.t ]

(* The payload is the LIMIT, not the measure. Reporting the actual size would
     mean measuring the whole tree, which is the thing the limit exists to
     avoid. *)
let pp_error fmt : [< error ] -> unit = function
  | `Duplicate_local_binder v ->
      Fmt.pf fmt "local %a is bound twice on one path" Local_var.pp v
  | `Duplicate_reducer_binder v ->
      Fmt.pf fmt "reducer %a is bound twice on one path" Reduce_var.pp v
  | `Free_reducer v -> Fmt.pf fmt "free reducer %a" Reduce_var.pp v
  | `Too_deep limit -> Fmt.pf fmt "depth exceeds limit %d" limit
  | `Too_large limit -> Fmt.pf fmt "size exceeds limit %d" limit
  | `Unbound_local v -> Fmt.pf fmt "unbound local %a" Local_var.pp v

type duplicate = Local of Local_var.t | Reducer of Reduce_var.t

(* A variable bound again inside its own scope. Two independently built
     fragments composed without freshening is how this arises in practice, and
     it is a real defect rather than shadowing: the inner binder captures
     references meant for the outer one, so evaluation silently changes.
     [prev] makes [Local_var.t] a binder for the first time, hence the two
     namespaces ([bound]/[lbound]) mirroring [Fold.free_reducers]/
     [Fold.locals]'s own scope masking. *)
let duplicate_binder e =
  let rec go bound lbound (e : Value.t) =
    match e with
    | Value.Const _ | Value.Value_of_index _ | Value.Load _ | Value.Intrinsic _
    | Value.Local _ | Value.Local_at _ | Value.Local_scan_at _ ->
        None
    | Value.Binary (_, a, b) -> (
        match go bound lbound a with None -> go bound lbound b | some -> some)
    | Value.Unary (_, a) | Value.Round_f32 a -> go bound lbound a
    | Value.Select (c, a, b) -> (
        let guard =
          match c with
          | Bool.Value_lt (x, y) -> (
              match go bound lbound x with
              | None -> go bound lbound y
              | some -> some)
          | Bool.Index_eq _ -> None
        in
        match guard with
        | Some _ -> guard
        | None -> (
            match go bound lbound a with
            | None -> go bound lbound b
            | some -> some))
    | Value.Reduce r ->
        if Reduce_var.Set.mem r.Reduction.var bound then
          Some (Reducer r.Reduction.var)
        else
          go (Reduce_var.Set.add r.Reduction.var bound) lbound r.Reduction.body
    | Value.Scan_at (s, _, _) -> (
        if Reduce_var.Set.mem s.Scan.lane bound then Some (Reducer s.Scan.lane)
        else
          match
            go (Reduce_var.Set.add s.Scan.lane bound) lbound s.Scan.init
          with
          | Some _ as d -> d
          | None ->
              if Reduce_var.Set.mem s.Scan.step bound then
                Some (Reducer s.Scan.step)
              else if Local_var.Set.mem s.Scan.prev lbound then
                Some (Local s.Scan.prev)
              else
                go
                  (Reduce_var.Set.add s.Scan.lane
                     (Reduce_var.Set.add s.Scan.step bound))
                  (Local_var.Set.add s.Scan.prev lbound)
                  s.Scan.update)
  in
  go Reduce_var.Set.empty Local_var.Set.empty e

(* The limits come FIRST, and are metered by one traversal carrying both. Both
     scope traversals recurse over the whole tree, so running them ahead of the
     limits would exhaust the stack on precisely the oversized input the limits
     exist to reject -- a resource guard that only works on trees that did not
     need one. So an expression past a configured limit is reported as
     [`Too_large]/[`Too_deep] even when it is also ill-scoped.

     An absent limit becomes [max_int], which cannot trip: the two budgets have
     to ride the same walk (see [Fold.measure]), so leaving one out must mean an
     unreachable bound rather than a separate pass. With both absent there is
     nothing to bound, and the walk is skipped. *)
let fragment ?max_size ?max_depth ?(allowed_free = Reduce_var.Set.empty) ~locals
    e =
  let open Err.Syntax in
  let or_unbounded = function Some l -> l | None -> Stdlib.max_int in
  let* () =
    match (max_size, max_depth) with
    | None, None -> Err.return ()
    | _ -> (
        let max_size = or_unbounded max_size
        and max_depth = or_unbounded max_depth in
        match Fold.exceeds ~max_size ~max_depth e with
        | Some `Size -> Err.fail (`Too_large max_size)
        | Some `Depth -> Err.fail (`Too_deep max_depth)
        | None -> Err.return ())
  in
  let* () =
    match
      Local_var.Set.min_elt_opt (Local_var.Set.diff (Fold.locals e) locals)
    with
    | Some v -> Err.fail (`Unbound_local v)
    | None -> Err.return ()
  in
  (* [allowed_free] is for exactly one caller's use: a vector local's OWN
     binder is free by construction within its stored value (that is what
     lets its body "mention the binder"), so the scope check must not treat
     that specific, expected freedom as the composition defect this check
     otherwise exists to catch. Anything else in the diff is still a real
     violation. *)
  let* () =
    match
      Reduce_var.Set.min_elt_opt
        (Reduce_var.Set.diff (Fold.free_reducers e) allowed_free)
    with
    | Some v -> Err.fail (`Free_reducer v)
    | None -> Err.return ()
  in
  match duplicate_binder e with
  | Some (Local v) -> Err.fail (`Duplicate_local_binder v)
  | Some (Reducer v) -> Err.fail (`Duplicate_reducer_binder v)
  | None -> Err.return ()

let value ?max_size ?max_depth e =
  fragment ?max_size ?max_depth ~locals:Local_var.Set.empty e
