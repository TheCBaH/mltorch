(* Deliberately narrow. Division parameters, load roles and intrinsic
     dimensions are NOT rechecked here: the smart constructors and the [Index]
     GADT make each of those unconstructable through the public API, so a rule
     for them could never be turned red, and CLAUDE.md is explicit that a check
     which has never failed is not evidence. What remains is exactly what
     COMPOSITION can still violate. *)
type error =
  [ `Duplicate_binder of Reduce_var.t
  | `Free_reducer of Reduce_var.t
  | `Too_deep of int
  | `Too_large of int
  | `Unbound_local of Local_var.t ]

(* The payload is the LIMIT, not the measure. Reporting the actual size would
     mean measuring the whole tree, which is the thing the limit exists to
     avoid. *)
let pp_error fmt : [< error ] -> unit = function
  | `Duplicate_binder v ->
      Fmt.pf fmt "reducer %a is bound twice on one path" Reduce_var.pp v
  | `Free_reducer v -> Fmt.pf fmt "free reducer %a" Reduce_var.pp v
  | `Too_deep limit -> Fmt.pf fmt "depth exceeds limit %d" limit
  | `Too_large limit -> Fmt.pf fmt "size exceeds limit %d" limit
  | `Unbound_local v -> Fmt.pf fmt "unbound local %a" Local_var.pp v

(* A variable bound again inside its own scope. Two independently built
     fragments composed without freshening is how this arises in practice, and
     it is a real defect rather than shadowing: the inner binder captures
     references meant for the outer one, so evaluation silently changes. *)
let duplicate_binder e =
  let rec go bound (e : Value.t) =
    match e with
    | Value.Const _ | Value.Value_of_index _ | Value.Load _ | Value.Intrinsic _
    | Value.Local _ ->
        None
    | Value.Binary (_, a, b) -> (
        match go bound a with None -> go bound b | some -> some)
    | Value.Unary (_, a) | Value.Round_f32 a -> go bound a
    | Value.Select (c, a, b) -> (
        let guard =
          match c with
          | Bool.Value_lt (x, y) -> (
              match go bound x with None -> go bound y | some -> some)
          | Bool.Index_eq _ -> None
        in
        match guard with
        | Some _ -> guard
        | None -> ( match go bound a with None -> go bound b | some -> some))
    | Value.Reduce r ->
        if Reduce_var.Set.mem r.Reduction.var bound then Some r.Reduction.var
        else go (Reduce_var.Set.add r.Reduction.var bound) r.Reduction.body
  in
  go Reduce_var.Set.empty e

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
let fragment ?max_size ?max_depth ~locals e =
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
  let* () =
    match Reduce_var.Set.min_elt_opt (Fold.free_reducers e) with
    | Some v -> Err.fail (`Free_reducer v)
    | None -> Err.return ()
  in
  match duplicate_binder e with
  | Some v -> Err.fail (`Duplicate_binder v)
  | None -> Err.return ()

let value ?max_size ?max_depth e =
  fragment ?max_size ?max_depth ~locals:Local_var.Set.empty e
