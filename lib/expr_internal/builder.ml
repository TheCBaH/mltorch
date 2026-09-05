(* Reducer identities must be fresh during construction AND rewriting. The
     supply is threaded, not global: [run] starts a fresh identity namespace,
     and [run_from] CONTINUES an existing one, for when several computations
     have to share a namespace rather than each restarting at ordinal 0. Two
     computations run from [initial] deliberately reuse ordinals -- identity is
     meaningful only within one expression, and composing across that boundary
     is what [Rewrite.freshen] is for. *)
type state = int
type 'a t = state -> 'a * state

exception Supply_exhausted
(** Raised by [fresh_reduce] when the ordinal would wrap. A wrapped ordinal
    would re-mint a live identity and break scope, alpha-equivalence, comparison
    and hashing at once, so this is checked rather than assumed -- a
    trusted-capacity boundary, named so it is visible. *)

let initial : state = 0
let return x s = (x, s)

let bind m f s =
  let x, s = m s in
  f x s

let map f m s =
  let x, s = m s in
  (f x, s)

let run_from s m = m s
let run m = fst (m initial)
let fresh s = if s = Stdlib.max_int then raise Supply_exhausted else (s, s + 1)
let fresh_reduce = fresh
let fresh_local = fresh

module Syntax = struct
  let ( let* ) = bind
  let ( let+ ) m f = map f m
end

(* Allocates the variable, hands its index expression to the body, and threads
     the updated supply through. Going through here rather than building a
     [Reduction.t] by hand is what makes scope correct by construction: the
     callback cannot name a variable that is not this reduction's, and the
     variable cannot escape into a sibling. *)
let reduction ~kind ~lo ~hi body s =
  let v, s = fresh_reduce s in
  let b, s = body (Index.reduce v) s in
  (Value.reduce { Reduction.kind; var = v; lo; hi; body = b }, s)

(* Above the [2 * width] arithmetic below (metering and state reservation)
   can ever be trusted not to overflow: a generous sanity ceiling, comfortably
   inside every backend's 32-bit-safe half-range, far above any real scan
   width. Rejected as [Bad_width], the same as [width <= 0]. *)
let scan_width_ceiling = 0x3FFF_FFFF

(* Mints [lane]/[step] (reducers, via the shared reduce supply) and [prev]
   (the first local BINDER this language has), hands them to [init]/[update]
   exactly as [reduction] hands its own binder to its body, then validates
   the fully-built descriptor: [steps]/[width] must be sane, [step]/[prev]
   (freshly minted, so never legitimately free elsewhere) must not appear
   free in [init], and the descriptor's WORST CASE -- [2 * width] live state,
   [steps * width] updates -- must clear [limits]. Runtime metering
   ([Scan_meter]) is still required after this: a checked descriptor can be
   composed under another reduction or passed to a wider evaluator, contexts
   this construction-time check cannot see. *)
let scan ~limits ~width ~steps ~init ~update s =
  let lane, s = fresh_reduce s in
  let step, s = fresh_reduce s in
  let prev, s = fresh_local s in
  let init_body, s = init ~lane:(Index.reduce lane) s in
  let update_body, s =
    update ~step:(Index.reduce step) ~lane:(Index.reduce lane)
      ~previous_at:(fun i -> Value.local_at prev i)
      s
  in
  let result =
    if steps < 0 then Err.fail (Scan.Bad_steps steps)
    else if width <= 0 || width > scan_width_ceiling then
      Err.fail (Scan.Bad_width width)
    else if Local_var.Set.mem prev (Fold.locals init_body) then
      Err.fail Scan.Prev_in_init
    else if Reduce_var.Set.mem step (Fold.free_reducers init_body) then
      Err.fail Scan.Step_in_init
    else if 2 * width > Scan_limits.max_state limits then
      Err.fail (Scan.State_over_limit { limit = Scan_limits.max_state limits })
    else
      let worst_updates = Int64.mul (Int64.of_int steps) (Int64.of_int width) in
      if Int64.compare worst_updates (Scan_limits.max_updates limits) > 0 then
        Err.fail
          (Scan.Updates_over_limit { limit = Scan_limits.max_updates limits })
      else
        Err.return
          {
            Scan.width;
            steps;
            lane;
            step;
            prev;
            init = init_body;
            update = update_body;
          }
  in
  (result, s)
