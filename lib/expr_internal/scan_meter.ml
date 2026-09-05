(* [live_state]/[updates_remaining] are internal mutable counters, never
   exposed: a caller that could set or replace them directly could under- or
   over-count relative to what actually ran. The only public mutator is
   [charge_update]; [reserve]/[release] (state, not exported through
   [Expr_api]) are for [Eval]'s own inline [Scan_at] evaluation, which is the
   only caller that allocates the two-row buffer this meter's state budget
   describes. *)
type t = {
  limits : Scan_limits.t;
  mutable live_state : int;
  mutable updates_remaining : int64;
}

type error =
  | State_over_limit of { limit : int }
  | Updates_exhausted of { limit : int64 }

let pp_error fmt = function
  | State_over_limit { limit } -> Fmt.pf fmt "scan state exceeds limit %d" limit
  | Updates_exhausted { limit } ->
      Fmt.pf fmt "scan updates exhausted at limit %Ld" limit

let create ~limits =
  { limits; live_state = 0; updates_remaining = Scan_limits.max_updates limits }

(* Exactly [limit] calls succeed; the next fails BEFORE its update body is
   evaluated -- the standalone inline evaluator and both Region executors
   call this same function, so their off-by-one behavior at the boundary
   cannot drift apart. *)
let charge_update t =
  if Int64.compare t.updates_remaining 0L <= 0 then
    Err.fail (Updates_exhausted { limit = Scan_limits.max_updates t.limits })
  else (
    t.updates_remaining <- Int64.sub t.updates_remaining 1L;
    Err.return ())

let remaining t = t.updates_remaining

(* Reserves against the NESTING PEAK, not a running total: catches a
   row-zero projection (no updates, but state is still live) and a scan
   nested inside another scan's initializer. [width] is already bounded well
   below any overflow risk by [Scan.create]'s [Bad_width] check, so [2 *
   width] cannot wrap. *)
let reserve t ~width =
  let need = 2 * width in
  if t.live_state + need > Scan_limits.max_state t.limits then
    Err.fail (State_over_limit { limit = Scan_limits.max_state t.limits })
  else (
    t.live_state <- t.live_state + need;
    Err.return ())

(* Restores the prior level on every exit -- success, a typed error, and an
   [Err.Escape] unwind alike. The caller is responsible for calling this on
   every path; [Eval.value]'s [Scan_at] arm is the only caller and does so
   with [Fun.protect]. *)
let release t ~width = t.live_state <- t.live_state - (2 * width)
