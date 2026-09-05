type t = Expr_repr.scan = {
  width : int;
  steps : int;
  lane : Reduce_var.t;
  step : Reduce_var.t;
  prev : Local_var.t;
  init : Expr_repr.value;
  update : Expr_repr.value;
}

(* [Bad_steps]/[Bad_width] carry the offending value (structural defects, not
   early-stopping budgets). [State_over_limit]/[Updates_over_limit] carry the
   LIMIT, per the "payload is the limit, not the measure" convention -- both
   are early-stopping budget checks against the descriptor's worst case.
   [Unbounded_reduction_context] is raised by [Scan_admission.check], not by
   construction: a scan itself does not know what encloses it. *)
type error =
  | Bad_steps of int
  | Bad_width of int
  | Prev_in_init
  | State_over_limit of { limit : int }
  | Step_in_init
  | Unbounded_reduction_context
  | Updates_over_limit of { limit : int64 }

let pp_error fmt = function
  | Bad_steps n -> Fmt.pf fmt "invalid scan step count %d" n
  | Bad_width n -> Fmt.pf fmt "invalid scan width %d" n
  | Prev_in_init ->
      Fmt.string fmt "the previous-row reader is free in the initializer"
  | State_over_limit { limit } -> Fmt.pf fmt "scan state exceeds limit %d" limit
  | Step_in_init -> Fmt.string fmt "the step index is free in the initializer"
  | Unbounded_reduction_context ->
      Fmt.string fmt "scan nested under a statically unbounded reduction"
  | Updates_over_limit { limit } ->
      Fmt.pf fmt "scan updates exceed limit %Ld" limit
