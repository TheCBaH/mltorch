(* Melange route's recognizer for probe_expr.ml's [--deep] deep-index
   negative control (tail-call plan, Stage 7; see .ai/). Measured directly
   (a scratch probe, reverted, recorded in .ai/expr_tailcall_design.md): a
   200,000-deep [Index.add] chain overflows [eval_index]'s ordinary
   recursion as a genuine JS [RangeError] ("Maximum call stack size
   exceeded"), which Melange's runtime wraps and re-raises as an OCaml
   [Js.Exn.Error] -- still in-process catchable, just a different exception
   shape than native/jsoo's own [Stack_overflow]. *)

let run_in_process = true

(* [msg] is a short, single-line JS error message -- this loop's own depth
   is bounded by its length, unrelated to the stack exhaustion it detects. *)
let contains ~needle msg =
  let n = String.length needle and m = String.length msg in
  let rec at i = i + n <= m && (String.sub msg i n = needle || at (i + 1)) in
  at 0

let is_exhausted = function
  | Js.Exn.Error e -> (
      match Js.Exn.message e with
      | Some msg -> contains ~needle:"call stack" msg
      | None -> false)
  | _ -> false
