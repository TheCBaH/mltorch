module Field = struct
  type t = Max_state | Max_updates
end

module Invalid = struct
  type t = { field : Field.t; value : int64 }
end

type error = Invalid of Invalid.t
type t = { max_state : int; max_updates : int64 }

(* Unlike [Kernel.Limits.create], zero is valid for both fields: [max_state =
   0] forbids every scan (each reserves [2 * width >= 2] live state);
   [max_updates = 0] admits only descriptors that can never update ([steps =
   0], whose trace is just the initial row). Only the upper, exclusive
   [v >= hard] rule is shared with [Kernel.Limits.create]. These hard ceilings
   are policy choices, not empirically discovered stack-overflow frontiers
   like [Kernel.Limits.Hard.depth]/[eval_recursion] -- see the scan design
   record's array-capacity probe. *)
let hard_max_state = 1_048_576
let hard_max_updates = 1_048_576L

let create ~max_state ~max_updates =
  let open Err.Syntax in
  let* () =
    if max_state < 0 || max_state >= hard_max_state then
      Err.fail
        (Invalid
           { Invalid.field = Field.Max_state; value = Int64.of_int max_state })
    else Err.return ()
  in
  let+ () =
    if
      Int64.compare max_updates 0L < 0
      || Int64.compare max_updates hard_max_updates >= 0
    then
      Err.fail
        (Invalid { Invalid.field = Field.Max_updates; value = max_updates })
    else Err.return ()
  in
  { max_state; max_updates }

let max_state t = t.max_state
let max_updates t = t.max_updates

let pp_error fmt = function
  | Invalid { Invalid.field; value } ->
      Fmt.pf fmt "invalid %s = %Ld"
        (match field with
        | Field.Max_state -> "max_state"
        | Field.Max_updates -> "max_updates")
        value

(* Default headroom over the corpus-derived per-key census (6,144 updates,
   ~6,528 resident slots for the dominant LSTM shape) -- see the scan design
   record's "Chosen defaults and hard ceilings". *)
let default = Err.or_raise ~pp_error (create ~max_state:8192 ~max_updates:8192L)
