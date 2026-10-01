(* The failure record a Wasm kernel fills: [struct model_error] of the C
   backend, byte for byte, so one decoder serves both. [kind] is the position
   of the failure's kind in [Loop_js_failure.Kind.all] (closed and
   alphabetical, so the number is a stable ABI); [v] holds the kind's fields in
   [Loop_js_failure.fields] order, a coordinate taking six slots. [invocation]
   is filled by the schedule, never by a kernel. *)
let error_words = 12
let kind_offset = 0
let invocation_offset = 4
let slot_offset k = 8 + (8 * k)
let record_bytes = slot_offset error_words

let kind_index k =
  let rec go i = function
    | [] -> invalid_arg "Loop_wasm_failure.kind_index"
    | k' :: rest -> if k' = k then i else go (i + 1) rest
  in
  go 0 Loop_js_failure.Kind.all
