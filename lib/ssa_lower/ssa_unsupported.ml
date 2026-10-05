(* A construct lowering does not handle yet, named, so a refusal is a fact a
   test can assert and a sweep can tally. Anything outside the supported slice is
   refused with one of these and never half-lowered: a partially lowered program
   is a defect however plausible its output. A refusal is a verdict about the
   request, distinct from every runtime failure of a program that was lowered. *)
type construct =
  | Argmax_reduction
  | Filled_input
  | Gather_index
  | Index_literal
  | Index_operation
  | Int64_value
  | Intrinsic
  | Load_format of string
  | Local_read
  | Max_reduction
  | Region_program
  | Scan_read
  | Select
  | Unary_operation
  | Unmaterialized_source
  | Virtual_use

type t = { at : Tensor_id.t; construct : construct }

let construct_name = function
  | Argmax_reduction -> "argmax reduction"
  | Filled_input -> "filled input"
  | Gather_index -> "gather index"
  | Index_literal -> "index literal outside the index domain"
  | Index_operation -> "min, max, clamp or division on an index"
  | Int64_value -> "int64 value"
  | Intrinsic -> "intrinsic"
  | Load_format f -> "load of format " ^ f
  | Local_read -> "region local read"
  | Max_reduction -> "max reduction"
  | Region_program -> "region program"
  | Scan_read -> "scan read"
  | Select -> "select"
  | Unary_operation -> "unary operation"
  | Unmaterialized_source -> "load of an unmaterialized value"
  | Virtual_use -> "virtual use"

let pp fmt { at; construct } =
  Fmt.pf fmt "%a: %s is not lowered" Tensor_id.pp at (construct_name construct)
