(* A construct lowering does not handle yet, named, so a refusal is a fact a
   test can assert and a sweep can tally. Anything outside the current slice is
   refused with one of these, never half-lowered: a partially lowered program is
   a defect however plausible its output. *)
type construct =
  | Gather_index
  | Int64_expression
  | Int64_value
  | Load_format of string
  | Local_read
  | Max_pool_intrinsic
  | Region_admission
  | Region_program
  | Unmaterialized_source
  | Virtual_use

type t = { at : Tensor_id.t; construct : construct }

let construct_name = function
  | Gather_index -> "gather index"
  | Int64_expression -> "int64 expression"
  | Int64_value -> "int64 value"
  | Load_format f -> "load of format " ^ f
  | Local_read -> "region local read"
  | Max_pool_intrinsic -> "max-pool intrinsic"
  | Region_admission -> "region program over its admission budget"
  | Region_program -> "region program"
  | Unmaterialized_source -> "load of an unmaterialized value"
  | Virtual_use -> "virtual use"

let pp fmt { at; construct } =
  Fmt.pf fmt "%a: %s is not lowered" Tensor_id.pp at (construct_name construct)
