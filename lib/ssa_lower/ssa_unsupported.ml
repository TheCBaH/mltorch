(* A construct lowering does not handle yet, named, so a refusal is a fact a
   test can assert and a sweep can tally. Anything outside the supported slice is
   refused with one of these and never half-lowered: a partially lowered program
   is a defect however plausible its output. A refusal is a verdict about the
   request, distinct from every runtime failure of a program that was lowered. *)
type construct =
  | Index_divisor
  | Index_literal
  | Load_format of string
  | Local_read
  | Region_admission
  | Region_program
  | Scan_read
  | Unmaterialized_source
  | Virtual_use

type t = { at : Tensor_id.t; construct : construct }

let construct_name = function
  | Index_divisor -> "index divisor that is not a positive index literal"
  | Index_literal -> "index literal outside the index domain"
  | Load_format f -> "load of format " ^ f
  | Local_read -> "region local read"
  | Region_admission -> "region program over its admission budget"
  | Region_program -> "region program"
  | Scan_read -> "scan read"
  | Unmaterialized_source -> "load of an unmaterialized value"
  | Virtual_use -> "virtual use"

let pp fmt { at; construct } =
  Fmt.pf fmt "%a: %s is not lowered" Tensor_id.pp at (construct_name construct)
