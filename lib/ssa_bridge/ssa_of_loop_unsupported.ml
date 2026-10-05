(* A Loop construct the converter does not cover, named so a refusal is a fact a
   test can assert. The converter never half-converts: anything outside the
   covered slice (in particular a guard, whose evaluation site the SSA form
   must reproduce) is refused. *)
type construct =
  | Alloc
  | Argmax_pair
  | Array
  | Charge_scan
  | Float_max
  | Fma
  | Guard
  | I64
  | If
  | Index_literal
  | Index_operation
  | Load_format of string
  | Meter
  | Scan_state
  | Select
  | Unary
  | Unassigned_temp

type t = { construct : construct }

let construct_name = function
  | Alloc -> "array allocation"
  | Argmax_pair -> "paired argmax update"
  | Array -> "array access"
  | Charge_scan -> "scan charge"
  | Float_max -> "float max"
  | Fma -> "fused multiply-add"
  | Guard -> "guard"
  | I64 -> "int64 value"
  | If -> "if"
  | Index_literal -> "index literal outside the index domain"
  | Index_operation -> "min, max, clamp or division on an index"
  | Load_format f -> "load of format " ^ f
  | Meter -> "scan meter"
  | Scan_state -> "scan state"
  | Select -> "select"
  | Unary -> "unary operation"
  | Unassigned_temp -> "read of an unassigned temporary"

let pp fmt { construct } =
  Fmt.pf fmt "%s is not converted" (construct_name construct)
