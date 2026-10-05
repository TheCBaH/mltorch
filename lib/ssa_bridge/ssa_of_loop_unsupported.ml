(* A Loop construct the converter does not cover, named so a refusal is a fact a
   test can assert. The converter never half-converts: anything outside the
   covered slice (in particular a guard whose meaning it has no recipe for, whose
   evaluation site the SSA form must reproduce) is refused. *)
type construct =
  | Alloc
  | Array
  | Charge_scan
  | Fma
  | Guard
  | I64
  | Index_divisor
  | Index_literal
  | Load_format of string
  | Meter
  | Scan_state
  | Unassigned_temp

type t = { construct : construct }

let construct_name = function
  | Alloc -> "array allocation"
  | Array -> "array access"
  | Charge_scan -> "scan charge"
  | Fma -> "fused multiply-add"
  | Guard -> "guard"
  | I64 -> "int64 value"
  | Index_divisor -> "index divisor that is not a positive index literal"
  | Index_literal -> "index literal outside the index domain"
  | Load_format f -> "load of format " ^ f
  | Meter -> "scan meter"
  | Scan_state -> "scan state"
  | Unassigned_temp -> "read of an unassigned temporary"

let pp fmt { construct } =
  Fmt.pf fmt "%s is not converted" (construct_name construct)
