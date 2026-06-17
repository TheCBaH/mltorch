(* OCaml encoding of c10::Device — a (DeviceType, index) pair. It crosses the C
   ABI as [struct atc_device] (atg_shim.h); the [device] / [device_opt] ctypes
   views in operation_description.ml marshal it, a negative type tag encoding an
   absent [Device?]. [index] is the c10::DeviceIndex; -1 means unspecified (the
   current/only device of that type, e.g. plain "cpu"). *)
type t = { type_ : Device_type.t; index : int }

let make ?(index = -1) type_ = { type_; index }

(* The host device of this CPU-only build. *)
let cpu = { type_ = Device_type.CPU; index = -1 }
