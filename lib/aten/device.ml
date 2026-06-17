(* OCaml encoding of c10::Device — a (DeviceType, index) pair. It crosses the C
   ABI as [struct atc_device] (atg_shim.h); the [device] / [device_opt] ctypes
   views in operation_description.ml marshal it, a negative type tag encoding an
   absent [Device?]. [index] is the c10::DeviceIndex; -1 means unspecified (the
   current/only device of that type, e.g. plain "cpu"). *)

(* c10::DeviceType (torch/headeronly/core/DeviceType.h). *)
module Type = struct
  type t =
    | CPU
    | CUDA
    | MKLDNN
    | OPENGL
    | OPENCL
    | IDEEP
    | HIP
    | FPGA
    | MAIA
    | XLA
    | Vulkan
    | Metal
    | XPU
    | MPS
    | Meta
    | HPU
    | VE
    | Lazy
    | IPU
    | MTIA
    | PrivateUse1

  let to_int = function
    | CPU -> 0
    | CUDA -> 1
    | MKLDNN -> 2
    | OPENGL -> 3
    | OPENCL -> 4
    | IDEEP -> 5
    | HIP -> 6
    | FPGA -> 7
    | MAIA -> 8
    | XLA -> 9
    | Vulkan -> 10
    | Metal -> 11
    | XPU -> 12
    | MPS -> 13
    | Meta -> 14
    | HPU -> 15
    | VE -> 16
    | Lazy -> 17
    | IPU -> 18
    | MTIA -> 19
    | PrivateUse1 -> 20

  let of_int = function
    | 0 -> Some CPU
    | 1 -> Some CUDA
    | 2 -> Some MKLDNN
    | 3 -> Some OPENGL
    | 4 -> Some OPENCL
    | 5 -> Some IDEEP
    | 6 -> Some HIP
    | 7 -> Some FPGA
    | 8 -> Some MAIA
    | 9 -> Some XLA
    | 10 -> Some Vulkan
    | 11 -> Some Metal
    | 12 -> Some XPU
    | 13 -> Some MPS
    | 14 -> Some Meta
    | 15 -> Some HPU
    | 16 -> Some VE
    | 17 -> Some Lazy
    | 18 -> Some IPU
    | 19 -> Some MTIA
    | 20 -> Some PrivateUse1
    | _ -> None
end

type t = { type_ : Type.t; index : int }

let make ?(index = -1) type_ = { type_; index }

(* The host device of this CPU-only build. *)
let cpu = { type_ = Type.CPU; index = -1 }
