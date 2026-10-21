(* OCaml encoding of c10::Device — a (DeviceType, index) pair. It crosses the C
   ABI as [struct atc_device] (atg_shim.h); the [device] / [device_opt] ctypes
   views in aten_operation_description.ml marshal it, a negative type tag encoding an
   absent [Device?]. [index] is the c10::DeviceIndex; -1 means unspecified (the
   current/only device of that type, e.g. plain "cpu"). *)

module Type : sig
  type t =
    | CPU
    | CUDA
    | FPGA
    | HIP
    | HPU
    | IDEEP
    | IPU
    | Lazy
    | MAIA
    | MKLDNN
    | Meta
    | Metal
    | MPS
    | MTIA
    | OPENCL
    | OPENGL
    | PrivateUse1
    | VE
    | Vulkan
    | XLA
    | XPU

  val to_int : t -> int
  (** The c10 numeric tag; the table follows c10's ABI. *)

  val of_int : int -> t option
  (** [None] for a tag that names no known device type. *)
end

type t = { type_ : Type.t; index : int }

val make : ?index:int -> Type.t -> t
(** [index] defaults to [-1], the unspecified device. *)

val cpu : t
(** The host device of this CPU-only build. *)
