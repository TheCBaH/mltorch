(* Type descriptions for the ctypes stub generator.

   c10::Scalar is marshalled as the tagged POD [struct atc_scalar] (atg_shim.h).
   It must be declared HERE (through the TYPE functor) rather than with a bare
   [Ctypes.structure]: the stub generator then binds it to the header struct and
   passes it by value, instead of re-emitting a conflicting local definition.
   The [scalar] / [scalar_opt] views over it live in aten_operation_description.ml. *)
module Types (S : Ctypes.TYPE) = struct
  open S

  (* The opaque tensor handle ([struct atc_tensor_opaque*]). Declared here (never
     sealed — only ever used through a pointer) so [atc_tensor] in
     aten_function_description.ml and the struct fields below share one type. *)
  let tensor_opaque : [ `atc_tensor_opaque ] Ctypes.structure typ =
    structure "atc_tensor_opaque"

  (* The opaque Tensor[] result container ([struct atc_tensor_list_opaque*]),
     declared here for the same reason as [tensor_opaque] and likewise never
     sealed -- it is only ever used through a pointer. *)
  let tensor_list_opaque : [ `atc_tensor_list_opaque ] Ctypes.structure typ =
    structure "atc_tensor_list_opaque"

  (* The shared int/float payload (union atc_scalar_value). *)
  let scalar_value : [ `atc_scalar_value ] Ctypes.union typ =
    union "atc_scalar_value"

  let value_i = field scalar_value "i" int64_t
  let value_d = field scalar_value "d" double
  let value_b = field scalar_value "b" bool
  let () = seal scalar_value

  (* [tag] is C [enum atc_scalar_tag], which is int-sized; bind it as int. *)
  let scalar_struct : [ `atc_scalar ] Ctypes.structure typ =
    structure "atc_scalar"

  let scalar_tag = field scalar_struct "tag" int
  let scalar_v = field scalar_struct "v" scalar_value
  let () = seal scalar_struct

  (* c10::Device as a POD (atg_shim.h): [type] is the DeviceType code (negative
     = absent Device?), [index] the DeviceIndex. Declared here so the stub
     generator binds it to the header struct rather than re-emitting it. *)
  let device_struct : [ `atc_device ] Ctypes.structure typ =
    structure "atc_device"

  let device_type_code = field device_struct "type" int8_t
  let device_index = field device_struct "index" int8_t
  let () = seal device_struct

  (* Multi-output op result structs (atc_tensors2 / atc_tensors3), filled
     through an out-param. Fields are properly typed [ptr tensor_opaque] (=
     atc_tensor), so reading one yields a handle directly. *)
  let tensors2_struct : [ `atc_tensors2 ] Ctypes.structure typ =
    structure "atc_tensors2"

  let tensors2_v0 = field tensors2_struct "v0" (ptr tensor_opaque)
  let tensors2_v1 = field tensors2_struct "v1" (ptr tensor_opaque)
  let () = seal tensors2_struct

  let tensors3_struct : [ `atc_tensors3 ] Ctypes.structure typ =
    structure "atc_tensors3"

  let tensors3_v0 = field tensors3_struct "v0" (ptr tensor_opaque)
  let tensors3_v1 = field tensors3_struct "v1" (ptr tensor_opaque)
  let tensors3_v2 = field tensors3_struct "v2" (ptr tensor_opaque)
  let () = seal tensors3_struct
end
