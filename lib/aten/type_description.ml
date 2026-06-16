(* Type descriptions for the ctypes stub generator.

   c10::Scalar is marshalled as the tagged POD [struct atc_scalar] (atg_shim.h).
   It must be declared HERE (through the TYPE functor) rather than with a bare
   [Ctypes.structure]: the stub generator then binds it to the header struct and
   passes it by value, instead of re-emitting a conflicting local definition.
   The [scalar] / [scalar_opt] views over it live in operation_description.ml. *)
module Types (S : Ctypes.TYPE) = struct
  open S

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
end
