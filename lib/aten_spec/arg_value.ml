(* The uniform value of a decoded op argument — the bridge currency between the
   per-op codecs (generated in aten_op_spec, which build these via each op's
   [to_args]) and the runner (which lowers them to a Pytorch_types.Argument and,
   for tensors, synthesizes the data). One constructor per supported
   Aten_op_config.param_type. *)

type t =
  | Tensor of Tensor_spec.t
  | Tensor_opt of Tensor_spec.t option
  | Tensor_list of Tensor_spec.t list
  | Int of int
  | Int_opt of int option
  | Int_list of int list
  | Int_list_opt of int list option
  | Float of Float32.t
  | Float_opt of Float32.t option
  | Bool of bool
  | Bool_opt of bool option
  | Scalar of Scalar_value.t
  | Scalar_opt of Scalar_value.t option
  | Str of string

(* Encode a decoded argument to generic JSON. The per-op codecs decode the typed
   form; this is the matching write side used by the Op_spec envelope. An absent
   optional encodes as JSON null. The shapes mirror each per-op field's codec
   exactly (a plain Int is a bare number, a Scalar is a tagged object, a Float is
   an int32-hex string), so decode (encode v) = v. *)
let to_json : t -> Jsont.json = function
  | Tensor t -> Spec_util.enc Tensor_spec.jsont t
  | Tensor_opt None -> Spec_util.jnull
  | Tensor_opt (Some t) -> Spec_util.enc Tensor_spec.jsont t
  | Tensor_list ts ->
      Spec_util.jarr (List.map (Spec_util.enc Tensor_spec.jsont) ts)
  | Int i -> Spec_util.jint i
  | Int_opt None -> Spec_util.jnull
  | Int_opt (Some i) -> Spec_util.jint i
  | Int_list xs -> Spec_util.jarr (List.map Spec_util.jint xs)
  | Int_list_opt None -> Spec_util.jnull
  | Int_list_opt (Some xs) -> Spec_util.jarr (List.map Spec_util.jint xs)
  | Float f -> Spec_util.enc Float32.jsont f
  | Float_opt None -> Spec_util.jnull
  | Float_opt (Some f) -> Spec_util.enc Float32.jsont f
  | Bool b -> Spec_util.jbool b
  | Bool_opt None -> Spec_util.jnull
  | Bool_opt (Some b) -> Spec_util.jbool b
  | Scalar sv -> Spec_util.enc Scalar_value.jsont sv
  | Scalar_opt None -> Spec_util.jnull
  | Scalar_opt (Some sv) -> Spec_util.enc Scalar_value.jsont sv
  | Str s -> Spec_util.jstr s
