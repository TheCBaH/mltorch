open Inference_compare_schema

module Operation = struct
  type t = {
    name : Id.Codec_operation.t;
    implementation_digest : Digest.Implementation.t;
    arity : int;
    lossless : bool;
    infer : Tensor_spec.t list -> (Tensor_spec.t, Codec_error.t) result;
    apply :
      Logical_value.Context.t ->
      Logical_value.t list ->
      (Logical_value.t, Codec_error.t) Err.t;
  }
end

type t = Operation.t list

let empty = []

let find t name =
  List.find_opt
    (fun (o : Operation.t) -> Id.Codec_operation.equal o.name name)
    t

let register t (o : Operation.t) =
  match find t o.name with
  | Some _ -> Err.fail (`Duplicate_operation o.name)
  | None -> Ok (t @ [ o ])

let capabilities t =
  List.map
    (fun (o : Operation.t) ->
      {
        Capability.Operation.operation = o.name;
        implementation_digest = o.implementation_digest;
        arity = o.arity;
        lossless = (if o.lossless then Supported else Unsupported);
      })
    t
