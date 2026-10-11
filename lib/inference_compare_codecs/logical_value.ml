open Inference_compare_schema

module Context = struct
  type t = {
    codecs : Element_codec.Registry.t;
    hashers : Payload_hasher.Registry.t;
    algorithm : Id.Digest_algorithm.t;
    ledger : Owned_blob.Ledger.t;
  }
end

type t = {
  spec : Tensor_spec.t;
  storage : Owned_blob.t;
  digest : Digest.Logical_value.t;
  elements : int;
}

let fail e = Err.fail ~pp_error:Codec_error.pp e
let ( let* ) = Err.bind

let lookup (ctx : Context.t) (spec : Tensor_spec.t) =
  match Element_codec.Registry.find ctx.codecs spec.dtype.codec with
  | Some c -> Ok c
  | None -> fail (Codec_error.Unknown_codec spec.dtype.codec)

(* The digest covers the canonical spec and the bytes, so equal bytes under
   two dtypes or shapes hash differently. *)
let digest_of_parts (ctx : Context.t) spec bytes =
  let header =
    Digest_preimage.of_json Logical_value (Tensor_spec.encode spec) ^ "\n"
  in
  let hasher_error _ = Codec_error.Hasher_unavailable ctx.algorithm in
  let run () =
    let ( let* ) = Err.bind in
    let* st = Payload_hasher.create ctx.hashers ctx.algorithm in
    let* () = Payload_hasher.update_string st header in
    (* Bounded chunks: the digest must not depend on how a payload is split,
       and a backend need not accept an arbitrarily large update. *)
    let chunk = 1 lsl 16 in
    let rec go pos =
      if pos >= String.length bytes then Ok ()
      else
        let len = min chunk (String.length bytes - pos) in
        let* () = Payload_hasher.update st bytes ~pos ~len in
        go (pos + len)
    in
    let* () = go 0 in
    Payload_hasher.finish st
  in
  match run () with
  | Error e -> fail (hasher_error (Err.Error.kind e))
  | Ok d -> (
      match
        Digest.Logical_value.make
          (Digest.Payload.algorithm d)
          ~hex:(Digest.Payload.hex d)
      with
      | Ok v -> Ok v
      | Error _ -> fail (Codec_error.Hasher_unavailable ctx.algorithm))

let make (ctx : Context.t) (spec : Tensor_spec.t) bytes =
  let* codec = lookup ctx spec in
  let* expected =
    Shape_math.byte_count ~width:codec.byte_width spec.logical_shape
  in
  if String.length bytes <> expected then
    fail
      (Codec_error.Byte_length_mismatch
         {
           expected = Int64.of_int expected;
           actual = Int64.of_int (String.length bytes);
         })
  else
    let* digest = digest_of_parts ctx spec bytes in
    let* storage = Owned_blob.create ctx.ledger bytes in
    Ok { spec; storage; digest; elements = expected / codec.byte_width }

let spec v = v.spec
let digest v = v.digest
let contents v = Owned_blob.contents v.storage
let storage v = v.storage
let element_count v = v.elements

let codec (ctx : Context.t) v =
  match lookup ctx v.spec with
  | Ok c -> c
  | Error _ -> invalid_arg "Logical_value.codec: unregistered codec"

let equal a b =
  Tensor_spec.equal a.spec b.spec
  && String.equal
       (Owned_blob.contents a.storage)
       (Owned_blob.contents b.storage)

let drop v = Owned_blob.drop v.storage

let of_integers ctx (spec : Tensor_spec.t) values =
  let* codec = lookup ctx spec in
  let n = List.length values in
  let b = Bytes.make (n * codec.byte_width) '\000' in
  List.iteri (fun i v -> Element_codec.set_bits codec b i v) values;
  make ctx spec (Bytes.unsafe_to_string b)

let of_floats ctx (spec : Tensor_spec.t) values =
  let* codec = lookup ctx spec in
  let n = List.length values in
  let b = Bytes.make (n * codec.byte_width) '\000' in
  List.iteri
    (fun i v ->
      let bits =
        match codec.element with
        | Float_binary32 -> Int64.of_int32 (Int32.bits_of_float v)
        | _ -> Int64.bits_of_float v
      in
      Element_codec.set_bits codec b i bits)
    values;
  make ctx spec (Bytes.unsafe_to_string b)
